import argparse
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from timm.utils import ModelEmaV2
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.Missmodel_train import missFusionM
from model.TModel  import FusionM
from utils.dataset import H5Dataset
from utils.loss import MissFusionLoss

import warnings
warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def load_basemodel(ckpt, device):
    model = FusionM().to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def build_optimizer_and_scheduler(model, lr, weight_decay, steps_per_epoch, num_epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

    total_steps = max(1, steps_per_epoch * num_epochs)
    warmup_steps = min(steps_per_epoch, total_steps)
    if total_steps == warmup_steps:
        scheduler = LinearLR(optimizer, start_factor=0.001, total_iters=warmup_steps)
    else:
        warmup = LinearLR(optimizer, start_factor=0.001, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    return optimizer, scheduler


def Stage1(args, model, FM, loss_fn, train_loader, device):
        model.freeze_adv()
        model.freeze_enc()
        ema = ModelEmaV2(model, decay=0.999)
        model.train()

        epochs = args.Wp
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"stage1 trainable params: {trainable / 1e6:.2f}M")

        opt, scheduler = build_optimizer_and_scheduler(
            model, args.lr, args.weight_decay, len(train_loader), epochs
        )

        total_steps = len(train_loader) * epochs
        warmup_steps_tau = int(total_steps * 0.1)
        global_step = 0
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            num_batches = 0
            avg_losses = defaultdict(float)

            if epoch >10:
                model.Dec.F4M.wp = False
                model.Dec.F3M.wp = False
                model.Dec.F2M.wp = False
                model.Dec.F1M.wp = False

            pbar = tqdm(train_loader, desc=f"model_warmup-Epoch{epoch}/{epochs}")
            for VIS, NIR, TIR in pbar:
                VIS = VIS.to(device)
                NIR = NIR.to(device)
                TIR = TIR.to(device)
                opt.zero_grad(set_to_none=True)

                with torch.no_grad():
                    y_plus, f0_plus, _ = FM(VIS, TIR, NIR)

                (
                    yhat, f,
                    alpha, beta, mu,
                    _, _,
                    sub1, sub2, sub3, sub4
                ) = model(VIS, TIR)
                tau = min(1.0, global_step / warmup_steps_tau)


                loss, info = loss_fn.alll(
                    yhat, y_plus, f0_plus, f,
                    sub1, sub2, sub3, sub4,
                    mu, alpha, beta, tau
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()
                ema.update(model)

                epoch_loss += loss.item()
                num_batches += 1
                for k, v in info.items():
                    avg_losses[k] += v
                pshow = {k: round(v / num_batches, 4) for k, v in avg_losses.items()}
                pshow["tau"] = tau
                pbar.set_postfix(pshow)
                global_step += 1

        return model


def build_optimizer_and_scheduler_from_params(params, lr, weight_decay, steps_per_epoch, num_epochs,start_factor=0.001):
    params = list(params)
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

    total_steps = max(1, steps_per_epoch * num_epochs)
    warmup_steps = min(steps_per_epoch, total_steps)
    if total_steps == warmup_steps:
        scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_steps)
    else:
        warmup = LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    return optimizer, scheduler

def split_stage2_params(model):
    adv_params = list(model.advModel.parameters())
    main_params = list(model.evi_head.parameters())
    main_params += list(model.Dec.parameters())
    main_params += list(model.fuse_head.parameters())

    return main_params, adv_params

def set_requires_grad(params, flag: bool):
    for p in params:
        p.requires_grad = flag


def stage2_train(args, model, FM, loss_fn, train_loader, device):
    model.freeze_all()
    main_params, adv_params = split_stage2_params(model)
    print(f" main params: {sum(p.numel() for p in main_params) / 1e6:.2f}M")
    print(f"adv  params: {sum(p.numel() for p in adv_params) / 1e6:.2f}M")

    for stage, epochs in zip(["adv_warm", "exchange_up"],[args.Adv_Wp, args.Epochs]):

        ema = ModelEmaV2(model, decay=0.999)
        if stage == "adv_warm":
            opt_adv, sch_adv = build_optimizer_and_scheduler_from_params(
                adv_params, args.lr, args.weight_decay, len(train_loader), epochs
            )
            model.train()
            set_requires_grad(main_params, False)
            set_requires_grad(adv_params, True)
            for epoch in range(1, epochs + 1):
                avg_losses = defaultdict(float)
                num_batches = 0
                pbar = tqdm(train_loader, desc=f"adv_warmup-Ep{epoch}/{epochs}")
                for VIS, NIR, TIR in pbar:
                    VIS = VIS.to(device)
                    NIR = NIR.to(device)
                    TIR = TIR.to(device)
                    with torch.no_grad():
                        y_plus, _, _ = FM(VIS, TIR, NIR)
                        y_safe, _, _ = FM(VIS, TIR)

                    opt_adv.zero_grad(set_to_none=True)

                    (
                        y_adv, f,
                        _, _, _,
                        u_c, u_mu,
                        _, _, _, _
                    ) = model(VIS, TIR, apply_adv=True)
                    loss_adv, info = loss_fn.adv_loss(
                        y_adv, y_plus, y_safe,
                        u_c, u_mu
                    )
                    loss_adv.backward()
                    torch.nn.utils.clip_grad_norm_(adv_params, 1.0)
                    opt_adv.step()
                    sch_adv.step()
                    ema.update(model)

                    num_batches += 1
                    for k, v in info.items():
                        avg_losses[k] += v
                    pshow = {k: round(v / num_batches, 4) for k, v in avg_losses.items()}
                    pbar.set_postfix(pshow)

        else:

            opt_main, sch_main = build_optimizer_and_scheduler_from_params(
                main_params, args.lr, args.weight_decay, len(train_loader), epochs, start_factor=0.1
            )
            opt_adv, sch_adv = build_optimizer_and_scheduler_from_params(
                adv_params, args.lr, args.weight_decay, len(train_loader), epochs, start_factor=0.1
            )


            ema = ModelEmaV2(model, decay=0.999)
            for epoch in range(1, epochs + 1):
                model.train()
                avg_losses = defaultdict(float)
                num_batches = 0

                pbar = tqdm(train_loader, desc=f"Ep{epoch}/{epochs}")
                for VIS, NIR, TIR in pbar:
                    VIS = VIS.to(device)
                    NIR = NIR.to(device)
                    TIR = TIR.to(device)
                    with torch.no_grad():
                        y_plus, f0_plus, _ = FM(VIS, TIR, NIR)
                        y_safe, f0_safe, _ = FM(VIS, TIR)

                    # (A) clean forward
                    set_requires_grad(main_params, True)
                    set_requires_grad(adv_params, False)
                    opt_main.zero_grad(set_to_none=True)
                    (
                        y_clean, f,
                        alpha, beta, mu,
                        _, _,
                        sub1, sub2, sub3, sub4
                    ) = model(VIS, TIR, apply_adv=False)

                    loss_clean, info1 = loss_fn.alll(
                        y_clean, y_plus, f0_plus, f,
                        sub1, sub2, sub3, sub4,
                        mu, alpha, beta, 1.
                    )

                    loss_clean.backward()
                    torch.nn.utils.clip_grad_norm_(main_params, 1.0)
                    opt_main.step()



                    # -----------------------------
                    # (B) adversary step: maximize risk under budgets
                    # -----------------------------
                    set_requires_grad(main_params, False)
                    set_requires_grad(adv_params, True)
                    opt_adv.zero_grad(set_to_none=True)

                    (
                        y_adv, f,
                        _, _, _,
                        u_c, u_mu,
                        _, _, _, _
                    ) = model(VIS, TIR, apply_adv=True)
                    loss_adv, info2 = loss_fn.adv_loss(
                        y_adv, y_plus, y_safe,
                        u_c, u_mu
                    )
                    info2.update(info1)
                    loss_adv.backward()
                    torch.nn.utils.clip_grad_norm_(adv_params, 1.0)
                    opt_adv.step()
                    sch_adv.step()



                    # -----------------------------
                    # (C) main model step: minimize attacked risk
                    # -----------------------------

                    set_requires_grad(main_params, True)
                    set_requires_grad(adv_params, False)
                    opt_main.zero_grad(set_to_none=True)
                    model.freeze_evi()
                    (
                        y_adv, f,
                        _, _, _,
                        u_c, u_mu,
                        sub1, sub2, sub3, sub4
                    ) = model(VIS, TIR, apply_adv=True)

                    with torch.no_grad():
                        (
                            y_clean,
                            _, _, _, _,
                            _, _, _, _, _, _,
                        ) = model(VIS, TIR, apply_adv=False)


                    loss_main, info= loss_fn.model_loss(
                        y_clean, y_adv, y_plus,
                        sub1, sub2, sub3, sub4,
                    )

                    info.update(info2)
                    loss_main.backward()
                    torch.nn.utils.clip_grad_norm_(main_params, 1.0)
                    opt_main.step()
                    sch_main.step()
                    ema.update(model)

                    num_batches += 1
                    for k, v in info.items():
                        avg_losses[k] += v
                    pshow = {k: round(v / num_batches, 4) for k, v in avg_losses.items()}
                    pbar.set_postfix(pshow)

                ckpt_path = os.path.join(args.ckpt_dir, f"ep{epoch}.pth")
                torch.save(ema.module.state_dict(), ckpt_path)


def main():
    parser = argparse.ArgumentParser(description="Fusion-Centric Training for NIR-Missing")
    parser.add_argument('--dataset', type=str, default='../datasets/TVNF/train/imgsize_128_stride_200.h5')
    parser.add_argument('--Wp', type=int, default=20)
    parser.add_argument('--Adv_Wp', type=int, default=10)
    parser.add_argument('--Epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--full_ckpt', type=str, default='../ckpt_TNV/TNV_120.pth')
    parser.add_argument('--subspace', type=str, default='../ckpt_TNV/Subspaces(32,4).pt')
    parser.add_argument('--ckpt_dir', type=str, default='../ckpt')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print('=== Fusion-Centric Training ===')

    train_loader = DataLoader(
        H5Dataset(args.dataset),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    FM = load_basemodel(args.full_ckpt, device)

    model = missFusionM(learnable_bank=True, Evi_adv=True).to(device)
    model.load_state_dict(torch.load(args.full_ckpt, map_location=device), strict=False)

    payload = torch.load(args.subspace, map_location=device)


    model.Dec.F4M.bank.load_bank(payload["c4"], payload["b4"])
    model.Dec.F3M.bank.load_bank(payload["c3"], payload["b3"])
    model.Dec.F2M.bank.load_bank(payload["c2"], payload["b2"])
    model.Dec.F1M.bank.load_bank(payload["c1"], payload["b1"])

    loss_fn = MissFusionLoss(payload).to(device)

    model = Stage1(args, model, FM, loss_fn, train_loader, device)

    stage2_train(args, model, FM, loss_fn, train_loader, device)



if __name__ == '__main__':
    main()
