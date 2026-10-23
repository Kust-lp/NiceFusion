# -*- coding: utf-8 -*-
import argparse
import os
import sys
from collections import defaultdict

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from tqdm import tqdm
from model.TModel import *
from utils.dataset import *
from utils.loss import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def Emain():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="../datasets/TVNF/train/imgsize_128_stride_200.h5")
    parser.add_argument('--max_patience', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint', type=str, default="../ckpt_TM")
    args = parser.parse_args()

    train_loader = DataLoader(
        H5Dataset(args.dataset),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=False,
        pin_memory=True
    )

    model =FusionM().to(args.device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_in_MB = total_params * 4 / (1024 * 1024)
    print(f"Trainable parameters: {total_params / 1e6:.2f}M ({size_in_MB:.2f} MB)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * 1  # 约 1 个 epoch 或 1000 iter
    warmup = LinearLR(opt, start_factor=0.001, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(opt, T_max=(total_steps - warmup_steps), eta_min=1e-6)
    scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])
    ema = ModelEmaV2(model, decay=0.999)

    loss_fn = OracleFusionLoss().to(args.device)
    os.makedirs(args.checkpoint, exist_ok=True)

    for epoch in range(1,args.epochs+1):
        model.train()
        avg_loss = defaultdict(float)
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}") as pbar:
            for VIS, NIR, TIR in train_loader:
                VIS = VIS.to(args.device)
                NIR = NIR.to(args.device)
                TIR = TIR.to(args.device)

                opt.zero_grad(set_to_none=True)
                fu, _, _ = model(VIS, TIR, NIR)
                L, info = loss_fn([VIS, TIR, NIR], fu)

                L.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ema.update(model)
                scheduler.step()

                for k, v in info.items():
                    avg_loss[k] += v
                pbar.set_postfix({k: v / (pbar.n + 1) for k, v in avg_loss.items()})
                pbar.update(1)

        eval_model = ema.module
        torch.save(eval_model.state_dict(), f"{args.checkpoint}/TNV_{epoch}.pth")



if __name__ == "__main__":
    Emain()
