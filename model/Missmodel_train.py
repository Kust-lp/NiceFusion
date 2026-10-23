import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from TModel import *
from EviAdversary import EvidenceAdversary



def topk_simplex(pi: torch.Tensor, k: int = 2):
    """
    pi: [B, M, H, W], sums to 1 along dim=1.
    Keep only top-k modes and renormalize.
    """
    if k >= pi.shape[1]:
        return pi

    _, topi = torch.topk(pi, k=k, dim=1)
    mask = torch.zeros_like(pi).scatter_(1, topi, 1.0)
    pi_sparse = pi * mask
    pi_sparse = pi_sparse / (pi_sparse.sum(dim=1, keepdim=True) + 1e-6)
    return pi_sparse

class EvidenceHead(nn.Module):
    def __init__(self, c, head=8, ffn_expansion_factor=2,
                 bias=False, LayerNorm_type='WithBias'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            TransformerBlock(dim=c, num_heads=head,
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type),
            nn.Conv2d(c, c//2, 3, 1, 1),
            TransformerBlock(dim=c//2, num_heads=head,
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type),
            nn.Conv2d(c//2, 2, 3, 1, 1),
        )

    def forward(self, x):
        out = self.net(x)
        alpha = F.softplus(out[:,:1,:,:])+1
        beta = F.softplus(out[:,1:,:,:])+1
        S = alpha + beta
        mu = alpha / S
        return mu, alpha, beta, S


class CoordHead(nn.Module):
    def __init__(self, c, num_modes, rank, head=8, ffn_expansion_factor=2,
                 bias=False, LayerNorm_type='WithBias'):
        super().__init__()
        self.num_modes = num_modes
        self.rank = rank
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            TransformerBlock(dim=c, num_heads=head,
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type),
            # TransformerBlock(dim=c, num_heads=head,
            #                  ffn_expansion_factor=ffn_expansion_factor,
            #                  bias=bias, LayerNorm_type=LayerNorm_type),
            nn.Conv2d(c, num_modes * rank, 3, 1, 1),
        )

    def forward(self, x):
        z = self.net(x)
        B, _, H, W = z.shape
        z = z.view(B, self.num_modes, self.rank, H, W)
        return z




# =========================================================
# affine subspace bank
# =========================================================

class AffineSubspaceBank(nn.Module):
    """
    centers           : fixed execution centers (mode anchors)
    bases             : learnable execution bases (except null mode)
    centers_anchor    : fixed anchor centers for supervision only
    bases_anchor      : fixed anchor bases for supervision only

    Mode 0 is reserved for null action and kept fixed at zero.
    """
    def __init__(self, c, mode_num, mode_dim, learn_bases=True):
        super().__init__()
        self.mode_num = mode_num
        self.mode_dim = mode_dim
        self.learn_bases = learn_bases

        centers = torch.zeros(mode_num, c)
        bases = torch.zeros(mode_num, c, mode_dim)
        self.register_buffer("centers", centers.clone(), persistent=True)
        self.center_offset = nn.Parameter(torch.zeros_like(centers))

        if learn_bases:
            self.bases = nn.Parameter(bases.clone())
        else:
            self.register_buffer("bases", bases.clone(), persistent=True)

    @torch.no_grad()
    def load_bank(self, centers: torch.Tensor, bases: torch.Tensor):

        device = self.centers.device
        dtype = self.centers.dtype
        centers = centers.to(device=device, dtype=dtype)
        bases = bases.to(device=device, dtype=dtype)

        # fixed execution anchors
        self.centers.copy_(centers)
        if isinstance(self.bases, nn.Parameter):
            self.bases.data.copy_(bases)
        else:
            self.bases.copy_(bases)

    def forward_topk(self, pi, z):

        B, M, H, W = pi.shape
        R = z.shape[2]
        C = self.centers.shape[1]
        ccenter = self.centers + self.center_offset * 0.1
        # ccenter = self.centers

        mask = (pi != 0)  # [B, M, H, W]

        k = int(mask.sum(dim=1).max().item())
        idx = torch.nonzero(mask.permute(0, 2, 3, 1), as_tuple=False)  # [B*H*W*k, 4]
        topi = idx[:, 3].view(B, H, W, k).permute(0, 3, 1, 2).contiguous()  # [B, K, H, W]

        w = torch.gather(pi, 1, topi)  # [B, K, H, W]
        w = w / (w.sum(dim=1, keepdim=True) + 1e-6)

        idx_z = topi.unsqueeze(2).expand(-1, -1, R, -1, -1)  # [B, K, R, H, W]
        z_sel = torch.gather(z, 1, idx_z)  # [B, K, R, H, W]

        idx_hwk = topi.permute(0, 2, 3, 1).contiguous()  # [B, H, W, K]

        centers_sel = F.embedding(idx_hwk, ccenter)  # [B, H, W, K, C]
        centers_sel = centers_sel.permute(0, 3, 4, 1, 2).contiguous()  # [B, K, C, H, W]

        bases_flat = self.bases.reshape(self.mode_num, -1)  # [M, C*R]
        bases_sel = F.embedding(idx_hwk, bases_flat)  # [B, H, W, K, C*R]
        bases_sel = bases_sel.view(B, H, W, k, C, R).permute(0, 3, 4, 5, 1, 2).contiguous()
        # [B, K, C, R, H, W]

        local = centers_sel + torch.einsum("bkcrhw,bkrhw->bkchw", bases_sel, z_sel)
        delta = torch.einsum("bkhw,bkchw->bchw", w, local)
        return delta

    def forward(self, pi, z):
        ccenter = self.centers + self.center_offset * 0.1
        # ccenter = self.centers
        if self.training:
            local = ccenter[None, :, :, None, None] + torch.einsum(
                "mcr,bmrhw->bmchw", self.bases, z
            )
            delta = torch.einsum("bmhw,bmchw->bchw", pi, local)
        else:
            delta = self.forward_topk(pi, z)

        return delta

    def basis_regularization_balance(self, pi, eps=1e-6):

        M = pi.shape[1]
        usage = pi.mean(dim=(0, 2, 3))
        usage = usage / (usage.sum() + eps)

        target = torch.full_like(usage, 1.0 / M)
        balance = (usage * ((usage + eps).log() - (target + eps).log())).sum()

        bt_b = torch.einsum("mcr,mcs->mrs", self.bases, self.bases)
        eye = torch.eye(self.mode_dim, device= self.bases.device, dtype= self.bases.dtype)[None]
        orth = ((bt_b - eye) ** 2).mean()
        return balance + orth



class ModeRouter(nn.Module):
    def __init__(self, c, num_modes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(c, num_modes, 1, 1, 0),
        )

    def forward(self, x):
        logits = self.net(x)
        pi = torch.softmax(logits, dim=1)
        return logits, pi

class RiskFiLM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c+1, c, 1),
            nn.GELU(),
            nn.Conv2d(c, c, 1),
        )
        self.gamma_head = nn.Conv2d(c, c, 1)
        self.beta_head = nn.Conv2d(c, c, 1)
    def forward(self, f,mu):
        f = self.net(torch.cat([f, mu], dim=1))
        gamma = self.gamma_head(f)
        beta = self.beta_head(f)
        return torch.tanh(gamma), torch.tanh(beta)
        # return gamma, beta


def randomk_simplex(pi: torch.Tensor, k: int = 2, retain_topk_weights: bool = False):

    M = pi.shape[1]
    if not 1 <= k <= M:
        raise ValueError(f"k must be between 1 and {M}, got {k}")
    if k == M:
        return pi if retain_topk_weights else torch.full_like(pi, 1.0 / M)

    selected_idx = torch.rand_like(pi).topk(k, dim=1).indices

    weights = torch.full_like(selected_idx, 1.0 / k, dtype=pi.dtype)
    return torch.zeros_like(pi).scatter_(1, selected_idx, weights)

class FCModule(nn.Module):
    def __init__(self, c, num_modes, rank, k, learnable_bank=True, head=8, ffn_expansion_factor=2,
                 bias=False, LayerNorm_type='WithBias'):
        super().__init__()

        self.topk = k
        self.wp = True
        self.mode_head = ModeRouter(c, num_modes)

        self.coord_head = CoordHead(c, num_modes, rank, head=head,
                                    ffn_expansion_factor=ffn_expansion_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type)
        self.risk_head = RiskFiLM(c)
        self.bank = AffineSubspaceBank(c, num_modes, rank, learn_bases=learnable_bank)

    def forward(self, f, mu, c):

        z = self.coord_head(f)

        _, pi = self.mode_head(f)



        if self.wp:
            pi_use = pi
        else:
            pi_use = topk_simplex(pi, self.topk)

        delta_hat = self.bank(pi_use, z)

        g, b = self.risk_head(delta_hat, mu)
        # g, b = self.risk_head(f, mu)

        delta_hat = c * (g * delta_hat + b)


        f_corr = f + delta_hat

        l_sub = self.bank.basis_regularization_balance(pi)


        return f_corr, pi, l_sub


class FusionDecoder(nn.Module):
    def __init__(self, channel, num_blocks, head, ffn_expansion_factor, bias, LayerNorm_type, num_modes, rank, topk, learnable_bank, Evi_adv):
        super().__init__()

        self.Dec = nn.ModuleList(
            [TransformerBlock(dim=channel, num_heads=head, ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks)])



        self.F4M = FCModule(
            c=channel,
            num_modes=num_modes,
            rank=rank,
            k=topk,
            learnable_bank=learnable_bank,
            head=head,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type

        )
        self.F3M = FCModule(
            c=channel,
            num_modes=num_modes,
            rank=rank,
            k=topk,
            learnable_bank=learnable_bank,
            head=head,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type

        )
        self.F2M = FCModule(
            c=channel,
            num_modes=num_modes,
            rank=rank,
            k=topk,
            learnable_bank=learnable_bank,
            head=head,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type

        )

        self.F1M = FCModule(
            c=channel,
            num_modes=num_modes,
            rank=rank,
            k=topk,
            learnable_bank=learnable_bank,
            head=head,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type

        )

    # def forward(self, f, mu, c):
    #
    #     f1 = self.Dec[0](f)
    #     f1_corr, pi1, sub1 = self.F1M(f1, mu, c)
    #
    #     f2 = self.Dec[1](f1_corr)
    #     f3_corr = self.Dec[2](f2)
    #
    #     f4 = self.Dec[3](f3_corr)
    #     f4_corr, pi4, sub4 = self.F4M(f4, mu, c)
    #
    #     return (
    #         f1_corr, f4_corr,
    #         pi1,  pi4,
    #         sub1,  sub4
    #     )

    def forward(self, f, mu, c):

        f1 = self.Dec[0](f)
        f1_corr, pi1, sub1 = self.F1M(f1, mu, c)

        f2 = self.Dec[1](f1_corr)
        f2_corr, pi2, sub2 = self.F2M(f2, mu, c)

        f3 = self.Dec[2](f2_corr)
        f3_corr, pi3, sub3 = self.F3M(f3, mu, c)

        f4 = self.Dec[3](f3_corr)
        f4_corr, pi4, sub4 = self.F4M(f4, mu, c)

        return (
            f1_corr, f2_corr, f3_corr, f4_corr,
            pi1, pi2, pi3, pi4,
            sub1, sub2, sub3, sub4
        )


class missFusionM(nn.Module):
    def __init__(self, channel=48, num_blocks=4,
                 ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias',
                 head=8, num_modes=32, rank=4, topk=2, tau_c=0.2, rho=0.15, learnable_bank=True, Evi_adv=False):
        super().__init__()
        self.tau_c = tau_c

        self.Enc = Encoder(channel, num_blocks, ffn_expansion_factor, bias, LayerNorm_type, head)

        self.evi_head = EvidenceHead(channel, head=head,
                                     ffn_expansion_factor=ffn_expansion_factor,
                                     bias=bias, LayerNorm_type=LayerNorm_type)

        self.advModel = EvidenceAdversary(channel, head=head,
                                          ffn_expansion_factor=ffn_expansion_factor,
                                          bias=bias, LayerNorm_type=LayerNorm_type,
                                          rho=rho) if Evi_adv else nn.Identity()


        self.Dec = FusionDecoder(channel, num_blocks, head, ffn_expansion_factor, bias, LayerNorm_type, num_modes, rank, topk, learnable_bank, Evi_adv)
        self.fuse_head = outHead(in_channels=channel, mode="sigmoid" )

    def forward(self, v, t, apply_adv=False):
        v0 = v
        v= self.Enc(v)
        t= self.Enc(t)
        f = (v + t)/2

        mu, alpha, beta, S = self.evi_head(f)
        c = 1.0 - torch.exp(-self.tau_c * torch.clamp(S-2, min=0.0) )

        if apply_adv:
            u_c, u_mu = self.advModel(f, mu, c)
            mu_adv = torch.sigmoid(torch.logit(mu.clamp(1e-4, 1 - 1e-4)) + u_mu)
            c_adv = torch.sigmoid(torch.logit(c.clamp(1e-4, 1 - 1e-4)) + u_c)
        else:
            u_c = torch.zeros_like(c)
            u_mu = torch.zeros_like(mu)
            mu_adv, c_adv = mu, c


        (
            f1_corr, f2_corr, f3_corr, f4_corr,
            pi1, pi2, pi3, pi4,
            sub1, sub2, sub3, sub4
        ) = self.Dec(f, mu_adv, c_adv)



        fuse = self.fuse_head(f4_corr, v0)


        return (
            fuse, f,
            alpha, beta, mu,
            u_c, u_mu,
            sub1, sub2, sub3, sub4
        )


    def forward_pi(self, v, t, apply_adv=False):
        v= self.Enc(v)
        t= self.Enc(t)
        f = (v + t)/2

        mu, alpha, beta, S = self.evi_head(f)
        c = 1.0 - torch.exp(-self.tau_c * torch.clamp(S-2, min=0.0) )

        if apply_adv:
            u_c, u_mu = self.advModel(f, mu, c)
            mu_adv = torch.sigmoid(torch.logit(mu.clamp(1e-4, 1 - 1e-4)) + u_mu)
            c_adv = torch.sigmoid(torch.logit(c.clamp(1e-4, 1 - 1e-4)) + u_c)
        else:
            u_c = torch.zeros_like(c)
            u_mu = torch.zeros_like(mu)
            mu_adv, c_adv = mu, c


        (
            f1_corr, f2_corr, f3_corr, f4_corr,
            pi1, pi2, pi3, pi4,
            sub1, sub2, sub3, sub4
        ) = self.Dec(f, mu_adv, c_adv)

        return (
            pi1, pi2, pi3, pi4
        )


    def freeze_adv(self):
        for p in self.parameters():
            p.requires_grad = True
        for p in self.advModel.parameters():
            p.requires_grad = False

    def freeze_enc(self):
        for p in self.Enc.parameters():
            p.requires_grad = False

    def freeze_evi(self):
        for p in self.evi_head.parameters():
            p.requires_grad = False

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False





