import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from TModel import *

class EvidenceAdversary(nn.Module):
    def __init__(self, c, head=8, ffn_expansion_factor=2,
                 bias=False, LayerNorm_type='WithBias', rho=0.5):
        super().__init__()
        self.rho = float(rho)

        in_ch = c + 2

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, 1, 1, bias=bias),
            TransformerBlock(dim=c, num_heads=head,
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type),
            nn.Conv2d(c, c // 2, 3, 1, 1),
            TransformerBlock(dim=c // 2, num_heads=head,
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type),
            nn.Conv2d(c // 2, 2, 3, 1, 1),
        )

    @staticmethod
    def _energy_clip(u_c, u_mu, rho: float):
        if rho is None or rho <= 0:
            return u_c, u_mu
        energy = (u_c.square() + u_mu.square()).mean(dim=(1, 2, 3), keepdim=True)
        rho_t = torch.tensor(rho, device=energy.device, dtype=energy.dtype)
        scale = torch.sqrt(torch.clamp(rho_t / (energy + 1e-12), max=1.0))
        return u_c * scale, u_mu * scale

    def forward(self,f, mu, c):
        inp = torch.cat([f, mu, c], dim=1)
        out = torch.tanh(self.net(inp))
        u_mu = out[:,:1,:,:]
        u_c  = out[:,1:,:,:]

        u_c, u_mu = self._energy_clip(u_c, u_mu, self.rho)

        return u_c, u_mu