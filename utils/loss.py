from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sobelxy(nn.Module):
    def __init__(self):
        super().__init__()
        kernelx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernely = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('weightx', kernelx)
        self.register_buffer('weighty', kernely)

    def forward(self, x):
        gx = F.conv2d(x, self.weightx, padding=1)
        gy = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(gx) + torch.abs(gy)

def mode_soft_ce(logits, target_q):
    logp = F.log_softmax(logits, dim=1)
    return -(target_q * logp).sum(dim=1).mean()

@torch.no_grad()
def repair_strength_target_from_delta(delta_true, scale: float = 0.05):
    mag = delta_true.abs().mean(dim=1, keepdim=True)   # [B,1,H,W]
    mu_t = 1.0 - torch.exp(-mag /scale)
    return mu_t.clamp(0.0, 1.0)



class EvidentialBetaLoss(nn.Module):

    def __init__(self, mode = "Beta"): #"expected"
        super().__init__()
        self.mode = mode

    @staticmethod
    def _expected_log_loss(target: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:

        S = alpha + beta
        digamma_S = torch.digamma(S)
        loss = target * (digamma_S - torch.digamma(alpha)) + (1.0 - target) * (digamma_S - torch.digamma(beta))
        return loss  # (B,C)

    @staticmethod
    def beta_nll(target, alpha, beta, eps=1e-4):
        t = target.clamp(eps, 1.0 - eps)
        S = alpha + beta
        # -log BetaPDF(t | alpha, beta)
        nll = -((alpha - 1) * torch.log(t) + (beta - 1) * torch.log(1.0 - t)
                - (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(S)))
        return nll

    @staticmethod
    def _kl_beta_uniform(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:

        S = alpha + beta
        t1 = - (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(S))
        t2 = (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S))
        t3 = (beta  - 1.0) * (torch.digamma(beta)  - torch.digamma(S))
        return t1 + t2 + t3  # (B,C)

    def forward(self, info, alpha, beta, pred, target, anneal: float = 1.0) :


        if self.mode == "expected":
            # expected log-loss
            logloss = self._expected_log_loss(target, alpha, beta).mean()  # (B,C)
        else:
            logloss = self.beta_nll(target, alpha, beta, eps=1e-4).mean()

        kl = self._kl_beta_uniform(alpha, beta).mean()              # (B,C)

        S = alpha + beta
        info['S'] = S.mean().item()

        evi_error =(target - pred).pow(2)
        evi_penalty = (evi_error * S).mean()
        loss_edl = torch.mean(logloss + anneal * kl + 0.5*evi_penalty)  # (B,C)


        # loss_l1 = F.l1_loss(pred, target)
        # loss_edl =  loss_l1 * 0.1 + loss_edl

        return loss_edl, info



class OracleFusionLoss(nn.Module):
    def __init__(self, grad_weight=10.0):
        super().__init__()
        self.sobelconv = Sobelxy()
        self.grad_weight = grad_weight

    @staticmethod
    def select_max_abs_with_sign(xs):
        stack = torch.stack(xs, dim=1)                # [B,M,1,H,W]
        idx = torch.argmax(stack.abs(), dim=1, keepdim=True)
        return torch.gather(stack, 1, idx).squeeze(1)

    def forward(self, modalities, fu):
        # modalities: list of tensors, e.g. [vis, nir, tir] or [vis, tir]
        info = defaultdict(float)

        x_ref = modalities[0]
        for x in modalities[1:]:
            x_ref = torch.max(x_ref, x)
        loss_l1 = F.l1_loss(fu, x_ref)
        info["L1_max"] = loss_l1.item()

        g_list = []
        for x in modalities:
            g = self.sobelconv(x)
            g_list.append(g)

        g_fu = self.sobelconv(fu)
        g_ref = self.select_max_abs_with_sign(g_list)

        loss_grad = F.l1_loss(g_fu, g_ref)
        info["Grad"] = loss_grad.item()

        loss = loss_l1 + self.grad_weight * loss_grad
        info["Total"] = loss.item()
        return loss, info

class MissFusionLoss(nn.Module):
    def __init__(self,
                 bank,
                 delta_scale: float = 0.05,
                 grad_weight: float = 10.0,
                 lambda_beta: float = 0.5,
                 lambda_sub: float = 1e-2,
                 lambda_tv = 0.01,
    ):
        super().__init__()
        self.sobel = Sobelxy()
        self.evi = EvidentialBetaLoss()
        self.delta_scale = delta_scale

        self.grad_weight = grad_weight
        self.lambda_beta = lambda_beta
        self.lambda_sub = lambda_sub
        self.lambda_tv = lambda_tv

        self.register_buffer("c1", bank["c1"].float())
        self.register_buffer("c2", bank["c2"].float())
        self.register_buffer("c3", bank["c3"].float())
        self.register_buffer("c4", bank["c4"].float())
        self.register_buffer("b1", bank["b1"].float())
        self.register_buffer("b2", bank["b2"].float())
        self.register_buffer("b3", bank["b3"].float())
        self.register_buffer("b4", bank["b4"].float())


    @torch.no_grad()
    def soft_mode_targets(self, delta_true, centers, bases, tau=0.01, k=2):
        B, C, H, W = delta_true.shape

        Q, _ = torch.linalg.qr(bases, mode="reduced")
        x = delta_true.permute(0, 2, 3, 1).reshape(-1, C)

        xc = x[:, None, :] - centers[None, :, :]
        coef = torch.einsum("nmc,mcr->nmr", xc, Q)
        recon = centers[None, :, :] + torch.einsum("nmr,mcr->nmc", coef, Q)
        err = ((x[:, None, :] - recon) ** 2).mean(dim=-1)
        q = torch.softmax(-err / tau, dim=1)
        q = q.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return q

    def alll(self,
                yhat, y, f0_plus, f0_cur,
                sub1, sub2, sub3, sub4,
                mu, alpha, beta, tau):

        info = defaultdict(float)


        mu_t = repair_strength_target_from_delta((f0_plus-f0_cur), scale=self.delta_scale).detach()
        loss_evi, info = self.evi(info, alpha, beta, mu, mu_t, anneal=tau)
        info["evi"] = loss_evi.item()


        loss_sub = (sub1 + sub2 + sub3 + sub4) / 4.

        loss_img = F.l1_loss(yhat, y)

        g_fu= self.sobel(yhat)
        g_ref= self.sobel(y)
        loss_grad = F.l1_loss(g_fu, g_ref)

        loss = (

                self.lambda_beta * loss_evi +
                loss_img + self.grad_weight * loss_grad +
                self.lambda_sub * loss_sub
            )

        info["img"] = loss_img.item()
        info["grad"] = loss_grad.item()
        info["sub"] = loss_sub.item()
        info["total"] = loss.item()

        return loss, info


    @staticmethod
    def tv_loss(x):
        tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return tv_h + tv_w


    def adv_loss(self,
                      y_adv, y_plus, y_safe,
                      u_c, u_mu):
        info = defaultdict(float)

        loss_img = F.l1_loss(y_adv, y_plus)

        g_fu = self.sobel(y_adv)
        g_ref = self.sobel(y_plus)
        loss_grad = F.l1_loss(g_fu, g_ref)

        loss_img_safe = F.l1_loss(y_adv, y_safe)
        g_safe = self.sobel(y_safe)
        loss_grad_safe = F.l1_loss(g_fu, g_safe)

        tv = self.tv_loss(u_c) + self.tv_loss(u_mu)


        loss_adv = -(loss_img + self.grad_weight * loss_grad) + (loss_img_safe + self.grad_weight * loss_grad_safe)  + self.lambda_tv * tv

        info["advl1"] = loss_img.item()
        info["safel1"] = loss_img_safe.item()
        info["advg"] = loss_grad.item()
        info["safeg"] = loss_grad_safe.item()
        info["adv_tv"] = tv.item()
        info["adv"] = loss_adv.item()
        return loss_adv, info

    def model_loss(self,
                   y_clean, y_adv, y_plus,
                   sub1, sub2, sub3, sub4,
                   ):
        info = defaultdict(float)

        y_clean = y_clean.detach()
        y_plus = y_plus.detach()
        l1_adv = F.l1_loss(y_adv, y_plus)
        l1_c_a = F.l1_loss(y_adv, y_clean)

        g_clean = self.sobel(y_clean)
        g_adv = self.sobel(y_adv)
        g_ref = self.sobel(y_plus)
        l_g_adv = F.l1_loss(g_adv, g_ref)
        l_g_c_a = F.l1_loss(g_adv, g_clean)


        loss_sub = (sub1 + sub2 + sub3 + sub4) / 4.

        loss_model = (
                # 0.1 *(l1_c_a + self.grad_weight * l_g_c_a)
                l1_c_a + self.grad_weight * l_g_c_a
                + self.lambda_sub * loss_sub
                + l1_adv + self.grad_weight * l_g_adv
        )

        info["l1a"] = l1_adv.item()
        info["ga"] = l_g_adv.item()
        info["l1ca"] = l1_c_a.item()
        info["gca"] = l_g_c_a.item()

        return loss_model, info




