import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.TModel import FusionM
from sklearn.cluster import MiniBatchKMeans

from utils.dataset import H5Dataset
import  math

class OracleSafePair(nn.Module):
    def __init__(self, fckpt, device="cuda:0"):
        super().__init__()
        self.full = FusionM().to(device)
        self.full.load_state_dict(torch.load(fckpt, map_location=device))
        self.full.eval()
        for p in self.full.parameters():
            p.requires_grad = False



    @torch.no_grad()
    def forward(self, v, n, t, d = 1 ):
        y_plus, _, [fp1, fp2, fp3, fp4] = self.full(v, t, n=n)
        y_safe, _, [fs1, fs2, fs3, fs4] = self.full(v, t, n=None)
        if d == 1:
            return (fp1 - fs1).permute(0, 2, 3, 1).reshape(-1, fp1.shape[1])

        if d == 2:
            return (fp2 - fs2).permute(0, 2, 3, 1).reshape(-1, fp2.shape[1])

        if d ==3:
            return (fp3 - fs3).permute(0, 2, 3, 1).reshape(-1, fp3.shape[1])
        if d ==4:
            return (fp4 - fs4).permute(0, 2, 3, 1).reshape(-1, fp4.shape[1])




def select_active_tokens(tokens: torch.Tensor, keep_ratio=0.1):#0.3
    score = tokens.abs().mean(dim=1)
    k = max(1, int(tokens.shape[0] * keep_ratio))
    idx = torch.topk(score, k=k, largest=True).indices

    return tokens[idx]

def build_affine_subspaces(tokens: torch.Tensor, num_modes=64, rank=12):
    tokens_np = tokens.cpu().numpy()
    km = MiniBatchKMeans(n_clusters=num_modes, batch_size=4096, random_state=42)
    labels = km.fit_predict(tokens_np)
    centers_np = km.cluster_centers_

    N, C = tokens.shape
    centers = []
    bases = []

    for m in range(num_modes):
        cluster = tokens[labels == m]   # [Nm, C]

        if cluster.shape[0] == 0:
            c = torch.from_numpy(centers_np[m]).to(tokens).float()
            U = torch.zeros(C, rank, dtype=tokens.dtype, device=tokens.device)
        else:
            c = cluster.mean(dim=0)     # [C]
            xc = cluster - c[None, :]   # [Nm, C]

            Nm = xc.shape[0]
            r_eff = min(rank, C, max(0, Nm - 1))

            U = torch.zeros(C, rank, dtype=tokens.dtype, device=tokens.device)
            if r_eff > 0:
                _, _, vh = torch.linalg.svd(xc, full_matrices=False)
                U0 = vh[:r_eff].T.contiguous()
                U0 = torch.linalg.qr(U0, mode="reduced").Q
                U[:, :r_eff] = U0

        centers.append(c)
        bases.append(U)

    centers = torch.stack(centers, dim=0)   # [M+1, C]
    bases = torch.stack(bases, dim=0)       # [M+1, C, r]




    return centers.float().cpu(), bases.float().cpu()





if __name__ == '__main__':
    device = "cuda:0"

    pairm = OracleSafePair("../ckpt_TM/120.pth")
    dataset = H5Dataset("../datasets/TVNF/train/imgsize_128_stride_200.h5")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    payload = {}
    for d in [1,2,3,4]:
        All_delta = []

        for vis, nir, tir in tqdm(loader, desc=f"Delta_f{d}"):
            vis = vis.to(device)
            nir = nir.to(device)
            tir = tir.to(device)
            x = pairm(vis, nir, tir, d)
            All_delta.append(x.detach().cpu())

        All_delta = torch.cat(All_delta, dim=0)
        tokens = select_active_tokens(All_delta)


        c, b =  build_affine_subspaces(tokens,num_modes=32, rank=4)
        payload[f"c{d}"] = c
        payload[f"b{d}"] = b

    torch.save(payload, "../ckpt_TM/Subspaces(32,4).pt")