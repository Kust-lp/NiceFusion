import numbers
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

if triton is not None:
    @triton.jit
    def qk_partial_kernel(
        Q, K, PARTIAL,
        N: tl.constexpr, C: tl.constexpr, HEADS: tl.constexpr,
        QS0: tl.constexpr, QS1: tl.constexpr, QS2: tl.constexpr, QS3: tl.constexpr,
        KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr, KS3: tl.constexpr,
        SPLITS: tl.constexpr, BC: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """One CTA produces a C x C partial QK for a slice of spatial positions."""
        bh = tl.program_id(0)
        s = tl.program_id(1)
        b = bh // HEADS
        h = bh % HEADS

        row = tl.arange(0, BC)
        col = tl.arange(0, BC)
        n = s * BLOCK_N + tl.arange(0, BLOCK_N)
        q = tl.load(
            Q + b * QS0 + h * QS1 + row[:, None] * QS2 + n[None, :] * QS3,
            mask=(row[:, None] < C) & (n[None, :] < N), other=0.0,
        )
        k = tl.load(
            K + b * KS0 + h * KS1 + col[:, None] * KS2 + n[None, :] * KS3,
            mask=(col[:, None] < C) & (n[None, :] < N), other=0.0,
        )
        product = q[:, None, :] * k[None, :, :]
        partial = tl.sum(product, axis=2)

        offsets = ((bh * SPLITS + s) * BC * BC
                   + row[:, None] * BC + col[None, :])
        tl.store(PARTIAL + offsets, partial,
                 mask=(row[:, None] < C) & (col[None, :] < C))

    @triton.jit
    def qk_reduce_kernel(
        PARTIAL, OUT,
        C: tl.constexpr, SPLITS: tl.constexpr, BC: tl.constexpr,
        BS: tl.constexpr,
    ):
        """One CTA reduces all split partials for one QK output scalar."""
        bh = tl.program_id(0)
        pair = tl.program_id(1)
        row = pair // C
        col = pair % C
        s = tl.arange(0, BS)

        offsets = ((bh * SPLITS + s) * BC * BC + row * BC + col)
        parts = tl.load(PARTIAL + offsets, mask=s < SPLITS, other=0.0)
        val = tl.sum(parts, axis=0)
        tl.store(OUT + bh * C * C + pair, val)

class _SplitKQKEngine:
    """Scratch buffers for a single Attention module (inference only).

    Scratch is reused only for sequential calls on the same CUDA stream. Do not
    share one module concurrently across CUDA streams / threads.
    """
    def __init__(self, q, k, block_n=256, num_warps=8):
        if q.ndim != 4 or q.shape != k.shape:
            raise ValueError('Q/K must have identical [B, heads, C, N] shapes')
        if not (q.is_cuda and k.is_cuda and q.dtype == k.dtype == torch.float32):
            raise ValueError('Split-K Triton QK is CUDA FP32 only')
        b, heads, c, n = q.shape
        self.b, self.heads, self.c, self.n = b, heads, c, n
        self.block_n, self.num_warps = block_n, num_warps
        self.splits = triton.cdiv(n, block_n)
        self.bc = triton.next_power_of_2(c)
        self.bs = triton.next_power_of_2(self.splits)
        self.partial = torch.empty(
            (b * heads, self.splits, self.bc, self.bc),
            device=q.device, dtype=torch.float32)
        self.output = torch.empty(
            (b, heads, c, c), device=q.device, dtype=torch.float32)

    def __call__(self, q, k):
        qk_partial_kernel[(self.b * self.heads, self.splits)](
            q, k, self.partial,
            self.n, self.c, self.heads,
            *q.stride(), *k.stride(),
            self.splits, self.bc, self.block_n,
            num_warps=self.num_warps, enable_fp_fusion=False)
        qk_reduce_kernel[(self.b * self.heads, self.c * self.c)](
            self.partial, self.output,
            self.c, self.splits, self.bc, self.bs,
            num_warps=4, enable_fp_fusion=False)
        return self.output

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.use_triton_qk = False
        self.qk_block_n = 256
        self.qk_num_warps = 8
        self._qk_engine = None
        self._qk_engine_key = None

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        if torch.is_grad_enabled():
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        else:
            F.normalize(q, dim=-1, out=q)
            F.normalize(k, dim=-1, out=k)
        if (self.use_triton_qk and triton is not None and not self.training
                and not torch.is_grad_enabled()
                and q.is_cuda and q.dtype == torch.float32
                and k.dtype == torch.float32):
            key = (tuple(q.shape), tuple(q.stride()), tuple(k.stride()),
                   q.device, self.qk_block_n, self.qk_num_warps)
            if self._qk_engine is None or self._qk_engine_key != key:
                self._qk_engine = _SplitKQKEngine(
                    q, k, block_n=self.qk_block_n, num_warps=self.qk_num_warps)
                self._qk_engine_key = key
            qk = self._qk_engine(q, k)
        else:
            qk = q @ k.transpose(-2, -1)

        attn = qk * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class Encoder(nn.Module):
    def __init__(self,  channel, num_blocks, ffn_expansion_factor = 2, bias=False, LayerNorm_type='WithBias', head = 8):
        super(Encoder, self).__init__()

        self.FE = nn.Conv2d(1, channel, kernel_size=3, stride=1, padding=1)

        self.Enc = nn.Sequential(
            *[TransformerBlock(dim=channel, num_heads=head, ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.FE(x)
        x = self.Enc(x)
        return x

class outHead(nn.Module):

    def __init__(self, in_channels, mode="sigmoid"):
        super().__init__()
        self.mode = mode
        self.body =nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, 1, kernel_size=3, stride=1, padding=1, padding_mode="reflect"))

    def forward(self, x, v):
        x = self.body(x)
        x = x + v
        if self.mode == "sigmoid":
            return torch.sigmoid(x)
        if self.mode == "tanh":
            return torch.tanh(x)
        return x

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
        return mu, S

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
            nn.Conv2d(c, num_modes * rank, 3, 1, 1),
        )

    def forward(self, x):
        z = self.net(x)
        B, _, H, W = z.shape
        z = z.view(B, self.num_modes, self.rank, H, W)
        return z

if triton is not None:
    @triton.jit
    def _bank_compact_selected_kernel(
        SELECTED, WEIGHTS, Z, EFFECTIVE_CENTERS, BASES, OUT,
        HW: tl.constexpr, M: tl.constexpr, C: tl.constexpr, R: tl.constexpr,
        BP: tl.constexpr, BC: tl.constexpr,
    ):
        b = tl.program_id(2)
        px = tl.program_id(0) * BP + tl.arange(0, BP)
        ch = tl.program_id(1) * BC + tl.arange(0, BC)
        valid_px = px < HW
        valid = valid_px[None, :] & (ch[:, None] < C)

        m0 = tl.load(SELECTED + (b * 2 + 0) * HW + px, mask=valid_px, other=0)
        m1 = tl.load(SELECTED + (b * 2 + 1) * HW + px, mask=valid_px, other=0)
        w0 = tl.load(WEIGHTS + (b * 2 + 0) * HW + px, mask=valid_px, other=0.)
        w1 = tl.load(WEIGHTS + (b * 2 + 1) * HW + px, mask=valid_px, other=0.)
        proj0 = tl.full((BC, BP), 0., tl.float32)
        proj1 = tl.full((BC, BP), 0., tl.float32)
        for r in tl.static_range(R):
            z0 = tl.load(Z + (b * M * R + m0 * R + r) * HW + px,
                         mask=valid_px, other=0.)
            z1 = tl.load(Z + (b * M * R + m1 * R + r) * HW + px,
                         mask=valid_px, other=0.)
            idx0 = m0[None, :] * (C * R) + ch[:, None] * R + r
            idx1 = m1[None, :] * (C * R) + ch[:, None] * R + r
            basis0 = tl.load(BASES + idx0, mask=valid, other=0.)
            basis1 = tl.load(BASES + idx1, mask=valid, other=0.)
            proj0 = proj0 + basis0 * z0[None, :]
            proj1 = proj1 + basis1 * z1[None, :]

        cen0 = tl.load(EFFECTIVE_CENTERS + m0[None, :] * C + ch[:, None],
                       mask=valid, other=0.)
        cen1 = tl.load(EFFECTIVE_CENTERS + m1[None, :] * C + ch[:, None],
                       mask=valid, other=0.)
        local0 = cen0 + proj0
        local1 = cen1 + proj1
        out = w0[None, :] * local0 + w1[None, :] * local1
        tl.store(OUT + (b * C + ch[:, None]) * HW + px[None, :], out, mask=valid)

def _forward_selected_triton(self, topi, w, z):
    if triton is None:
        raise RuntimeError('Triton is not installed')
    if self.training:
        raise RuntimeError('Triton Bank is inference-only; call model.eval()')
    if not (topi.is_cuda and w.is_cuda and z.is_cuda):
        raise ValueError('Triton Bank requires CUDA tensors')
    if z.dtype != torch.float32 or w.dtype != torch.float32:
        raise ValueError('Triton Bank currently supports FP32 only')
    if topi.dtype != torch.int64:
        raise ValueError('Expected torch.topk int64 indices')
    B, K, H, W = topi.shape
    M = self.mode_num
    C = self.centers.shape[1]
    R = self.mode_dim
    if K != 2 or z.shape != (B, M, R, H, W) or w.shape != (B, K, H, W):
        raise ValueError('Expected Compact Top-2 indices/weights and matching z')
    if not all(t.is_contiguous() for t in (topi, w, z, self.bases)):
        raise ValueError('Expected contiguous Compact outputs and Bank inputs')

    centers = self.centers + self.center_offset * 0.1
    output = torch.empty((B, C, H, W), device=z.device, dtype=z.dtype)
    BP = getattr(self, '_bank_bp', 32)
    BC = getattr(self, '_bank_bc', 16)
    grid = (triton.cdiv(H * W, BP), triton.cdiv(C, BC), B)
    _bank_compact_selected_kernel[grid](
        topi, w, z, centers, self.bases, output,
        H * W, M, C, R, BP, BC,
        enable_fp_fusion=False, num_warps=4,
    )
    return output

class AffineSubspaceBank(nn.Module):
    """Affine centers and bases with an optional fused inference kernel."""
    def __init__(self, c, mode_num, mode_dim, learn_bases=True):
        super().__init__()
        self.mode_num = mode_num
        self.mode_dim = mode_dim
        self.learn_bases = learn_bases
        self.use_triton_bank = False
        self._bank_bp = 32
        self._bank_bc = 16

        centers = torch.zeros(mode_num, c)
        bases = torch.zeros(mode_num, c, mode_dim)
        self.register_buffer("centers", centers.clone(), persistent=True)
        self.center_offset = nn.Parameter(torch.zeros_like(centers))

        if learn_bases:
            self.bases = nn.Parameter(bases.clone())
        else:
            self.register_buffer("bases", bases.clone(), persistent=True)

    def forward_topk(self, pi, z):

        B, M, H, W = pi.shape
        R = z.shape[2]
        C = self.centers.shape[1]

        ccenter = self.centers + self.center_offset * 0.1

        k = self.infer_k

        topi = torch.topk(
            pi, k=k, dim=1, sorted=False
        ).indices

        w = torch.gather(pi, 1, topi)  # [B, K, H, W]
        w = w / (w.sum(dim=1, keepdim=True) + 1e-6)

        idx_z = topi.unsqueeze(2).expand(-1, -1, R, -1, -1)  # [B, K, R, H, W]
        z_sel = torch.gather(z, 1, idx_z)  # [B, K, R, H, W]

        idx_hwk = topi.permute(0, 2, 3, 1)  # [B, H, W, K]

        centers_sel = F.embedding(idx_hwk, ccenter)  # [B, H, W, K, C]
        centers_sel = centers_sel.permute(0, 3, 4, 1, 2)  # [B, K, C, H, W]

        bases_flat = self.bases.reshape(self.mode_num, -1)  # [M, C*R]
        bases_sel = F.embedding(idx_hwk, bases_flat)  # [B, H, W, K, C*R]
        bases_sel = bases_sel.view(B, H, W, k, C, R).permute(0, 3, 4, 5, 1, 2)

        local = centers_sel + torch.einsum("bkcrhw,bkrhw->bkchw", bases_sel, z_sel)
        delta = torch.einsum("bkhw,bkchw->bchw", w, local)
        return delta

    def forward_selected(self, topi, w, z):
        if self.use_triton_bank and not self.training and not torch.is_grad_enabled() and z.is_cuda and z.dtype == torch.float32:
            if triton is None:
                raise RuntimeError('Triton is required for CUDA Triton Bank inference.')
            if topi.shape[1] == 2:
                return _forward_selected_triton(self, topi, w, z)
        B, k, H, W = topi.shape
        R = z.shape[2]
        C = self.centers.shape[1]
        ccenter = self.centers + self.center_offset * 0.1

        idx_z = topi.unsqueeze(2).expand(-1, -1, R, -1, -1)
        z_sel = torch.gather(z, 1, idx_z)
        idx_hwk = topi.permute(0, 2, 3, 1)

        centers_sel = F.embedding(idx_hwk, ccenter)
        centers_sel = centers_sel.permute(0, 3, 4, 1, 2)
        bases_flat = self.bases.reshape(self.mode_num, -1)
        bases_sel = F.embedding(idx_hwk, bases_flat)
        bases_sel = bases_sel.view(B, H, W, k, C, R).permute(0, 3, 4, 5, 1, 2)

        local = centers_sel + torch.einsum("bkcrhw,bkrhw->bkchw", bases_sel, z_sel)
        return torch.einsum("bkhw,bkchw->bchw", w, local)

    def forward(self, pi, z):
        ccenter = self.centers + self.center_offset * 0.1
        if self.training:
            local = ccenter[None, :, :, None, None] + torch.einsum(
                "mcr,bmrhw->bmchw", self.bases, z
            )
            delta = torch.einsum("bmhw,bmchw->bchw", pi, local)
        else:
            delta = self.forward_topk(pi, z)

        return delta

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
        return pi

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

class FCModule(nn.Module):
    def __init__(self, c, num_modes, rank, k, learnable_bank=True, head=8, ffn_expansion_factor=2,
                 bias=False, LayerNorm_type='WithBias'):
        super().__init__()

        self.topk = k
        self.use_compact_topk = False
        self.mode_head = ModeRouter(c, num_modes)

        self.coord_head = CoordHead(c, num_modes, rank, head=head,
                                    ffn_expansion_factor=ffn_expansion_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type)
        self.risk_head = RiskFiLM(c)
        self.bank = AffineSubspaceBank(c, num_modes, rank, learn_bases=learnable_bank)
        self.bank.infer_k = k

    def forward(self, f, mu, c):

        z = self.coord_head(f)

        pi = self.mode_head(f)

        if self.training or not self.use_compact_topk or self.topk >= pi.shape[1]:
            pi_use = topk_simplex(pi, self.topk)
            delta_hat = self.bank(pi_use, z)
        else:
            p_sel, topi = torch.topk(pi, k=self.topk, dim=1)
            p_sel = p_sel / (p_sel.sum(dim=1, keepdim=True) + 1e-6)
            w = p_sel / (p_sel.sum(dim=1, keepdim=True) + 1e-6)
            delta_hat = self.bank.forward_selected(topi, w, z)

        g, b = self.risk_head(delta_hat, mu)

        delta_hat = c * (g * delta_hat + b)

        f_corr = f + delta_hat

        return f_corr

class FusionDecoder(nn.Module):
    def __init__(self, channel, num_blocks, head, ffn_expansion_factor, bias, LayerNorm_type, num_modes, rank, topk, learnable_bank):
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

    def forward(self, f, mu, c):

        f1 = self.Dec[0](f)
        f1_corr = self.F1M(f1, mu, c)

        f2 = self.Dec[1](f1_corr)
        f2_corr = self.F2M(f2, mu, c)

        f3 = self.Dec[2](f2_corr)
        f3_corr = self.F3M(f3, mu, c)

        f4 = self.Dec[3](f3_corr)
        f4_corr = self.F4M(f4, mu, c)

        return  f4_corr

class missFusionM(nn.Module):
    def __init__(self, channel=48, num_blocks=4,
                 ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias',
                 head=8, num_modes=32, rank=4, topk=2, tau_c=0.2,
                 learnable_bank=True, *, triton_qk=False,
                 compact_topk=False, triton_bank=False):
        super().__init__()
        self.tau_c = tau_c

        self.Enc = Encoder(channel, num_blocks, ffn_expansion_factor, bias, LayerNorm_type, head)

        self.evi_head = EvidenceHead(channel, head=head,
                                     ffn_expansion_factor=ffn_expansion_factor,
                                     bias=bias, LayerNorm_type=LayerNorm_type)

        self.Dec = FusionDecoder(channel, num_blocks, head, ffn_expansion_factor, bias, LayerNorm_type, num_modes, rank, topk, learnable_bank)
        self.fuse_head = outHead(in_channels=channel, mode="sigmoid" )
        self.set_acceleration(triton_qk=triton_qk,
                              compact_topk=compact_topk,
                              triton_bank=triton_bank)

    def forward(self, v, t):
        v0 = v
        v = self.Enc(v)
        t = self.Enc(t)
        f = (v + t)/2

        mu, S = self.evi_head(f)
        c = 1.0 - torch.exp(-self.tau_c * torch.clamp(S-2, min=0.0) )

        f4_corr = self.Dec(f, mu, c)

        fuse = self.fuse_head(f4_corr, v0)

        return fuse

    def set_acceleration(self, *, triton_qk=False, compact_topk=False, triton_bank=False):
        """Toggle the three optimizations without changing weights/state_dict.

        For the original PyTorch path disable all three.
        Triton Bank requires the Compact Top-K path.
        """
        if triton_bank and not compact_topk:
            raise ValueError("triton_bank=True requires compact_topk=True")
        for module in self.modules():
            if isinstance(module, Attention):
                module.use_triton_qk = bool(triton_qk)
            elif isinstance(module, FCModule):
                module.use_compact_topk = bool(compact_topk)
                module.bank.use_triton_bank = bool(triton_bank)
        return self


if __name__ == '__main__':
    import time

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = True

    model = missFusionM().to(device).eval()
    model.set_acceleration(
        triton_qk=True,
        compact_topk=True,
        triton_bank=True
    )
    x1 = torch.rand(1, 1, 640, 640, device=device)
    x2 = torch.rand(1, 1, 640, 640, device=device)

    with torch.inference_mode():
        for _ in range(20):
            _ = model(x1, x2)


    all_time = 0
    for _ in range(50):
        start = time.perf_counter()
        with torch.inference_mode():
                _ = model(x1, x2)
        end = time.perf_counter()

        total_time = end - start
        all_time += total_time
    inference_time = round(all_time/50, 3)
    fps = round(1 / inference_time, 3)

    print(f"Inference Time: {inference_time:.3f} s")
    print(f"FPS: {fps:.3f}")
