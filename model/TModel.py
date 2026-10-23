import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import numbers


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

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

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

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
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


class FusionDecoder(nn.Module):
    """
    Shared three-scale decoder used by both the VT backbone branch
    and the N-plugin residual branch.
    """
    def __init__(self, channel, num_blocks, head, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.num_blocks = num_blocks
        self.Dec =nn.ModuleList([TransformerBlock(dim=channel, num_heads=head, ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks)])

    def forward(self, x):
        x_list = []
        for d in self.Dec:
            x = d(x)
            x_list.append(x)
        return x, x_list


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

class FusionM(nn.Module):
    def __init__(self, channel = 48, num_blocks = 4, ffn_expansion_factor = 2, bias=False, LayerNorm_type='WithBias', head = 8):
        super(FusionM, self).__init__()

        self.Enc = Encoder(channel, num_blocks, ffn_expansion_factor, bias, LayerNorm_type, head)
        self.Dec = FusionDecoder(channel, num_blocks, head, ffn_expansion_factor, bias, LayerNorm_type)

        self.fuse_head = outHead(in_channels=channel, mode="sigmoid")

    def forward(self, v, t, n=None):
        v0 = v
        v = self.Enc(v)
        t = self.Enc(t)

        if n is None:
           x = (v+t)/2
        else:
            n = self.Enc(n)
            x = (v+t+n)/3

        fuse_feat, x_list = self.Dec(x)
        fuse = self.fuse_head(fuse_feat, v0)

        return  fuse, x, x_list

if __name__ == '__main__':
        from thop import profile, clever_format
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model = FusionM().to(device).eval()

        x1 = torch.rand(1, 1, 640, 480, device=device)
        x2 = torch.rand(1, 1, 640, 480, device=device)
        x3 = torch.rand(1, 1, 640, 480, device=device)

        with torch.no_grad():
            flops, params = profile(model, inputs=(x1, x2, x3,), verbose=True)
        flops, params = clever_format([flops, params], "%.3f")

        print("FLOPs:", flops)
        print("Params:", params)
