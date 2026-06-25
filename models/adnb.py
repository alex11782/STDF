import torch
import torch.nn as nn

from models.ip_adapter import CoupledCrossAttention, DecoupledCrossAttention


class SpectralDiffusion(nn.Module):
    def __init__(self, dim, k_eig=128, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.diffusion_time = nn.Parameter(torch.zeros(dim))

    def forward(self, x, evals, evecs):
        if not self.enabled:
            return x
        x_hat = torch.einsum("vk,bvc->bkc", evecs, x)
        t = torch.nn.functional.softplus(self.diffusion_time).view(1, 1, -1)
        kernel = torch.exp(-evals.view(1, -1, 1) * t)
        x_diff_hat = x_hat * kernel
        return torch.einsum("vk,bkc->bvc", evecs, x_diff_hat)


class ADNB(nn.Module):
    def __init__(
        self,
        dim,
        n_heads=4,
        k_eig=128,
        text_dim=512,
        sketch_dim=512,
        use_spectral_diffusion=True,
        use_dual_path_attention=True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.diffusion = SpectralDiffusion(dim, k_eig, enabled=use_spectral_diffusion)
        self.norm2 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        attention_cls = DecoupledCrossAttention if use_dual_path_attention else CoupledCrossAttention
        self.cross_attn = attention_cls(dim, n_heads, text_dim=text_dim, sketch_dim=sketch_dim)
        self.norm4 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, evals, evecs, text_emb, sketch_emb):
        res = x
        x = self.diffusion(self.norm1(x), evals, evecs) + res
        res = x
        x, _ = self.self_attn(self.norm2(x), self.norm2(x), self.norm2(x))
        x = x + res
        res = x
        x = self.cross_attn(self.norm3(x), text_emb, sketch_emb) + res
        res = x
        x = self.ffn(self.norm4(x)) + res
        return x
