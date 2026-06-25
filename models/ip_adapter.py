import torch
import torch.nn as nn


class DecoupledCrossAttention(nn.Module):
    def __init__(self, dim, n_heads=4, text_dim=512, sketch_dim=512):
        super().__init__()
        self.scale = dim ** -0.5
        self.n_heads = n_heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k_text = nn.Linear(text_dim, dim, bias=False)
        self.to_v_text = nn.Linear(text_dim, dim, bias=False)
        self.to_k_sketch = nn.Linear(sketch_dim, dim, bias=False)
        self.to_v_sketch = nn.Linear(sketch_dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, text_emb, sketch_emb):
        if sketch_emb.dim() == 2:
            sketch_emb = sketch_emb.unsqueeze(1)
        q = self.to_q(x)
        k_t = self.to_k_text(text_emb)
        v_t = self.to_v_text(text_emb)
        k_s = self.to_k_sketch(sketch_emb)
        v_s = self.to_v_sketch(sketch_emb)

        attn_t = (q @ k_t.transpose(-2, -1)) * self.scale
        attn_s = (q @ k_s.transpose(-2, -1)) * self.scale
        out_t = attn_t.softmax(dim=-1) @ v_t
        out_s = attn_s.softmax(dim=-1) @ v_s
        return self.to_out(out_t + out_s)


class CoupledCrossAttention(nn.Module):
    """Diagnostic baseline that concatenates sketch and text before attention."""

    def __init__(self, dim, n_heads=4, text_dim=512, sketch_dim=512):
        super().__init__()
        self.scale = dim ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(text_dim, dim, bias=False)
        self.to_v = nn.Linear(text_dim, dim, bias=False)
        self.sketch_to_text_dim = nn.Linear(sketch_dim, text_dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, text_emb, sketch_emb):
        if sketch_emb.dim() == 2:
            sketch_emb = sketch_emb.unsqueeze(1)
        sketch_tokens = self.sketch_to_text_dim(sketch_emb)
        tokens = torch.cat([text_emb, sketch_tokens], dim=1)
        q = self.to_q(x)
        k = self.to_k(tokens)
        v = self.to_v(tokens)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        return self.to_out(attn.softmax(dim=-1) @ v)
