"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

# sinusoidal position embedding


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# convolutional position embedding


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x: float["b n d"], mask: bool["b n"] | None = None):  # noqa: F722
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        out = x.permute(0, 2, 1)

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)

        return out


# rotary positional embedding related


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


# time step conditioning embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: float["b"]):  # noqa: F821
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # b d
        return time


class TextEmbedding(nn.Module):
    def __init__(
        self, 
        num_embeddings, 
        embedding_dim,
        padding_idx = None,
    ):
        super().__init__()
        self.text_embed = nn.Embedding(
            num_embeddings = num_embeddings, 
            embedding_dim = embedding_dim, 
            padding_idx = padding_idx,
        )
        print(f"Warning: TextEmbedding padding_idx is {padding_idx}.")
        # self.precompute_max_pos = 10000  # ~44s of 24khz audio
        # self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)

    def forward(self, text: int["b nt"]):  # noqa: F722
        batch, seq_len = text.shape[0], text.shape[1]
        text_emb = self.text_embed(text)  # b n -> b n d
        # batch_start = torch.zeros((batch,), dtype=torch.long)
        # pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos).to(text_emb.device) #[0,...,50]
        # text_pos_embed = self.freqs_cis[pos_idx]
        # text_emb = text_emb + text_pos_embed
        return text_emb


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class VAEProjector(nn.Module):
    def __init__(self, in_dim=64, d_model=1024, hidden=512, dropout=0.1):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim, elementwise_affine=True)
        self.fc1 = nn.Linear(in_dim, hidden*2)  # *2 for GEGLU
        self.act = GEGLU()
        self.fc2 = nn.Linear(hidden, d_model)
        self.out_norm = nn.LayerNorm(d_model, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)
        self.res = nn.Linear(in_dim, d_model) if in_dim != d_model else nn.Identity()
        self.scale = nn.Parameter(torch.ones(1))  # learnable scale

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        # x: [B, N, in_dim] 或 [B, in_dim]
        x0 = x
        x = self.in_norm(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.out_norm(x)
        x = x + self.res(x0)  # 短残差对齐
        return x * self.scale