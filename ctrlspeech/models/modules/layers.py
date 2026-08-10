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
from typing import Optional

from librosa.filters import mel as librosa_mel_fn
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio

from x_transformers.x_transformers import apply_rotary_pos_emb

from ..backbone.qwen3 import Qwen3Model, Qwen3Config


# raw wav to mel spec
mel_basis_cache = {}
hann_window_cache = {}


def get_bigvgan_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
    fmin=0,
    fmax=None,
    center=False,
):  # Copy from https://github.com/NVIDIA/BigVGAN/tree/main
    device = waveform.device
    key = f"{n_fft}_{n_mel_channels}_{target_sample_rate}_{hop_length}_{win_length}_{fmin}_{fmax}_{device}"

    if key not in mel_basis_cache:
        mel = librosa_mel_fn(sr=target_sample_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=fmin, fmax=fmax)
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)  # TODO: why they need .float()?
        hann_window_cache[key] = torch.hann_window(win_length).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_length) // 2
    waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    spec = torch.stft(
        waveform,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

    return mel_spec


def get_vocos_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
):
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mel_channels,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    ).to(waveform.device)
    if len(waveform.shape) == 3:
        waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'

    assert len(waveform.shape) == 2

    mel = mel_stft(waveform)
    mel = mel.clamp(min=1e-5).log()
    return mel


class MelSpec(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24_000,
        mel_spec_type="vocos",
    ):
        super().__init__()
        assert mel_spec_type in ["vocos", "bigvgan"], print("We only support two extract mel backend: vocos or bigvgan")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate

        if mel_spec_type == "vocos":
            self.extractor = get_vocos_mel_spectrogram
        elif mel_spec_type == "bigvgan":
            self.extractor = get_bigvgan_mel_spectrogram

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, wav):
        if self.dummy.device != wav.device:
            self.to(wav.device)

        mel = self.extractor(
            waveform=wav,
            n_fft=self.n_fft,
            n_mel_channels=self.n_mel_channels,
            target_sample_rate=self.target_sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        return mel


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


# Global Response Normalization layer (Instance Normalization ?)


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ConvNeXt-V2 Block https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
# ref: https://github.com/bfs18/e2_tts/blob/main/rfwave/modules.py#L108


class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


# RMSNorm


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.native_rms_norm = float(torch.__version__[:3]) >= 2.4

    def forward(self, x):
        if self.native_rms_norm:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = F.rms_norm(x, normalized_shape=(x.shape[-1],), weight=self.weight, eps=self.eps)
        else:
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = x * self.weight

        return x


# AdaLayerNorm
# return with modulated x for attn input, and params for later mlp modulation


class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNorm for final layer
# return only with modulated x for attn input, cuz no more mlp modulation


class AdaLayerNorm_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# FeedForward


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


# Attention with possible joint part
# modified from diffusers/src/diffusers/models/attention_processor.py


class Attention(nn.Module):
    def __init__(
        self,
        processor: JointAttnProcessor | AttnProcessor,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,  # if not None -> joint attention
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention equires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.processor = processor

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.context_dim = context_dim
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        if qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head, eps=1e-6)
            self.k_norm = RMSNorm(dim_head, eps=1e-6)
        else:
            raise ValueError(f"Unimplemented qk_norm: {qk_norm}")

        if self.context_dim is not None:
            self.to_q_c = nn.Linear(context_dim, self.inner_dim)
            self.to_k_c = nn.Linear(context_dim, self.inner_dim)
            self.to_v_c = nn.Linear(context_dim, self.inner_dim)
            if qk_norm is None:
                self.c_q_norm = None
                self.c_k_norm = None
            elif qk_norm == "rms_norm":
                self.c_q_norm = RMSNorm(dim_head, eps=1e-6)
                self.c_k_norm = RMSNorm(dim_head, eps=1e-6)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

        if self.context_dim is not None and not self.context_pre_only:
            self.to_out_c = nn.Linear(self.inner_dim, context_dim)

    def forward(
        self,
        x: float["b n d"],  # noised input x  # noqa: F722
        c: float["b n d"] = None,  # context c  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
    ) -> torch.Tensor:
        if c is not None:
            return self.processor(self, x, c=c, mask=mask, rope=rope, c_rope=c_rope)
        else:
            return self.processor(self, x, mask=mask, rope=rope)


# Attention processor


class AttnProcessor:
    def __init__(
        self,
        pe_attn_head: int | None = None,  # number of attention head to apply rope, None for all
    ):
        self.pe_attn_head = pe_attn_head

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding
    ) -> torch.FloatTensor:
        batch_size = x.shape[0]

        # `sample` projections
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # qk norm
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)

        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)

            if self.pe_attn_head is not None:
                pn = self.pe_attn_head
                query[:, :pn, :, :] = apply_rotary_pos_emb(query[:, :pn, :, :], freqs, q_xpos_scale)
                key[:, :pn, :, :] = apply_rotary_pos_emb(key[:, :pn, :, :], freqs, k_xpos_scale)
            else:
                query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
                key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            if mask.all().item():
                attn_mask = None
            else:
                attn_mask = mask
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
                attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


# Joint Attention processor for MM-DiT
# modified from diffusers/src/diffusers/models/attention_processor.py


class JointAttnProcessor:
    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x  # noqa: F722
        c: float["b nt d"] = None,  # context c, here text # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
    ) -> torch.FloatTensor:
        residual = x

        batch_size = c.shape[0]

        # `sample` projections
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # `context` projections
        c_query = attn.to_q_c(c)
        c_key = attn.to_k_c(c)
        c_value = attn.to_v_c(c)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_query = c_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_key = c_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_value = c_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # qk norm
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)
        if attn.c_q_norm is not None:
            c_query = attn.c_q_norm(c_query)
        if attn.c_k_norm is not None:
            c_key = attn.c_k_norm(c_key)

        # apply rope for context and noised input independently
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)
        if c_rope is not None:
            freqs, xpos_scale = c_rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            c_query = apply_rotary_pos_emb(c_query, freqs, q_xpos_scale)
            c_key = apply_rotary_pos_emb(c_key, freqs, k_xpos_scale)

        # joint attention
        query = torch.cat([query, c_query], dim=2)
        key = torch.cat([key, c_key], dim=2)
        value = torch.cat([value, c_value], dim=2)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = F.pad(mask, (0, c.shape[1]), value=True)  # no mask for c (text)
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        # Split the attention outputs.
        x, c = (
            x[:, : residual.shape[1]],
            x[:, residual.shape[1] :],
        )

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)
        if not attn.context_pre_only:
            c = attn.to_out_c(c)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)
            # c = c.masked_fill(~mask, 0.)  # no mask for c (text)

        return x, c


# DiT Block


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.0, qk_norm=None, pe_attn_head=None):
        super().__init__()

        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=AttnProcessor(pe_attn_head=pe_attn_head),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=qk_norm,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None):  # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x


# MMDiT Block https://arxiv.org/abs/2403.03206


class MMDiTBlock(nn.Module):
    r"""
    modified from diffusers/src/diffusers/models/attention.py

    notes.
    _c: context related. text, cond, etc. (left part in sd3 fig2.b)
    _x: noised input related. (right part)
    context_pre_only: last layer only do prenorm + modulation cuz no more ffn
    """

    def __init__(
        self, dim, heads, dim_head, ff_mult=4, dropout=0.1, context_dim=None, context_pre_only=False, qk_norm=None
    ):
        super().__init__()
        if context_dim is None:
            context_dim = dim
        self.context_pre_only = context_pre_only

        self.attn_norm_c = AdaLayerNorm_Final(context_dim) if context_pre_only else AdaLayerNorm(context_dim)
        self.attn_norm_x = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=JointAttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            context_dim=context_dim,
            context_pre_only=context_pre_only,
            qk_norm=qk_norm,
        )

        if not context_pre_only:
            self.ff_norm_c = nn.LayerNorm(context_dim, elementwise_affine=False, eps=1e-6)
            self.ff_c = FeedForward(dim=context_dim, mult=ff_mult, dropout=dropout, approximate="tanh")
        else:
            self.ff_norm_c = None
            self.ff_c = None
        self.ff_norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_x = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, c, t, mask=None, rope=None, c_rope=None):  # x: noised input, c: context, t: time embedding
        # pre-norm & modulation for attention input
        if self.context_pre_only:
            norm_c = self.attn_norm_c(c, t)
        else:
            norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(c, emb=t)
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(x, emb=t)

        # attention
        x_attn_output, c_attn_output = self.attn(x=norm_x, c=norm_c, mask=mask, rope=rope, c_rope=c_rope)

        # process attention output for context c
        if self.context_pre_only:
            c = None
        else:  # if not last layer
            c = c + c_gate_msa.unsqueeze(1) * c_attn_output

            norm_c = self.ff_norm_c(c) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            c_ff_output = self.ff_c(norm_c)
            c = c + c_gate_mlp.unsqueeze(1) * c_ff_output

        # process attention output for input x
        x = x + x_gate_msa.unsqueeze(1) * x_attn_output

        norm_x = self.ff_norm_x(x) * (1 + x_scale_mlp[:, None]) + x_shift_mlp[:, None]
        x_ff_output = self.ff_x(norm_x)
        x = x + x_gate_mlp.unsqueeze(1) * x_ff_output

        return c, x


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
    def __init__(self, text_num_embeds, text_dim):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds, text_dim)
        self.precompute_max_pos = 10000  # ~44s of 24khz audio
        self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)

    def forward(self, text: int["b nt"]):  # noqa: F722
        batch, seq_len = text.shape[0], text.shape[1]
        text = self.text_embed(text)  # b n -> b n d
        batch_start = torch.zeros((batch,), dtype=torch.long)
        pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
        text_pos_embed = self.freqs_cis[pos_idx]
        text = text + text_pos_embed
        return text


# differ from haiyang
class MelPrenet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_layers=2,
        dim=256,
        dropout_rate=0.5,
        random_prenet_dropout=False,
        activation="ReLU",
        prenet_shortcut=False,
        prenet_layer_norm=False,
        prenet_last_layer_norm=False,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.prenet = nn.ModuleList()
        self.random_prenet_dropout = random_prenet_dropout
        self.dim = dim
        self.out_channels = out_channels
        self.activation = getattr(nn, activation)()

        self.prenet_shortcut = prenet_shortcut
        self.prenet_layer_norm = prenet_layer_norm
        self.prenet_last_layer_norm = prenet_last_layer_norm
        for layer in range(n_layers):
            n_inputs = in_channels if layer == 0 else dim
            if not self.prenet_layer_norm:
                self.prenet += [nn.Sequential(nn.Linear(n_inputs, dim), self.activation)]
            else:
                self.prenet += [nn.Sequential(nn.Linear(n_inputs, dim), self.activation, nn.LayerNorm(dim))]
        if self.prenet_last_layer_norm:
            self.layer_norm = nn.LayerNorm(dim)

        self.out_proj = nn.Linear(dim, out_channels)

    def forward(self, x):
        for i in range(len(self.prenet)):
            if self.random_prenet_dropout and self.training:
                dropout_rate = torch.rand(1).item() * (self.dropout_rate + 0.1)
            else:
                dropout_rate = self.dropout_rate

            if self.prenet_shortcut and i > 0:
                x = F.dropout(self.prenet[i](x), dropout_rate) + x
            else:
                x = F.dropout(self.prenet[i](x), dropout_rate)
        if self.prenet_last_layer_norm:
            x = self.layer_norm(x)
        x = self.out_proj(x)
        return x


class MelEmbedding(nn.Module):
    def __init__(self, mel_prenet) -> None:
        super().__init__()
        self.mel_prenet = mel_prenet
        self.precompute_max_pos = 4096  # ~44s of 24khz audio
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(mel_prenet.out_channels, self.precompute_max_pos), persistent=False
        )

    def forward(self, mel):
        batch, seq_len = mel.shape[0], mel.shape[1]
        mel = self.mel_prenet(mel)
        batch_start = torch.zeros((batch,), dtype=torch.long)
        pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
        mel_pos_embed = self.freqs_cis[pos_idx]
        mel = mel + mel_pos_embed
        return mel


class VAEEmbedding(nn.Module):
    def __init__(self, num_channels, out_dim) -> None:
        super().__init__()
        self.proj = nn.Linear(num_channels, out_dim)
        self.precompute_max_pos = 4096  # ~44s of 24khz audio
        self.register_buffer("freqs_cis", precompute_freqs_cis(out_dim, self.precompute_max_pos), persistent=False)

    def forward(self, vae):
        batch, seq_len = vae.shape[0], vae.shape[1]
        vae = self.proj(vae)
        batch_start = torch.zeros((batch,), dtype=torch.long)
        pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
        vae_pos_embed = self.freqs_cis[pos_idx]
        vae = vae + vae_pos_embed
        return vae


class MelPostnet(nn.Module):
    """
    Mel-spectrogram postnet with optional VAE decoder and convolutional post-processing.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        out_channels: int = 80,
        dim: int = 256,
        dropout_rate: float = 0.5,
        using_postnet: bool = False,
        postnet_layers: int = 5,
        activation: str = "ReLU",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.activation_fn = getattr(nn, activation)

        # Linear layer to predict mean and log-variance
        self.proj = nn.Linear(in_channels, out_channels * 2)

        # VAE decoder (3-layer MLP)
        self.vae_decoder = self._build_mlp(
            in_dim=out_channels, hidden_dim=dim, out_dim=out_channels, n_layers=3, dropout=dropout_rate
        )

        # Optional CNN-based postnet
        self.postnet = (
            self._build_postnet(
                channels=dim, out_channels=out_channels, kernel_size=5, n_layers=postnet_layers, dropout=dropout_rate
            )
            if using_postnet
            else nn.Identity()
        )

    def _build_mlp(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, dropout: float) -> nn.Sequential:
        layers = []
        for i in range(n_layers):
            # Determine layer dimensions
            dim_in = in_dim if i == 0 else hidden_dim
            dim_out = out_dim if i == n_layers - 1 else hidden_dim

            layers.append(nn.Linear(dim_in, dim_out, bias=False))
            if i < n_layers - 1:
                layers.append(self.activation_fn())
            layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def _build_postnet(
        self, channels: int, out_channels: int, kernel_size: int, n_layers: int, dropout: float
    ) -> nn.Sequential:
        layers = []
        for i in range(n_layers):
            in_ch = out_channels if i == 0 else channels
            out_ch = out_channels if i == n_layers - 1 else channels

            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2, bias=False))
            layers.append(nn.BatchNorm1d(out_ch))
            if i < n_layers - 1:
                layers.append(self.activation_fn())
            layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * eps
        """
        std = torch.exp(0.5 * logvar) * temp
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        # Predict mean and logvar
        mu, logvar = self.proj(x).chunk(2, dim=-1)

        # Sample and decode
        z = self.reparameterize(mu, logvar)
        decoded = self.vae_decoder(z)
        vae_out = z + decoded

        # Apply postnet if enabled
        # if self.postnet is not None:
        # Transpose for Conv1d: (B, T, C) -> (B, C, T)
        conv_in = vae_out.transpose(1, 2)
        conv_out = self.postnet(conv_in).transpose(1, 2)
        return vae_out + conv_out, mu, logvar, vae_out

        # return vae_out, mu, logvar


class CausualAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, head_dim=None, attn_bias=False, qk_norm=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=attn_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=attn_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=attn_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=attn_bias)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=1e-6)  # unlike olmo, only on the head dim!
            self.k_norm = RMSNorm(self.head_dim, eps=1e-6)  # thus post q_norm does not need reshape

    def forward(self, x, attn_mask=None, past_key_value=None, rope=None, using_cache=False):
        bsz, seq_len, embed_dim = x.size()
        query_states = self.q_proj(x).view(bsz, self.num_attention_heads, seq_len, self.head_dim).contiguous()
        key_states = self.k_proj(x).view(bsz, self.num_attention_heads, seq_len, self.head_dim).contiguous()
        value_states = (
            self.v_proj(x).view(bsz, self.num_attention_heads, seq_len, self.head_dim).contiguous()
        )  # bsz seq_len dim -> bsz nheads seq head_dim

        if self.qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query_states = apply_rotary_pos_emb(query_states, freqs, q_xpos_scale)  # bsz num_heads seq_len head_dim
            key_states = apply_rotary_pos_emb(key_states, freqs, k_xpos_scale)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)  # concat at seq_len dim
        past_kv = (key_states, value_states) if using_cache else None

        attn = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=None, dropout_p=0.0, is_causal=True
        )

        attn = (
            attn.transpose(1, 2).contiguous().view(bsz, seq_len, embed_dim)
        )  # bsz nheads seq head_dim -> bsz seq nheads head_dim -> bsz seq_len dim

        attn = self.o_proj(attn)
        return attn, past_kv


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class CausualAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        activation,
        head_dim=None,
        norm="rms_norm",
        attn_bias=False,
        qk_norm=False,
    ):
        super().__init__()

        self.self_attn = CausualAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            attn_bias=attn_bias,
            qk_norm=qk_norm,
        )

        self.ffn = MLP(
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            activation=activation,
        )

        if norm == "layer_norm":
            self.ffn_norm = nn.LayerNorm(hidden_size)
            self.attn_norm = nn.LayerNorm(hidden_size)
        elif norm == "rms_norm":
            self.ffn_norm = RMSNorm(hidden_size, eps=1e-6)
            self.attn_norm = RMSNorm(hidden_size, eps=1e-6)

    def forward(
        self,
        x,
        attn_mask=None,
        past_key_values=None,
        using_cache=False,
        rope=None,
    ):
        residual = x
        x = self.attn_norm(x)
        x, past_kv = self.self_attn(x, attn_mask, past_key_value=past_key_values, rope=rope, using_cache=using_cache)
        x = residual + x

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x, past_kv


class DownsampleBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride):
        super().__init__()
        # PyTorch的Conv1D需要(B, C, L)格式，所以通道数是in_dim
        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            stride=stride
        )
        # LayerNorm作用在最后一个维度(Dim)上
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (Batch, Time, Dim)
        # 1. 交换维度以适应Conv1D: (B, T, D) -> (B, D, T)
        x = x.permute(0, 2, 1)

        # 2. 应用卷积层
        x = self.conv(x) # -> (B, D_out, T_new)

        # 3. 换回维度以应用LayerNorm: (B, D_out, T_new) -> (B, T_new, D_out)
        x = x.permute(0, 2, 1)

        # 4. 应用归一化和激活函数
        x = self.norm(x)
        x = self.activation(x)
        return x


class StopPredictor(nn.Module):
    def __init__(
        self,
        dim: int,
        output_dim: int,
        num_layers: int = 3,
        nhead: int = 8,
        dim_feedfwd: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        config = Qwen3Config(
            hidden_size = dim,
            intermediate_size = dim_feedfwd,
            num_hidden_layers = num_layers,
            num_attention_heads = nhead,
            num_key_value_heads=1,
        )
        self.model = Qwen3Model(config)
        del self.model.embed_tokens
        self.proj = nn.Linear(dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,                      # (B, T, D)
        src_key_padding_mask: Optional[torch.Tensor] = None, # (B, T)
        past_key_values: Optional[tuple] = None,
        use_cache: Optional[bool] = None,
    ) -> torch.Tensor:                        # (B, T) or (B, T), past_key_values
        outputs = self.model(
            inputs_embeds=x, 
            attention_mask=src_key_padding_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.proj(outputs.last_hidden_state).squeeze(-1)
        
        if use_cache:
            return logits, outputs.past_key_values
        return logits, None


class MLPStopPredictor(nn.Module):
    def __init__(
        self,
        dim: int,
        output_dim: int = 1,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        x: torch.Tensor,                      # (B, T, D)
        src_key_padding_mask: Optional[torch.Tensor] = None, # (B, T)
        past_key_values: Optional[tuple] = None,
        use_cache: Optional[bool] = None,
    ) -> tuple:                               # (B, T, output_dim), None
        logits = self.mlp(x)  # (B, T, output_dim)

        return logits, None


class StopPredictorconcat(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int = 3,
        nhead: int = 8,
        dim_feedfwd: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(dim*2,dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim_feedfwd,
            dropout=dropout,
            batch_first=True,  # 输入格式为 (B, T, D)
            activation=F.gelu, # 使用GELU激活函数
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.proj = nn.Linear(dim, 1)

    def forward(
        self,
        x: torch.Tensor,                      # (B, T, D)
        src_key_padding_mask: Optional[torch.Tensor] = None, # (B, T)
    ) -> torch.Tensor:                        # (B, T)
        B, T, _ = x.shape
        device = x.device
        x = self.input_proj(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=T,
            device=device
        ) # Shape: (T, T)
       
        h = self.encoder(
            x,
            mask=causal_mask,                       # 因果掩码
            src_key_padding_mask=src_key_padding_mask  # padding 掩码
        )
        logits = self.proj(h).squeeze(-1)
        return logits