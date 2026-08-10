"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from ..modules.layers import (
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNorm_Final,
)

# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        dim,
        out_dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.0,
        ff_mult=4,
        qk_norm=None,
        pe_attn_head=None,
        long_skip_connection=False,
        checkpoint_activations=False,
        **kwargs,
    ):
        super().__init__()

        self.out_dim = out_dim
        # self.time_embed = TimestepEmbedding(dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=dim) #?

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, out_dim)

        self.checkpoint_activations = checkpoint_activations
        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        t,  # time step  # noqa: F821 F722
        # noisy_mask: None,
        mask: bool["b n"] | None = None,  # noqa: F722
        patch_size=4,
        random_time: bool = False,
    ):
        batch_size = t.shape[0]
        folded_batch_size, seq_len = x.shape[0], x.shape[1]
        if not random_time:
            # 如果random time 开了，注释掉
            t = t.repeat_interleave(folded_batch_size // batch_size, dim=0)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)
        output = output[:,-patch_size:,]
        if not random_time:
            output = output.contiguous().view(batch_size, -1, self.out_dim) #如果random time 开了，注释掉

        return output
