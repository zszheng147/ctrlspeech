from typing import Optional, Union

import torch
import torch.nn as nn

from ..backbone.qwen3 import Qwen3Config, Qwen3Model


class AggregationEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        patch_size: int,
        pool_type: str = "avg",
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.pool_type = pool_type
        self.config = Qwen3Config(
            hidden_size = hidden_size,
            intermediate_size = intermediate_size,
            num_hidden_layers = num_hidden_layers,
            num_attention_heads = num_attention_heads,
            num_key_value_heads=1,
        )
        self.model = Qwen3Model(self.config)
        del self.model.embed_tokens
        
        # learnable summary token
        if pool_type == "cls":
            # self.summary_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.summary_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        
    def get_summary_token(self, hidden_states):
        if self.pool_type == "cls":
            return hidden_states[:, 0]
        elif self.pool_type == "avg":
            return hidden_states.mean(dim=1)
        else:
            raise ValueError(f"Invalid pool type: {self.pool_type}")   
        

    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


    def to_4d(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        dtype: torch.dtype,
        key_value_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
        key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is
        causal, a causal mask will be added.
        """
        input_shape = (attention_mask_2d.shape[0], query_length)

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        # if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
        #     if key_value_length is None:
        #         raise ValueError(
        #             "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
        #         )

        #     past_key_values_length = key_value_length - query_length
        #     causal_4d_mask = self._make_causal_mask(
        #         input_shape,
        #         dtype,
        #         device=attention_mask_2d.device,
        #         past_key_values_length=past_key_values_length,
        #         sliding_window=self.sliding_window,
        #     )
        # elif self.sliding_window is not None:
        #     raise NotImplementedError("Sliding window is currently only implemented for causal masking")

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = self._expand_mask(
            attention_mask_2d, dtype, tgt_len=input_shape[-1]
        ).to(attention_mask_2d.device)

        if causal_4d_mask is not None:
            expanded_attn_mask = causal_4d_mask.masked_fill(expanded_attn_mask.bool(), torch.finfo(dtype).min)

        # expanded_attn_mask + causal_4d_mask can cause some overflow
        expanded_4d_mask = expanded_attn_mask
        return expanded_4d_mask
        
    def forward(
        self,
        x: torch.Tensor,  # [B, L, D]
        padding_mask: torch.Tensor,  # [B, L]
    ) -> torch.Tensor:   
    
        B, L, D = x.shape
        num_patches = L // self.patch_size
        
        # Reshape to patches
        folded_x = x.contiguous().view(B * num_patches, self.patch_size, D)  # (B,T,D) -> (B*N, patch_size, D)
        folded_padding_mask = padding_mask.contiguous().view(B * num_patches, self.patch_size)

        # Add summary token to each patch
        if self.pool_type == "cls":
            summary_tok = self.summary_token.expand(B * num_patches, -1, -1)
            inputs_embeds = torch.cat([summary_tok, folded_x], dim=1)  # [B*N, 1+P, D]
            summary_mask = torch.ones((B * num_patches, 1), dtype=torch.bool, device=folded_padding_mask.device)
            attention_mask = torch.cat([summary_mask, folded_padding_mask], dim=1)
        else:
            inputs_embeds = folded_x
            attention_mask = folded_padding_mask
        
        attention_mask = self.to_4d(
            attention_mask, attention_mask.shape[1],
            key_value_length=attention_mask.shape[1], dtype=torch.float32
        )
        
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Extract summary tokens
        summary_tokens = self.get_summary_token(hidden_states).view(B, num_patches, D)

        # Compute final padding mask
        padding_mask = padding_mask.view(B, num_patches, -1)
        summary_padding_mask = torch.sum(padding_mask, dim=-1) > 0

        return summary_tokens, summary_padding_mask