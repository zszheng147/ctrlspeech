import math
from dataclasses import dataclass, field
from typing import Literal
import random
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

from einops import rearrange, repeat
from torchdiffeq import odeint

from .embeds import TimestepEmbedding, VAEProjector
from .modules import CausalAR, AggregationEncoder, StopPredictor, MLPStopPredictor
from .backbone.dit import DiT
from .vae.online_feature import load_state, process_online


@dataclass
class LocDiTConfig:
    name: Literal["DiT", "Unett"] = "DiT"

    # Model
    model: dict = field(default_factory=dict)
    history_vae_window_size: int = 4

    # Training
    random_time: bool = False
    time_schedule: bool = False
    drop_cond_prob: float = 0.1
    speaker_drop_prob: float = 0.5
    emotion_drop_prob: float = 0.1
    # Inference
    odeint_kwargs: dict = field(default_factory=lambda: {
        # atol = 1e-5,
        # rtol = 1e-5,
        "method": "euler",
    })

    def __post_init__(self):
        # if self.name != "DiT":
        if self.name not in ["DiT", "Unett"]:
            raise ValueError


class DiTar(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.audio_channels = config.audio_channels
        self.dim = config.dim
        self.patch_size = config.patch_size
        self.text_vocab_size = config.text_vocab_size
        self.mlp_hidden_dim = config.mlp_hidden_dim

        self.causalAR = CausalAR(
            qwen_config_path=config.backbone.qwen_config_path,
            pretrained_LM_path=config.backbone.pretrained_LM_path,
            load_pretrained_weights=config.backbone.load_pretrained_weights,
            load_phoneme=config.backbone.load_phoneme,
            weighted_layers=config.backbone.weighted_layers,
        )

        self.vae_projector = VAEProjector(
            in_dim=self.audio_channels, 
            d_model=self.dim,
            hidden=self.mlp_hidden_dim,
            dropout=0.1,
        )
        self.use_seperate_linear = config.use_seperate_linear
        if self.use_seperate_linear:
            self.vae_projector_for_dit = VAEProjector(
                in_dim=self.audio_channels,
                d_model=self.dim,
                hidden=self.mlp_hidden_dim,
                dropout=0.1,
            )

        self.time_embedding = TimestepEmbedding(dim=self.dim)
        self.cond_projection = nn.Sequential(
            nn.Linear(self.dim + 192, self.dim * 2),
            nn.SiLU(),
            nn.Linear(self.dim * 2, self.dim)
        )
        self.aggregation_encoder = AggregationEncoder(**config.aggregation_encoder)
        
        self.LocDiT_config = LocDiTConfig(**config.loc_decoder)
        self.LocDiT = DiT(**self.LocDiT_config.model)

        self.stop_predictor_type = config.stop_predictor_type

        # self.stop_projection = StopPredictor(
        #     dim=self.dim,
        #     output_dim=3,
        #     num_layers=3,
        #     nhead=8,
        #     dim_feedfwd=self.dim * 4
        # )
        self.stop_projection = MLPStopPredictor(
            dim=self.dim,
            output_dim=3,
            hidden_dim=self.dim // 2,
            dropout=0.5
        )

        self.drop_condition = config.drop_condition

        self.audio_type = config.audio_type
        path, model_tag = config.vocoder.path.rsplit('/', 1)

        generator = load_state(save_path=path, tag=model_tag).eval()
        self.generator = generator

        self.use_ar_l1_loss = config.loss.use_ar_l1_loss
        self.use_stop_loss = config.loss.use_stop_loss
        self.use_vae_projected_l1_loss = config.loss.use_vae_projected_l1_loss

        self.pitch_embedding = nn.Embedding(128, self.dim)
        self.loudness_embedding = nn.Embedding(64, self.dim)
        self.duration_embedding = nn.Embedding(192, self.dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self, raw_audio, audio_lengths, speaker_embs, 
        text_inputs, text_masks=None, duration_segments=None, 
        pitch=None, loudness=None, stresses=None, emotions=None
    ):
        device = self.device
        with torch.no_grad():
            vae_features = process_online(raw_audio, self.generator).transpose(1, 2).detach()
            valid_audio_lengths = torch.ceil(audio_lengths / self.generator.hop_length).long()
            vae_padding_masks = torch.arange(vae_features.shape[1], device=device) < valid_audio_lengths.unsqueeze(-1)

        (
            ar_input_embeds, 
            modality_type_ids, 
            padded_vae_features, 
            padded_vae_padding_masks,
            vae_projected, 
            text_masks, 
            vae_aggregated_masks
        ) = self.get_ar_input(
            vae_features=vae_features,
            vae_padding_masks=vae_padding_masks,
            speaker_embs=speaker_embs,
            text_inputs=text_inputs,
            text_masks=text_masks,
            duration_segments=duration_segments,
            pitch=pitch,
            loudness=loudness,
            stresses=stresses,
        )

        ar_padding_mask = torch.cat([text_masks, vae_aggregated_masks], dim=1)

        ar_pred = self.causalAR(
            input_embeds=ar_input_embeds,
            padding_mask=ar_padding_mask,
            modality_type_ids=modality_type_ids,
        )

        original_ar_pred = ar_pred.clone()
        ar_pred = ar_pred[:, text_masks.shape[-1]-1:-1]

        if self.use_ar_l1_loss:
            ar_pred_mask = ar_padding_mask[:, text_masks.shape[-1]-1:-1]
            ar_pred_mask[:, 0] = True # B, n_patch
            ar_l1_loss = torch.abs(ar_pred[ar_pred_mask]).mean() #? IDK
        else:
            ar_l1_loss = torch.tensor(0, device=device)

        # Flow matching stage: 
        # x_1 -> Add noise -> x_0
        if not self.LocDiT_config.random_time:
            time = torch.rand((ar_pred.shape[0],), dtype=ar_pred.dtype, device=device)
            t = time[..., None, None]  # (B, 1, 1)
        else:
            n_patches = ar_pred.shape[1]
            time = torch.rand((ar_pred.shape[0], n_patches), dtype=ar_pred.dtype, device=device)
            t = time.repeat_interleave(self.patch_size, dim = 1)  # (B, n_patches * patch_size)
            t = t[..., None]  # (B, n_patches * patch_size, 1)
            time = time.view(-1)

        
        x1 = padded_vae_features
        x0 = torch.randn_like(x1)

        if not self.LocDiT_config.time_schedule:
            xt = (1 - t) * x0 + t * x1
        else:
            xt = torch.cos(math.pi * t / 2) * x0 + torch.sin(math.pi * t / 2) * x1
        flow = x1 - x0

        locdit_input, locdit_mask, vae_projected_l1_loss = self.get_decoder_input(
            vae_features=padded_vae_features,
            # vae_projected=vae_projected,      # (B, T_padded, D_model)
            vae_padding_mask=padded_vae_padding_masks,
            h_predict=ar_pred,
            noisy_input=xt,
        )

        time_embed = self.time_embedding(time)  # (B, D_model)
        cond_embed = torch.cat([time_embed, speaker_embs], dim=-1)  # (B, D_model + 192)
        cond_embed = self.cond_projection(cond_embed)  # (B, D_model)
        
        pred = self.LocDiT(
            x=locdit_input,
            t=cond_embed,
            mask=locdit_mask,
            patch_size=self.patch_size,
            random_time=self.LocDiT_config.random_time,
        )
        if self.LocDiT_config.random_time:
            pred = pred.contiguous().view(vae_projected.shape[0], n_patches*self.patch_size, -1)

        # Diffusion loss
        # loss = F.mse_loss(pred, flow, reduction='none')
        diff_loss = F.l1_loss(pred, flow, reduction='none')
        diff_loss = diff_loss[padded_vae_padding_masks].mean()

        if self.use_stop_loss:
            input_for_stop = original_ar_pred[:, :-1, :]
            input_for_stop = (input_for_stop * 0.5).detach() + input_for_stop * (1 - 0.5)
            
            stop_padding_mask = ar_padding_mask[:, :-1]
            stop_logits, _ = self.stop_projection(input_for_stop, stop_padding_mask)

            # Stop loss - Three-class classification: 0 (first), 1 (middle), 2 (last)
            _, aggr_seq_len = vae_aggregated_masks.shape
            
            # Find first valid token index
            first_true_index = vae_aggregated_masks.int().argmax(dim=1, keepdim=True)  # (B, 1)
            
            # Find last valid token index
            aggr_mask_flipped = vae_aggregated_masks.flip(dims=[1])
            last_true_index = aggr_seq_len - aggr_mask_flipped.int().argmax(dim=1, keepdim=True) - 1  # (B, 1)
            
            _idxs = repeat(
                torch.arange(aggr_seq_len, device=device),
                'n_patches -> b n_patches',
                b=ar_pred.shape[0],
            )
            
            # Create three-class targets for speech tokens
            # 0: first token, 1: middle tokens, 2: last token
            stop_targets = torch.ones(ar_pred.shape[0], aggr_seq_len, dtype=torch.long, device=device)
            stop_targets[_idxs == first_true_index] = 0  # First token
            stop_targets[_idxs == last_true_index] = 2  # Last token
            # Middle tokens remain as 1
            # Set padding positions to -100 (will be ignored in loss calculation)
            stop_targets[~vae_aggregated_masks] = -100
            
            # Text tokens: set to -100 (ignored in loss calculation)
            text_stop_targets = torch.full(
                (ar_pred.shape[0], text_masks.shape[-1]-1), -100, dtype=torch.long,
                device=device
            )
            stop_targets = torch.cat([text_stop_targets, stop_targets], dim=1)
            
            # Use cross entropy loss for multi-class classification with ignore_index
            stop_loss = F.cross_entropy(
                stop_logits.view(-1, stop_logits.shape[-1]), 
                stop_targets.view(-1),
                ignore_index=-100
            )
            
            # Calculate accuracy (only on valid positions, excluding ignore_index=-100)
            stop_pred = stop_logits.argmax(dim=-1)  # (B, seq_len)
            valid_mask = (stop_targets != -100)  # Mask for non-ignored positions
            correct_predictions = (stop_pred == stop_targets) & valid_mask
            stop_acc = correct_predictions.sum().float() / valid_mask.sum().float()
        else:
            stop_loss = torch.tensor(0, device=device)
            stop_acc = torch.tensor(0, device=device)

        return {
            "diff_loss": diff_loss, 
            "stop_loss": stop_loss, 
            "stop_acc": stop_acc,
            "ar_l1_loss": ar_l1_loss,
            "vae_projected_l1_loss": vae_projected_l1_loss,
        }

    def _aggregate_segment_embed(self, frame_embed, segments):
        """Mean-pool frame-level embeddings into one embedding per duration segment.

        ``duration`` (``segments``) may span a longer range than the available
        ``pitch``/``loudness`` frames. Segments that fall entirely beyond the
        available frames contribute a zero embedding (no pitch/loudness
        conditioning there); partially-covered segments are averaged over only
        the frames that exist.
        """
        n_frame = frame_embed.shape[0]
        zero = frame_embed.new_zeros(frame_embed.shape[-1])

        out = []
        for start, end in segments:
            if start >= n_frame:
                # No pitch/loudness data for this segment -> add nothing.
                out.append(zero)
            else:
                e = min(end, n_frame)  # clamp to available frames
                if start != e:
                    out.append(frame_embed[start:e].mean(dim=0))
                else:
                    out.append(frame_embed[start])
        return torch.stack(out, dim=0)

    def get_pitch_loudness_embed(self, pitch, loudness, segment):
        pitch_embeds = []
        loudness_embeds = []
        duration_embeds = []

        for idx, (p, l, s) in enumerate(zip(pitch, loudness, segment)):
            pitch_embed = self.pitch_embedding(torch.from_numpy(p).long().to(self.device))
            pitch_embeds.append(self._aggregate_segment_embed(pitch_embed, s))

            loudness_embed = self.loudness_embedding(torch.from_numpy(l).long().to(self.device))
            loudness_embeds.append(self._aggregate_segment_embed(loudness_embed, s))

            duration = torch.tensor(
                [end - start if start != end else 0 for start, end in s],
                device=self.device
            )
            duration_embed = self.duration_embedding(duration)
            duration_embeds.append(duration_embed)
        return pitch_embeds, loudness_embeds, duration_embeds

    def get_ar_input(
        self, vae_features, vae_padding_masks, speaker_embs, text_inputs, text_masks, 
        duration_segments=None, pitch=None, loudness=None
    ):
        text_embeds = self.causalAR.model.embed_tokens(text_inputs)

        if duration_segments is not None and pitch is not None and loudness is not None:
            pitch_embeds, loudness_embeds, duration_embeds = self.get_pitch_loudness_embed(
                pitch, loudness, duration_segments
            )
            pitch_embeds = rnn.pad_sequence(pitch_embeds, padding_value=0, batch_first=True)
            loudness_embeds = rnn.pad_sequence(loudness_embeds, padding_value=0, batch_first=True)
            duration_embeds = rnn.pad_sequence(duration_embeds, padding_value=0, batch_first=True)
            if self.training:
                if torch.rand(1) < 0.5:
                    text_embeds += pitch_embeds
                if torch.rand(1) < 0.5:
                    text_embeds += loudness_embeds
                if torch.rand(1) < 0.5:
                    text_embeds += duration_embeds
            else:
                text_embeds[:, :pitch_embeds.shape[1]] += pitch_embeds
                text_embeds[:, :loudness_embeds.shape[1]] += loudness_embeds
                text_embeds[:, :duration_embeds.shape[1]] += duration_embeds

        # if stresses is not None:
        #     # Expected: stresses is token-aligned with text_inputs (B, L) and in {0, 1}.
        #     # We scale the added control to reduce OOD shifts when a single token flips to 1.
        #     stress_embed = self.stress_embedding(stresses)
        #     text_embeds = text_embeds + (0.5 * stress_embed)

        B, L, D = vae_features.shape
        pad_length = (-L) % self.patch_size

        padded_vae_features = F.pad(vae_features, (0, 0, 0, pad_length))
        padded_vae_padding_masks = F.pad(vae_padding_masks, (0, pad_length), value=False)

        vae_projected = self.vae_projector(padded_vae_features)

        if self.patch_size == 1:
            vae_aggregated = vae_projected
            vae_aggregated_masks = padded_vae_padding_masks
        else:
            vae_aggregated, vae_aggregated_masks = self.aggregation_encoder(
                vae_projected, 
                padding_mask=padded_vae_padding_masks
            )

        input_embeds = torch.cat([text_embeds, vae_aggregated], dim=1)
        modality_type_ids = torch.cat([
            torch.zeros(*text_embeds.shape[:-1]), torch.ones(*vae_aggregated.shape[:-1])
        ], dim=1).to(self.device, dtype=torch.int64)

        return (
            input_embeds,
            modality_type_ids,
            padded_vae_features,
            padded_vae_padding_masks,
            vae_projected,
            text_masks,
            vae_aggregated_masks,
        )
    
    def get_decoder_input(self, vae_features, vae_padding_mask, h_predict, noisy_input):
        device = self.device

        B, n_patches, D_model = h_predict.shape
        history_vae_window_size = self.LocDiT_config.history_vae_window_size

        # ------------------------------------------------------------------
        # 1. Context ("ctx") feature – the current AR hidden state
        # ------------------------------------------------------------------
        ctx = h_predict  # (B, n_patches, D_model)
        # if random.random() < self.LocDiT_config.drop_cond_prob:
        #     ctx = torch.zeros_like(ctx)
        folded_ctx = rearrange(ctx, 'b n_patches d_model -> (b n_patches) 1 d_model')
        
        # Pad *left* with n_history_vae zero‑tokens so the first window contains all zeros.
        if self.use_seperate_linear:
            dit_vae_projected = self.vae_projector_for_dit(vae_features)
            if self.use_vae_projected_l1_loss:
                vae_projected_l1_loss = torch.abs(dit_vae_projected[vae_padding_mask]).mean() #? IDK
            else:
                vae_projected_l1_loss = torch.tensor(0, device=device)
            
            _left_pad = torch.zeros(
                (B, history_vae_window_size, D_model), 
                device=device, dtype=dit_vae_projected.dtype
            )
            vae_projected__left_padded = torch.cat([
                _left_pad, dit_vae_projected,
            ], dim=1)  # (B, T_vae + n_history_vae, D_model)
        else:
            # TODO
            ...
    
        _left_pad_mask = torch.zeros(
            (B, history_vae_window_size), dtype=torch.bool, device=device
        )
        vae_padding_mask__left_padded = torch.cat([
            _left_pad_mask, vae_padding_mask,
        ], dim=1)  # (B, T_vae + n_history_vae)

        #滑动窗口，它沿着指定的维度滑动，并把窗口内的数据提取出来作为一个新的维度。
        hist_emb = vae_projected__left_padded.unfold(
            dimension = 1, 
            size = history_vae_window_size,  #指定每个窗口的大小
            step = self.patch_size, #指定窗口每次滑动的步长
        )[:, :n_patches, ...]  # (B, n_patches, D_model, n_history_vae) #(B, n_windows, D_model, size)
        hist_msk = vae_padding_mask__left_padded.unfold(
            dimension = 1, 
            size = history_vae_window_size,
            step = self.patch_size,
        )[:, :n_patches, ...]  # (B, n_patches, n_history_vae)

        # Merge batch & time for LocDiT.
        hist_emb = rearrange(
            hist_emb,
            'b n_patches d_model n_history_vae -> (b n_patches) n_history_vae d_model',
        )
        hist_msk = rearrange(
            hist_msk,
            'b n_patches n_history_vae -> (b n_patches) n_history_vae',
        )

        # 3. Current noisy patch  x_t  → embed to D_model
        # ------------------------------------------------------------------
        if self.use_seperate_linear:
            noisy_emb = self.vae_projector_for_dit(noisy_input)
        else:
            noisy_emb = self.vae_projector(noisy_input)
        
        noisy_emb = rearrange(
            noisy_emb, 
            'b (n_patches patch_size) d_model -> (b n_patches) patch_size d_model', 
            patch_size=self.patch_size
        )
        folded_vae_mask = rearrange(
            vae_padding_mask, 
            'b (n_patches patch_size) -> (b n_patches) patch_size', 
            patch_size=self.patch_size
        )

        # 4. Concatenate [ctx | history | noisy]
        # ------------------------------------------------------------------
        if self.drop_condition == "only_drop_ctx":
            if random.random() < self.LocDiT_config.drop_cond_prob:
                folded_ctx = torch.zeros_like(folded_ctx)
        elif self.drop_condition == "drop_ctx_or_his":
            if random.random() < self.LocDiT_config.drop_cond_prob:
                folded_ctx = torch.zeros_like(folded_ctx)
            if random.random() < self.LocDiT_config.drop_cond_prob:
                hist_emb = torch.zeros_like(hist_emb)
        elif self.drop_condition == "drop_ctx_and_his":
            if random.random() < self.LocDiT_config.drop_cond_prob:
                folded_ctx = torch.zeros_like(folded_ctx)
                hist_emb = torch.zeros_like(hist_emb)
        else:
            raise NotImplementedError
            
        combined_input = torch.cat((folded_ctx, hist_emb, noisy_emb), dim=1)
        
        ctx_mask = torch.ones((B * n_patches, 1), dtype=torch.bool, device=self.device)
        combined_mask = torch.cat((ctx_mask, hist_msk, folded_vae_mask), dim=1)

        return combined_input, combined_mask, vae_projected_l1_loss


    @torch.no_grad()
    def sample(
        self,
        prompt_audio,
        speaker_embs,
        text_inputs,
        text_masks,
        duration_segments=None,
        pitch=None,
        loudness=None,
        max_seq_length: int = 300,
        steps: int = 32,
        cfg_strength: float = 1.5,
        use_cache: bool = False,
        progress: bool = False,
    ):
        device = self.device
        def odeint_fn(t, x):
            time_embed = self.time_embedding(t.unsqueeze(0))
            # Apply speaker embedding processing (same as forward)
            cond_embed = torch.cat([time_embed, speaker_embs], dim=-1)  # (B, D_model + 192)
            cond_embed = self.cond_projection(cond_embed)  # (B, D_model)

            if self.use_seperate_linear:
                x = self.vae_projector_for_dit(x)
            else:
                x = self.vae_projector(x)
            cond_input = torch.cat([ctx, historical_patch, x], dim=1)

            cond_locdit_mask = torch.cat([
                torch.ones((1, 1), dtype=torch.bool, device=device),
                torch.ones((1, self.LocDiT_config.history_vae_window_size), dtype=torch.bool, device=device),
                torch.ones((1, self.patch_size), dtype=torch.bool, device=device)
            ], dim=1)

            pred = self.LocDiT(
                x=cond_input,
                t=cond_embed,
                mask=cond_locdit_mask,
                patch_size=self.patch_size,
                random_time=self.LocDiT_config.random_time,
            )

            if self.drop_condition == "only_drop_ctx" or self.drop_condition == "drop_ctx_or_his":
                # For unconditional: use zero speaker embedding
                uncond_speaker_embs = torch.zeros_like(speaker_embs)
                uncond_cond_embed = torch.cat([time_embed, uncond_speaker_embs], dim=-1)
                uncond_cond_embed = self.cond_projection(uncond_cond_embed)

                uncond_input = torch.cat([torch.zeros_like(ctx), historical_patch, x], dim=1)
                uncond_locdit_mask = torch.cat([
                    torch.ones((1, 1), dtype=torch.bool, device=device),
                    torch.ones((1, self.LocDiT_config.history_vae_window_size), dtype=torch.bool, device=device),
                    torch.ones((1, self.patch_size), dtype=torch.bool, device=device)
                ], dim=1)

                null_pred = self.LocDiT(
                    x=uncond_input,
                    t=uncond_cond_embed,
                    mask=uncond_locdit_mask,
                    patch_size=self.patch_size,
                    random_time=self.LocDiT_config.random_time,
                )
            else:
                null_pred = self.LocDiT(
                    x=uncond_input,
                    t=cond_embed,
                    mask=uncond_locdit_mask,
                    patch_size=self.patch_size,
                    random_time=self.LocDiT_config.random_time,
                )
            return pred + (pred - null_pred) * cfg_strength

        if len(prompt_audio.shape) == 2:
            prompt_audio = prompt_audio.unsqueeze(1)
        
        prompt_vae_features = process_online(prompt_audio, self.generator).transpose(1, 2) # (B, T, D)
        prompt_vae_padding_masks = torch.ones(
            prompt_vae_features.shape[:-1], dtype=torch.bool, device=device)

        (
            ar_input_embeds, 
            modality_type_ids, 
            padded_vae_features, 
            padded_vae_padding_masks,
            vae_projected, 
            text_masks, 
            vae_aggregated_masks
        ) = self.get_ar_input(
            vae_features=prompt_vae_features,
            vae_padding_masks=prompt_vae_padding_masks,
            speaker_embs=speaker_embs,
            text_inputs=text_inputs,
            text_masks=text_masks,
            duration_segments=duration_segments,
            pitch=pitch,
            loudness=loudness,
        )

        ar_padding_mask = torch.cat([text_masks, vae_aggregated_masks], dim=1)

        vae_results = torch.empty((1, 0, self.audio_channels), device=device)
        past_key_values = None
        stop_past_key_values = None
        for _ in tqdm(range(max_seq_length), disable=not progress):
            h_predict, past_key_values = self.causalAR.inference(
                input_embeds=ar_input_embeds,
                padding_mask=ar_padding_mask,
                modality_type_ids=modality_type_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            
            h_predict_for_stop = h_predict
            stop_mask = torch.ones(
                (1, h_predict_for_stop.shape[1]), 
                dtype=torch.bool, device=device
            )
            
            if use_cache:
                stop_output = self.stop_projection(
                    h_predict_for_stop, stop_mask, 
                    past_key_values=stop_past_key_values,
                    use_cache=True
                )
                stop_logits, stop_past_key_values = stop_output
                stop_logits = stop_logits[:, -1]
            else:
                stop_logits, _ = self.stop_projection(h_predict_for_stop, stop_mask)
                stop_logits = stop_logits[:, -1]

            h_predict = h_predict[:, [-1], :]

            y0 = torch.randn(1, self.patch_size, self.audio_channels, device=device)
            
            t_start = 0
            t = torch.linspace(t_start, 1, steps+1, device=device)

            # Sway sampling 
            sway_sampling_coef = -1
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t) #[33][nfe+1]
            
            if vae_results.numel() == 0:
                historical_patch = prompt_vae_features[:, -self.LocDiT_config.history_vae_window_size:, :]
            else:
                current_length = vae_results.shape[1]
                if current_length < self.LocDiT_config.history_vae_window_size:
                    historical_patch = torch.cat([prompt_vae_features[
                        :, -(self.LocDiT_config.history_vae_window_size-current_length):, :
                    ], vae_results], dim=1)
                else:
                    historical_patch = vae_results[:, -self.LocDiT_config.history_vae_window_size:, :]
            
            if self.use_seperate_linear:
                historical_patch = self.vae_projector_for_dit(historical_patch)
            else:
                historical_patch = self.vae_projector(historical_patch)
            
            ctx = h_predict
            trajectory = odeint(odeint_fn, y0, t, **self.LocDiT_config.odeint_kwargs)
            sampled = trajectory[-1]
            vae_results = torch.cat((vae_results, sampled), dim=1)

            # 停止判断（全batch判断）- Check if predicted class is 2 (last token)
            stop_pred_class = stop_logits.argmax(dim=-1)
            if stop_pred_class == 2 and vae_results.shape[1] > 10:
                break
        
            # Update for the next step
            input_vae_patch_emb = self.vae_projector(sampled)
            vae_patch_mask = torch.ones((1, self.patch_size), dtype=torch.bool, device=device)
            if self.patch_size == 1:
                aggregation_emb = input_vae_patch_emb
            else:
                aggregation_emb, aggregation_mask = self.aggregation_encoder(
                    input_vae_patch_emb, padding_mask=vae_patch_mask
                )  # B T/4 D
            
            if use_cache:
                ar_input_embeds = aggregation_emb
            else:
                ar_input_embeds = torch.cat([ar_input_embeds, aggregation_emb], dim=1)
            
            ar_padding_mask = torch.cat([
                ar_padding_mask, torch.ones((1, 1), dtype=torch.bool, device=device)
            ], dim=1)
            modality_type_ids = torch.cat([
                modality_type_ids, torch.ones((1, 1), dtype=torch.int64, device=device)
            ], dim=1)
    
        return vae_results