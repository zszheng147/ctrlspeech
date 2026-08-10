"""High-level CtrlSpeech inference API.

Two passes, and the second one is the point of the project:

1. **Baseline** — synthesise from a prompt voice and a target text
   (:meth:`CtrlSpeech.generate`), or skip synthesis entirely and adopt a real
   recording as the baseline (:meth:`CtrlSpeech.from_audio`).
2. **Controlled** — read the baseline's frame-level pitch and loudness curves
   and its phoneme boundaries, edit any of them, and resynthesise with the
   edited curves as conditioning (:meth:`CtrlSpeech.regenerate`).

Everything you do not edit is carried over from the baseline, so a pitch edit
leaves loudness and timing alone.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
import torch
from omegaconf import OmegaConf

from .align import MFAAligner
from .assets import download_assets
from .features import CosyVoiceSpeakerEmbedding, f0_to_coarse, get_pitch_and_loudness
from .models import DiTar, load_decoder
from .retime import MAX_DURATION_FRAMES, MAX_TIMELINE_FRAMES

SAMPLE_RATE = 16000
HOP_LENGTH = 160                    # 100 frames/s, matching the *100 in duration_segments
FPS = SAMPLE_RATE // HOP_LENGTH     # = 100
PITCH_BINS = 128
LOUDNESS_BINS = 64
LOUDNESS_DB_RANGE = 60.0            # loudness_db_max - loudness_db_floor
LOUDNESS_BINS_PER_DB = (LOUDNESS_BINS - 1) / LOUDNESS_DB_RANGE   # ~1.05 bin/dB

# Aliases; ctrlspeech.retime owns both limits so they cannot drift apart.
MAX_PHONE_FRAMES = MAX_DURATION_FRAMES
MAX_CONTROL_FRAMES = MAX_TIMELINE_FRAMES

# The SVAE encoder rates [4,4,5,5] give 400 samples per latent -> 40 Hz, and the
# AR model emits patch_size latents per step, so one step is 0.1 s of audio.
VAE_FRAMES_PER_SECOND = 40
AR_PATCH_SIZE = 4
AR_STEP_SECONDS = AR_PATCH_SIZE / VAE_FRAMES_PER_SECOND

DEFAULT_STEPS = 32
DEFAULT_CFG_STRENGTH = 1.5
TRIM_TOP_DB = 40


# ─────────────────────────────────────────────────────────────────────────────
# Text / timing helpers
# ─────────────────────────────────────────────────────────────────────────────
def estimate_max_seq_length(target_seconds):
    """Pick an AR step ceiling from the intended duration.

    The stop predictor normally ends the utterance early; this only has to avoid
    truncating it. A fixed cap of 100 steps covers just 10 s, which silently cuts
    off the tail once a duration edit stretches the sentence.
    """
    target_seconds = float(target_seconds)
    if not np.isfinite(target_seconds) or target_seconds <= 0:
        return 100
    steps = int(np.ceil(target_seconds / AR_STEP_SECONDS * 1.35)) + 10
    return int(np.clip(steps, 100, 400))


def tokenize_phones(prompt_phones, target_phones, text_tokenizer):
    """Join prompt and target phones with ``|`` and map them to token ids."""
    text_inputs = prompt_phones + " | " + target_phones
    text_inputs = [re.sub(r"\d", "", t) for t in text_inputs.split(" ")]
    input_ids = torch.LongTensor([text_tokenizer.get(t, 1) for t in text_inputs])
    input_ids = input_ids.unsqueeze(0)
    return input_ids, input_ids != 0


def parse_times(times):
    return [float(t) if t != "|" else 0.0 for t in times.split(" ")]


def build_prompt_segments(annotation):
    """Frame-index (start, end) pairs for every prompt phoneme token."""
    begins = parse_times(annotation["starts"])
    ends = parse_times(annotation["ends"])
    return [(int(s * FPS), int(e * FPS)) for s, e in zip(begins, ends)]


# ─────────────────────────────────────────────────────────────────────────────
# Global prosody edits, applied in coarse-bin space so they compose with drawing
# ─────────────────────────────────────────────────────────────────────────────
def shift_pitch_semitones(coarse, semitones):
    """Transpose a quantised pitch contour; unvoiced frames stay untouched."""
    coarse = np.asarray(coarse, dtype=float)
    voiced = coarse > 1
    if not voiced.any() or semitones == 0:
        return coarse.copy()

    mel_min = 1127 * np.log(1 + 65.0 / 700)
    mel_max = 1127 * np.log(1 + 650.0 / 700)
    f0_mel = (coarse - 1) * (mel_max - mel_min) / (PITCH_BINS - 2) + mel_min
    f0 = 700 * (np.exp(f0_mel / 1127) - 1)
    f0_shifted = f0 * (2.0 ** (semitones / 12.0))

    f0_input = np.where(voiced, f0_shifted, 0.0).astype(np.double)
    new_coarse = f0_to_coarse(f0_input.copy())

    out = coarse.copy()
    out[voiced] = new_coarse[voiced]
    return out


def shift_loudness_db(coarse, db):
    """Raise or lower a quantised loudness contour by a number of dB."""
    coarse = np.asarray(coarse, dtype=float)
    return np.clip(coarse + round(db * LOUDNESS_BINS_PER_DB), 0, LOUDNESS_BINS - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Generation:
    """Audio plus the prosody the model actually produced."""

    audio: np.ndarray
    pitch: np.ndarray
    loudness: np.ndarray
    sample_rate: int = SAMPLE_RATE

    @property
    def duration(self):
        return len(self.audio) / self.sample_rate

    def save(self, path):
        import soundfile as sf

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), np.clip(self.audio, -1.0, 1.0), self.sample_rate,
                 subtype="PCM_16")
        return path


@dataclass
class Baseline:
    """Everything :meth:`CtrlSpeech.regenerate` needs to re-run with edits.

    ``gen_*`` describes the audio being edited (a first-pass generation, or an
    uploaded recording). ``prompt_*`` describes the voice reference, whose
    prosody is prepended to the control curves as context.
    """

    # model input
    input_ids: torch.Tensor
    text_masks: torch.Tensor
    prompt_audio: torch.Tensor
    speaker_emb: torch.Tensor
    prompt_segments: list
    # prompt-side context
    prompt_np: np.ndarray
    prompt_duration: float
    prompt_f0: np.ndarray
    prompt_loud: np.ndarray
    # the editable baseline
    gen_np: np.ndarray
    gen_f0: np.ndarray
    gen_loud: np.ndarray
    gen_phoneme_data: list
    word_data: list
    target_token_refs: list
    target_words: str
    target_phones: str
    # sampling settings used to produce it
    steps: int = DEFAULT_STEPS
    cfg_strength: float = DEFAULT_CFG_STRENGTH
    source: str = "generated"
    extras: dict = field(default_factory=dict)

    @property
    def generation(self):
        return Generation(self.gen_np, self.gen_f0, self.gen_loud)


# ─────────────────────────────────────────────────────────────────────────────
# The model
# ─────────────────────────────────────────────────────────────────────────────
class CtrlSpeech:
    """Loaded DiTar + SVAE vocoder + speaker encoder."""

    def __init__(self, model, vocoder, text_tokenizer, speaker_embedding, device,
                 controllable=True, aligner=None, progress=False):
        self.model = model
        self.vocoder = vocoder
        self.text_tokenizer = text_tokenizer
        self.speaker_embedding = speaker_embedding
        self.device = device
        self.controllable = controllable
        # Off by default: the AR progress bar is noise inside a web app or a
        # notebook. Set it on the instance for a long CLI run.
        self.progress = progress
        self._aligner = aligner

    # -- construction ----------------------------------------------------
    @classmethod
    def from_pretrained(cls, model="control-600m", device=None, repo_id=None,
                        revision=None, aligner=None, progress=False):
        assets = download_assets(model=model, repo_id=repo_id, revision=revision)
        return cls.from_assets(
            assets, device=device, aligner=aligner, progress=progress
        )

    @classmethod
    def from_assets(cls, assets, device=None, aligner=None, progress=False):
        device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        config = OmegaConf.load(assets.config_path)
        # The published config carries placeholders; only local paths differ per
        # machine, so they are resolved here rather than baked into the YAML.
        config.model.vocoder.path = str(assets.svae_dir)
        config.model.backbone.qwen_config_path = str(assets.qwen_config_path)
        config.model.backbone.pretrained_LM_path = None
        # The fine-tuned AR weights are already in the checkpoint (1.7 GB of
        # model.causalAR.*), so pulling Qwen3-0.6B would only be overwritten.
        config.model.backbone.load_pretrained_weights = False

        text_tokenizer = json.loads(assets.vocab_path.read_text(encoding="utf-8"))

        net = DiTar(config.model)
        state_dict = _load_state_dict(assets.weights_path)
        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
        # strict=False is required: the checkpoint also carries the SVAE encoder
        # under generator.*, which DiTar rebuilt from metainfo.json.
        result = net.load_state_dict(state_dict, strict=False)
        _check_control_weights(result.missing_keys, assets.spec)
        net = net.to(device).eval()

        vocoder = load_decoder(local_path=str(assets.svae_dir)).to(device).eval()
        speaker_embedding = CosyVoiceSpeakerEmbedding(
            campplus_model=str(assets.campplus_path)
        )
        return cls(
            model=net,
            vocoder=vocoder,
            text_tokenizer=text_tokenizer,
            speaker_embedding=speaker_embedding,
            device=device,
            controllable=assets.spec.controllable,
            aligner=aligner,
            progress=progress,
        )

    @property
    def aligner(self):
        if self._aligner is None:
            self._aligner = MFAAligner()
        return self._aligner

    def align(self, waveform, transcript, expected_phones=None):
        """Force-align audio against its transcript (needs MFA installed)."""
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
        return self.aligner.align(waveform, transcript, expected_phones)

    # -- pass 1 ----------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt_wav, prompt_annotation, target_annotation,
                 steps=DEFAULT_STEPS, cfg_strength=DEFAULT_CFG_STRENGTH,
                 align=True):
        """Synthesise ``target_annotation``'s text in ``prompt_wav``'s voice.

        Both annotations are 4-line dicts (words / phones / starts / ends); use
        :func:`ctrlspeech.align.read_four_line_annotation` to load them.

        ``align=False`` skips the MFA pass, which makes the result unusable for
        duration editing but avoids the dependency for plain synthesis.
        """
        input_ids, text_masks = tokenize_phones(
            prompt_annotation["phones"], target_annotation["phones"],
            self.text_tokenizer,
        )

        prompt_np, _ = librosa.load(str(prompt_wav), sr=SAMPLE_RATE, mono=True)
        prompt_np, _ = librosa.effects.trim(prompt_np, top_db=TRIM_TOP_DB)
        prompt_duration = prompt_np.shape[-1] / SAMPLE_RATE
        prompt_audio = torch.from_numpy(prompt_np).unsqueeze(0).float()
        speaker_emb = self.speaker_embedding._extract_spk_embedding(prompt_audio)[0]

        _, prompt_f0, _, prompt_loud = get_pitch_and_loudness(
            prompt_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
        )

        prompt_segments = build_prompt_segments(prompt_annotation)
        duration_segments = prompt_segments + [(0, 0)]
        for start, end in zip(target_annotation["starts"].split(" "),
                              target_annotation["ends"].split(" ")):
            if start == "|":
                duration_segments.append((0, 0))
            else:
                duration_segments.append(
                    (int((float(start) + prompt_duration) * FPS),
                     int((float(end) + prompt_duration) * FPS))
                )

        gen_np = self._sample(
            prompt_audio=prompt_audio,
            speaker_emb=speaker_emb,
            input_ids=input_ids,
            text_masks=text_masks,
            duration_segments=duration_segments,
            pitch=prompt_f0,
            loudness=prompt_loud,
            max_seq_length=estimate_max_seq_length(
                max(parse_times(target_annotation["ends"]))
            ),
            steps=steps,
            cfg_strength=cfg_strength,
        )
        _, gen_f0, _, gen_loud = get_pitch_and_loudness(
            gen_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
        )

        if align:
            alignment = self.align(
                gen_np, target_annotation["words"],
                expected_phones=target_annotation["phones"],
            )
        else:
            alignment = {"phoneme_data": [], "word_data": [], "target_token_refs": []}

        return Baseline(
            input_ids=input_ids,
            text_masks=text_masks,
            prompt_audio=prompt_audio,
            speaker_emb=speaker_emb,
            prompt_segments=prompt_segments,
            prompt_np=prompt_np,
            prompt_duration=prompt_duration,
            prompt_f0=prompt_f0,
            prompt_loud=prompt_loud,
            gen_np=gen_np,
            gen_f0=gen_f0,
            gen_loud=gen_loud,
            gen_phoneme_data=alignment["phoneme_data"],
            word_data=alignment["word_data"],
            target_token_refs=alignment["target_token_refs"],
            target_words=target_annotation["words"],
            target_phones=target_annotation["phones"],
            steps=steps,
            cfg_strength=cfg_strength,
        )

    # -- pass 1, alternative: adopt a recording as the baseline -----------
    @torch.no_grad()
    def from_audio(self, audio, transcript, steps=DEFAULT_STEPS,
                   cfg_strength=DEFAULT_CFG_STRENGTH, min_seconds=0.5,
                   max_seconds=10.0):
        """Use a real recording as the baseline, skipping the first synthesis.

        The clip supplies the voice *and* all of the baseline prosody. Editing
        one dimension reuses the clip's own values for the others, so the result
        has the same shape as :meth:`generate` and the caller need not care
        which path produced it.

        ``audio`` is a path or a mono float array already at 16 kHz.
        """
        transcript = " ".join((transcript or "").split())
        if not transcript:
            raise ValueError("A transcript of the audio is required.")

        if isinstance(audio, (str, Path)):
            audio_np, _ = librosa.load(str(audio), sr=SAMPLE_RATE, mono=True)
        else:
            audio_np = np.asarray(audio, dtype=np.float32)
        audio_np, _ = librosa.effects.trim(audio_np, top_db=TRIM_TOP_DB)

        duration = audio_np.shape[-1] / SAMPLE_RATE
        if duration < min_seconds:
            raise ValueError(
                f"The audio is only {duration:.2f}s after trimming silence; "
                f"at least {min_seconds:g}s is required."
            )
        if duration > max_seconds:
            raise ValueError(
                f"The audio is {duration:.2f}s; at most {max_seconds:g}s is "
                "supported. Trim the clip and try again."
            )

        audio_t = torch.from_numpy(audio_np).unsqueeze(0).float()
        speaker_emb = self.speaker_embedding._extract_spk_embedding(audio_t)[0]
        _, f0, _, loud = get_pitch_and_loudness(
            audio_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
        )

        # One alignment supplies both the phone tokens and the duration segments,
        # so the prompt and target token streams line up by construction.
        alignment = self.align(audio_t, transcript)
        prompt_segments = build_prompt_segments(alignment)
        input_ids, text_masks = tokenize_phones(
            alignment["phones"], alignment["phones"], self.text_tokenizer
        )

        last_end_frame = max(
            (int(round(float(item[2]) * FPS)) for item in alignment["phoneme_data"]),
            default=0,
        )
        if last_end_frame > len(f0):
            raise RuntimeError(
                "MFA alignment extends past the extracted control curve "
                f"({last_end_frame} > {len(f0)} frames)."
            )

        return Baseline(
            input_ids=input_ids,
            text_masks=text_masks,
            prompt_audio=audio_t,
            speaker_emb=speaker_emb,
            prompt_segments=prompt_segments,
            prompt_np=audio_np,
            prompt_duration=duration,
            prompt_f0=f0,
            prompt_loud=loud,
            gen_np=audio_np,
            gen_f0=f0,
            gen_loud=loud,
            gen_phoneme_data=alignment["phoneme_data"],
            word_data=alignment["word_data"],
            target_token_refs=alignment["target_token_refs"],
            target_words=transcript,
            target_phones=alignment["phones"],
            steps=steps,
            cfg_strength=cfg_strength,
            source="upload",
        )

    # -- pass 2 ----------------------------------------------------------
    def build_edited_controls(self, baseline, pitch=None, loudness=None,
                              phonemes=None):
        """Quantise the edits and rebuild token-aligned duration segments."""
        pitch = baseline.gen_f0 if pitch is None else pitch
        loudness = baseline.gen_loud if loudness is None else loudness
        phonemes = baseline.gen_phoneme_data if phonemes is None else phonemes

        edited_f0 = np.clip(np.rint(pitch), 0, PITCH_BINS - 1).astype(np.int32)
        edited_loud = np.clip(np.rint(loudness), 0, LOUDNESS_BINS - 1).astype(np.int32)
        if len(edited_f0) != len(edited_loud):
            raise ValueError("Edited pitch and loudness lengths do not match")
        if len(phonemes) != len(baseline.gen_phoneme_data):
            raise ValueError("Edited phoneme count changed")

        # Rebuild the target token segments from the (possibly retimed) phoneme
        # boundaries. The prompt offset matches the curve actually concatenated
        # below, so the two never drift apart.
        duration_segments = list(baseline.prompt_segments) + [(0, 0)]
        prompt_offset = len(baseline.prompt_f0)
        for phone_ref in baseline.target_token_refs:
            if phone_ref is None:
                duration_segments.append((0, 0))
                continue
            item = phonemes[phone_ref]
            start = int(round(float(item[1]) * FPS))
            end = int(round(float(item[2]) * FPS))
            phone_frames = end - start
            if not 1 <= phone_frames <= MAX_PHONE_FRAMES:
                raise ValueError(
                    f"phoneme {item[0]!r} lasts {phone_frames} frames, outside the "
                    f"allowed range 1..{MAX_PHONE_FRAMES}"
                )
            if start < 0 or end > len(edited_f0):
                raise ValueError(
                    f"phoneme {item[0]!r} boundary [{start}, {end}) falls outside "
                    f"the control curve [0, {len(edited_f0)})"
                )
            duration_segments.append((prompt_offset + start, prompt_offset + end))

        if len(duration_segments) != baseline.input_ids.shape[1]:
            raise ValueError(
                "Duration-token count does not match text-token count: "
                f"{len(duration_segments)} vs {baseline.input_ids.shape[1]}"
            )

        control_f0 = np.concatenate([baseline.prompt_f0, edited_f0], axis=0)
        control_loud = np.concatenate([baseline.prompt_loud, edited_loud], axis=0)
        return duration_segments, control_f0, control_loud

    @torch.no_grad()
    def regenerate(self, baseline, pitch=None, loudness=None, phonemes=None,
                   steps=None, cfg_strength=None):
        """Resynthesise the baseline under edited control curves."""
        if not self.controllable:
            raise RuntimeError(
                "This checkpoint has no pitch / loudness / duration embeddings; "
                "load a control-* model to edit prosody."
            )
        duration_segments, control_f0, control_loud = self.build_edited_controls(
            baseline, pitch, loudness, phonemes
        )
        gen_np = self._sample(
            prompt_audio=baseline.prompt_audio,
            speaker_emb=baseline.speaker_emb,
            input_ids=baseline.input_ids,
            text_masks=baseline.text_masks,
            duration_segments=duration_segments,
            pitch=control_f0,
            loudness=control_loud,
            max_seq_length=estimate_max_seq_length(
                (len(control_f0) - len(baseline.prompt_f0)) / FPS
            ),
            steps=baseline.steps if steps is None else steps,
            cfg_strength=(
                baseline.cfg_strength if cfg_strength is None else cfg_strength
            ),
        )
        _, gen_f0, _, gen_loud = get_pitch_and_loudness(
            gen_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
        )
        return Generation(gen_np, gen_f0, gen_loud)

    # -- shared sampling -------------------------------------------------
    def _sample(self, prompt_audio, speaker_emb, input_ids, text_masks,
                duration_segments, pitch, loudness, max_seq_length, steps,
                cfg_strength):
        if not self.controllable:
            # Base checkpoints never learned the prosody embeddings; feeding them
            # would add randomly initialised vectors to every text token.
            duration_segments = pitch = loudness = None

        sampled = self.model.sample(
            prompt_audio=prompt_audio.to(self.device),
            speaker_embs=speaker_emb.unsqueeze(0).to(self.device),
            text_inputs=input_ids.to(self.device),
            text_masks=text_masks.to(self.device),
            duration_segments=None if duration_segments is None else [duration_segments],
            pitch=None if pitch is None else [pitch],
            loudness=None if loudness is None else [loudness],
            use_cache=True,
            max_seq_length=max_seq_length,
            steps=steps,
            cfg_strength=cfg_strength,
            progress=self.progress,
        )
        audio = self.vocoder.decode(sampled.float()).squeeze(0).detach().cpu()
        return audio.squeeze().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_state_dict(path):
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    return checkpoint.get("state_dict", checkpoint)


_CONTROL_KEYS = ("pitch_embedding", "loudness_embedding", "duration_embedding")


def _check_control_weights(missing, spec):
    """Fail loudly if a control checkpoint arrived without its prosody weights.

    Because the load has to be non-strict, a truncated or mismatched file would
    otherwise leave the prosody embeddings randomly initialised — the model
    would still run and just quietly ignore every control curve.
    """
    if not spec.controllable:
        return
    absent = [key for key in _CONTROL_KEYS if any(key in name for name in missing)]
    if absent:
        raise RuntimeError(
            f"{spec.key} is a controllable checkpoint but these weights are "
            f"missing: {', '.join(absent)}. The weights file looks incomplete."
        )
