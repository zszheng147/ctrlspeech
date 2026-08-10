"""CtrlSpeech — controllable expressive TTS with coarse-to-fine latent control."""

from .assets import MODELS, Assets, ModelSpec, download_assets
from .pipeline import (
    FPS,
    HOP_LENGTH,
    LOUDNESS_BINS,
    PITCH_BINS,
    SAMPLE_RATE,
    Baseline,
    CtrlSpeech,
    Generation,
    shift_loudness_db,
    shift_pitch_semitones,
)

__version__ = "0.1.0"

__all__ = [
    "Assets",
    "Baseline",
    "CtrlSpeech",
    "FPS",
    "Generation",
    "HOP_LENGTH",
    "LOUDNESS_BINS",
    "MODELS",
    "ModelSpec",
    "PITCH_BINS",
    "SAMPLE_RATE",
    "download_assets",
    "shift_loudness_db",
    "shift_pitch_semitones",
]
