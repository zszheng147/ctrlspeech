"""Resolve the weights and support files a CtrlSpeech model needs.

Everything is fetched from one Hugging Face repo, laid out as::

    control-600m/{config.yaml, model.safetensors, qwen_config.json}
    control-150m/{config.yaml, model.safetensors, qwen_config.json}
    base-600m/...
    base-150m/...
    svae/{metainfo.json, config.json, dac/{ema_state_dict.pth, weights.pth}}
    shared/{vocab.json, campplus.onnx}

Set ``CTRLSPEECH_ASSETS`` to a directory with that same layout to work fully
offline; nothing is downloaded then.
"""

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REPO_ID = os.environ.get("CTRLSPEECH_HF_REPO", "zhisheng01/CtrlSpeech")


@dataclass(frozen=True)
class ModelSpec:
    """One published checkpoint."""

    key: str
    folder: str
    params: str
    controllable: bool
    description: str


MODELS = {
    spec.key: spec
    for spec in [
        ModelSpec(
            key="control-600m",
            folder="control-600m",
            params="692M",
            controllable=True,
            description="Pitch / loudness / duration control. The demo default.",
        ),
        ModelSpec(
            key="control-150m",
            folder="control-150m",
            params="150M",
            controllable=True,
            description="Smaller controllable model for tighter GPU budgets.",
        ),
        ModelSpec(
            key="base-600m",
            folder="base-600m",
            params="689M",
            controllable=False,
            description="Zero-shot TTS without prosody control (ablation baseline).",
        ),
        ModelSpec(
            key="base-150m",
            folder="base-150m",
            params="148M",
            controllable=False,
            description="Smaller uncontrolled baseline.",
        ),
    ]
}


@dataclass(frozen=True)
class Assets:
    """Absolute paths to everything ``CtrlSpeech.from_pretrained`` needs."""

    spec: ModelSpec
    root: Path
    config_path: Path
    weights_path: Path
    qwen_config_path: Path
    svae_dir: Path
    vocab_path: Path
    campplus_path: Path

    @property
    def controllable(self):
        return self.spec.controllable


def _layout(root, spec):
    root = Path(root)
    folder = root / spec.folder
    weights = folder / "model.safetensors"
    if not weights.exists():
        # Accept the raw Lightning export too, so a local checkpoint directory
        # works before anything has been converted.
        legacy = folder / "model.ckpt"
        if legacy.exists():
            weights = legacy
    return Assets(
        spec=spec,
        root=root,
        config_path=folder / "config.yaml",
        weights_path=weights,
        qwen_config_path=folder / "qwen_config.json",
        svae_dir=root / "svae",
        vocab_path=root / "shared" / "vocab.json",
        campplus_path=root / "shared" / "campplus.onnx",
    )


def _verify(assets):
    missing = [
        str(path)
        for path in (
            assets.config_path,
            assets.weights_path,
            assets.qwen_config_path,
            assets.svae_dir / "metainfo.json",
            assets.vocab_path,
            assets.campplus_path,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "CtrlSpeech assets are incomplete under "
            f"{assets.root}:\n  " + "\n  ".join(missing)
        )
    return assets


def download_assets(model="control-600m", repo_id=None, revision=None, cache_dir=None):
    """Return local paths for ``model``, downloading from the Hub if needed.

    Only the requested model folder plus the shared files are fetched, so
    picking ``control-150m`` does not pull the 600M weights.
    """
    if model not in MODELS:
        raise ValueError(
            f"Unknown model {model!r}. Available: {', '.join(sorted(MODELS))}"
        )
    spec = MODELS[model]

    local_root = os.environ.get("CTRLSPEECH_ASSETS")
    if local_root:
        return _verify(_layout(local_root, spec))

    from huggingface_hub import snapshot_download

    root = snapshot_download(
        repo_id=repo_id or DEFAULT_REPO_ID,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=[f"{spec.folder}/*", "svae/*", "shared/*"],
    )
    return _verify(_layout(root, spec))
