"""SVAE encoder used to turn waveforms into the latents DiTar models.

Only the encoder half is kept: ``load_state`` builds the architecture and drops
the decoder branches, and the weights arrive later from the DiTar checkpoint
(they live under ``model.generator.*``). Waveform reconstruction goes through
``ctrlspeech.models.vocoder.load_decoder`` instead.
"""

import json
from pathlib import Path

import torch

from .dac.model import DAC

# BigVGAN decoder hyper-parameters. They are not read from metainfo.json because
# the checkpoint stores only the DAC/Discriminator sections.
DECODER_DICT = {
    "resblock": "1",
    "num_mels": 64,
    "upsample_rates": [5, 5, 2, 2, 2, 2],
    "upsample_kernel_sizes": [9, 9, 4, 4, 4, 4],
    "upsample_initial_channel": 1024,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "use_tanh_at_final": False,
    "use_bias_at_final": False,
    "activation": "snakebeta",
    "snake_logscale": True,
}


def load_state(save_path: str, tag: str = "latest", use_ema: bool = True):
    """Build the SVAE encoder described by ``<save_path>/<tag>/metainfo.json``.

    ``use_ema`` is accepted for call-site compatibility and ignored: no weights
    are loaded here.
    """
    folder = Path(save_path).expanduser() / tag
    metainfo_path = (folder / "metainfo.json").resolve()
    if not metainfo_path.exists():
        raise FileNotFoundError(
            f"SVAE metainfo.json not found at {metainfo_path}. Point the config's "
            "model.vocoder.path at the directory that holds the svae folder."
        )
    with open(metainfo_path, "r") as f:
        metainfo = json.load(f)

    generator = DAC(**metainfo["DAC"], decoder_dict=DECODER_DICT)
    del generator.projectors
    del generator.decoder
    del generator.decoder_proj
    generator.eval()
    return generator


@torch.no_grad()
def process_online(signal, generator, **kwargs):
    """Encode a waveform batch into pre-projection VAE latents (B, D, T)."""
    audio_data = generator.preprocess(signal, generator.sample_rate)
    latent, mu, log_var, kl_loss = generator.encode(audio_data)
    pre_proj_latent = generator.reparameterize(mu, log_var)
    return pre_proj_latent.transpose(1, 2)
