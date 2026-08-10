import os
import json
import torch

from .vae.dac.model import DAC

# load vocoder
def load_decoder(local_path="", use_ema=True):
    decoder_dict = {
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
    
    metainfo_path = os.path.join(local_path, "metainfo.json")
    with open(metainfo_path, "r") as f:
        metainfo = json.load(f)

    if not use_ema:
        ckpt_path = os.path.join(local_path, "dac", "weights.pth")
        model_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        filter_dict = {
            k: v
            for k, v in model_dict["state_dict"].items()
            if not k.startswith("projectors") and not k.startswith("encoder")
        }
    else:
        ckpt_path = os.path.join(local_path, "dac", "ema_state_dict.pth")
        model_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        ckpt_dict = {k.replace("ema_model.",""):v for k,v in model_dict.items()}
        filter_dict = {
            k: v for k, v in ckpt_dict.items()
            if not k.startswith("projectors") and not k.startswith("encoder")
        }
    
    decoder = DAC(**metainfo["DAC"], decoder_dict=decoder_dict)
    del decoder.projectors, decoder.encoder
    # vocoder.load_state_dict(filter_dict, strict=True)
    msg = decoder.load_state_dict(filter_dict, strict=False)
    unexpected = [k for k in msg.unexpected_keys if k not in ("initted", "step")]
    if msg.missing_keys or unexpected:
        print(f"[vocoder] missing={msg.missing_keys} unexpected={unexpected}")
    return decoder