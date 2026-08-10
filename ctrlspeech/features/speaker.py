"""
This script is used to extract the speaker embedding from the speech.
Source: 
"""
import onnxruntime
import torch

import torchaudio.compliance.kaldi as kaldi


class CosyVoiceSpeakerEmbedding:
    def __init__(self, campplus_model: str):  
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_model, sess_options=option, providers=["CPUExecutionProvider"]
        )

    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(
            speech,
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None, {
            self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()
        })[0].flatten().tolist()
        embedding = torch.tensor([embedding])
        return embedding

