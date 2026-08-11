# CtrlSpeech: Coarse-to-Fine Control for Expressive Speech Synthesis.

<p align="left">
  <a href="https://arxiv.org/abs/2608.08362"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/zhisheng01/CtrlSpeech"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffcc00" alt="Hugging Face"></a>
  <a href="https://zhishengzheng.com/ctrlspeech/"><img src="https://img.shields.io/badge/Demo-Samples-1f8acb" alt="Demo"></a>
</p>

CtrlSpeech is a zero-shot TTS model you can *steer after the fact*. Generate a
sentence, read back its pitch contour, loudness contour and phoneme boundaries,
change one of them, and resynthesise — the model follows the edit and leaves
everything else alone.

The prosody is conditioned per phoneme token, not per utterance, which is why a
single word can be stretched to twice its length while the rest of the sentence
keeps its original timing.

```
prompt voice ─┐
              ├─► DiTar (AR + LocDiT flow matching) ─► SVAE latents ─► waveform
target text ──┘         ▲
                        │  per-token pitch (128 bins) / loudness (64 bins) /
                        │  duration (frames) embeddings
                        └── edited by you between pass 1 and pass 2
```

---

## Install

```bash
conda create -n ctrlspeech python=3.11 -y
conda activate ctrlspeech

# Match torch to your CUDA version first
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

pip install -e .              # or: pip install -r requirements.txt
```

That single install covers the library, the CLI and the Panel demo.

**Montreal Forced Aligner** is required for anything that derives phoneme
boundaries from audio (duration editing, and adopting your own recording as a
baseline). It is conda-only:

```bash
conda install -c conda-forge montreal-forced-aligner
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

Plain synthesis from a pre-aligned annotation works without MFA.

Weights download automatically from the Hub on first use (~3.5 GB for one
model). To point at a local copy instead, set `CTRLSPEECH_ASSETS` to a directory
laid out like the Hub repo.

---

## Quick start

### Python

```python
from ctrlspeech import CtrlSpeech, shift_pitch_semitones

tts = CtrlSpeech.from_pretrained("control-600m")

# Adopt a real recording as the baseline — no first synthesis needed.
baseline = tts.from_audio("clip.wav", "If you dream a thing more than once, "
                                      "it's sure to come true.")

# Raise the pitch by 5 semitones; loudness and timing come from the clip.
result = tts.regenerate(baseline, pitch=shift_pitch_semitones(baseline.gen_f0, 5))
result.save("higher.wav")
```

### Command line

```bash
# Raise pitch, keep everything else
ctrlspeech --audio clip.wav --transcript-text "..." --pitch-shift 5 --out out.wav

# Stretch one word to 2x, then verify the result with MFA
ctrlspeech --audio clip.wav --transcript-text "..." \
    --stretch-word dreams --stretch-ratio 2 --out out.wav

# Full two-pass synthesis: prompt supplies the voice, the target recording
# supplies reference timing for its own text
ctrlspeech --prompt-wav demo/assets/dreams-prompt.wav --prompt-text demo/assets/dreams-prompt.txt \
    --target-wav demo/assets/dreams-target.wav --target-text demo/assets/dreams-target.txt \
    --loudness-shift 8 --out out.wav
```

From a checkout without installing, use `python scripts/generate.py` instead of
`ctrlspeech`.

### Interactive demo

```bash
panel serve demo/app.py --show --port 5006
```

Over SSH, forward the port from your laptop (`ssh -L 5006:localhost:5006 host`)
and start the server with `--allow-websocket-origin=localhost:5006`, then open
<http://localhost:5006/app>.

The demo lets you draw pitch and loudness contours freehand, drag word
boundaries, and compare *baseline vs. requested vs. achieved* after
regenerating. Drawing is a three-step cycle: **Draw** arms the gesture,
you draw, **Finish** locks the contour in. Before Draw (and after Finish),
dragging pans the plot.

`panel serve` does not hot-reload imported modules — restart it after editing
`ctrlspeech/` or `demo/interactive_plot.py`.

---

## Models

| Model | Params | Prosody control | Notes |
|---|---|---|---|
| `control-600m` | 692 M | pitch · loudness · duration | Default; used by the demo |
| `control-150m` | 150 M | pitch · loudness · duration | For tighter GPU budgets |
| `base-600m` | 689 M | — | Zero-shot TTS ablation baseline |
| `base-150m` | 148 M | — | Smaller ablation baseline |

The `base-*` checkpoints never learned the prosody embeddings; asking them for a
control edit raises an error rather than silently ignoring it.

All four sit in one Hub repo alongside the shared SVAE vocoder, CAM++ speaker
encoder and phoneme vocabulary. Only the model you ask for is downloaded.

---

## How the control works

Audio is analysed at **100 frames per second** (16 kHz, hop 160).

- **Pitch** — F0 mapped to 128 mel-spaced bins; bin 0 means unvoiced. Slider
  shifts skip unvoiced frames so silence is not given a pitch.
- **Loudness** — A-weighted dB in 64 bins, about 1.05 bins per dB.
- **Duration** — per-phoneme frame counts. `duration_embedding` has 192 entries,
  so one phoneme spans at most **191 frames (1.91 s)**; an edited timeline is
  capped at **2001 frames (20 s)**. Both limits live in `ctrlspeech/retime.py`.

Stretching a word rescales its phoneme boundaries uniformly and shifts
everything after it, so inter-word pauses keep their original length.

The AR model emits `patch_size=4` SVAE latents per step at 40 Hz, i.e. **0.1 s
per step**. The step budget is derived from the requested duration
(`estimate_max_seq_length`) rather than fixed, so a stretched sentence is not
truncated.

---

## Environment variables

| Variable | Purpose |
|---|---|
| `CTRLSPEECH_ASSETS` | Use a local asset directory; skips all downloads |
| `CTRLSPEECH_HF_REPO` | Override the Hub repo id |
| `CTRLSPEECH_MFA_CACHE` | Where MFA scratch files go (default `~/.cache/ctrlspeech/mfa`) |
| `CTRLSPEECH_MFA_DICT` | Path to `english_us_arpa.dict` |
| `CTRLSPEECH_DEMO_MODEL` | Which model the demo loads (default `control-600m`) |

MFA alignment uses a **sentence-level mini dictionary** built on the fly, and
falls back to the full 200k-word lexicon only when a word is out of vocabulary.
That is the difference between ~15 s and ~3 min per alignment; the fallback also
needs about 500 MB of scratch space, so point `CTRLSPEECH_MFA_CACHE` at a disk
with room.

---

## Repository layout

```
ctrlspeech/
  pipeline.py    CtrlSpeech: generate / from_audio / regenerate
  assets.py      Hub resolution for weights and support files
  retime.py      word-level retiming; owns the duration limits
  cli.py         the `ctrlspeech` command
  align/         MFA wrapper, 4-line annotation format
  features/      pitch, loudness and speaker-embedding extraction
  models/        DiTar, Qwen3 backbone, LocDiT, SVAE/DAC vocoder
demo/
  app.py             Panel application
  interactive_plot.py Bokeh contour and phoneme editor
  assets/            bundled example clips + examples.json
scripts/
  generate.py        CLI entry point for a checkout
tests/
  test_demo_flow.py  headless check of all three control paths
```

Adding a demo example means dropping a wav plus a plain-text transcript into
`demo/assets/` and listing it in `examples.json`; the phoneme timings are
force-aligned on first use and cached next to the clip.

---

## Licence

The code in this repository is **MIT** (see [LICENSE](LICENSE)). The published **weights are CC BY-NC 4.0 — non-commercial**. 


### Intended use

These models clone a speaker's voice from a few seconds of reference audio. Use
only recordings you have the right to use, disclose synthetic speech as
synthetic, and do not impersonate anyone without their consent.

## Citation
```
@inproceedings{zheng2026ctrlspeech,
  title     = {CtrlSpeech: Coarse-to-Fine Control for Expressive Speech Synthesis},
  author    = {Zheng, Zhisheng and Sun, Xiaohang and Liu, Zhu and Chen, Caren and Kumar, Rohith and Aggarwal, Manoj and Medioni, Gerard and Harwath, David},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
}
```
