"""Command-line CtrlSpeech: synthesise a baseline, then optionally control it.

Two ways to get a baseline:

  ``--prompt-wav`` + ``--target-wav``
      Synthesise the target sentence in the prompt's voice. The target recording
      supplies the reference timing for its own text; it is not used for voice.

  ``--audio``
      Adopt an existing recording as the baseline. No first synthesis happens,
      so the clip's own pitch, loudness and word timings become the starting
      point and whatever you do not edit is reused verbatim.

Any of the three controls can then be applied, and the model resynthesises.
Installed as the ``ctrlspeech`` command; from a checkout, run
``python scripts/generate.py`` instead.

    # +5 semitones over the whole utterance, keeping timing and loudness
    ctrlspeech --audio clip.wav --transcript-text "..." \\
        --pitch-shift 5 --out out.wav

    # stretch one word to twice its length
    ctrlspeech --audio clip.wav --transcript-text "..." \\
        --stretch-word dreams --stretch-ratio 2 --out out.wav

Transcripts may be passed inline (``--transcript-text``) or as a file
(``--transcript``). Every path that needs phoneme boundaries goes through MFA.
"""

import argparse
from pathlib import Path

import numpy as np

from .align import annotate_audio, normalize_word
from .assets import MODELS
from .pipeline import (
    HOP_LENGTH,
    SAMPLE_RATE,
    CtrlSpeech,
    shift_loudness_db,
    shift_pitch_semitones,
)
from .retime import retime_word_curves


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="control-600m", choices=sorted(MODELS))
    p.add_argument("--device", default=None, help="cuda, cuda:1, cpu (default: auto)")
    p.add_argument("--out", type=Path, required=True)

    src = p.add_argument_group("baseline from an existing recording")
    src.add_argument("--audio", type=Path, help="Clip to adopt as the baseline.")
    src.add_argument("--transcript", type=Path, help="Text file for --audio.")
    src.add_argument("--transcript-text", help="Inline text for --audio.")

    syn = p.add_argument_group("baseline by synthesis")
    syn.add_argument("--prompt-wav", type=Path, help="Voice reference.")
    syn.add_argument("--prompt-text", type=Path, help="Transcript of --prompt-wav.")
    syn.add_argument("--target-wav", type=Path, help="Timing reference for the text.")
    syn.add_argument("--target-text", type=Path, help="Transcript of --target-wav.")

    ctl = p.add_argument_group("controls (applied to the baseline)")
    ctl.add_argument("--pitch-shift", type=float, default=0.0, help="Semitones.")
    ctl.add_argument("--loudness-shift", type=float, default=0.0, help="dB.")
    ctl.add_argument("--stretch-word", help="Word whose duration to change.")
    ctl.add_argument("--stretch-ratio", type=float, help="Multiplier, e.g. 2.0.")
    ctl.add_argument("--stretch-seconds", type=float, help="Absolute target length.")

    p.add_argument("--steps", type=int, default=32, help="Flow-matching ODE steps.")
    p.add_argument("--cfg-strength", type=float, default=1.5)
    p.add_argument(
        "--save-baseline", type=Path,
        help="Also write the un-edited baseline audio here.",
    )
    return p.parse_args()


def read_transcript(path, inline, what):
    if inline:
        return " ".join(inline.split())
    if path:
        return " ".join(Path(path).read_text(encoding="utf-8").split())
    raise SystemExit(f"A transcript is required for {what}.")


def find_word(words, wanted):
    target = normalize_word(wanted)
    for idx, word in enumerate(words):
        if normalize_word(str(word["label"])) == target:
            return idx
    labels = ", ".join(str(w["label"]) for w in words)
    raise SystemExit(f"Word {wanted!r} is not in the utterance. Words: {labels}")


def main():
    args = parse_args()

    has_recording = args.audio is not None
    has_synthesis = args.prompt_wav is not None
    if has_recording == has_synthesis:
        raise SystemExit("Pass either --audio or --prompt-wav/--target-wav, not both.")

    wants_control = bool(
        args.pitch_shift or args.loudness_shift or args.stretch_word
    )
    if wants_control and not MODELS[args.model].controllable:
        raise SystemExit(
            f"{args.model} has no prosody embeddings; use a control-* model to "
            "apply pitch / loudness / duration edits."
        )

    print(f"Loading {args.model} …", flush=True)
    tts = CtrlSpeech.from_pretrained(args.model, device=args.device, progress=True)

    if has_recording:
        transcript = read_transcript(args.transcript, args.transcript_text, "--audio")
        print(f"Adopting {args.audio.name} as the baseline (aligning with MFA) …")
        baseline = tts.from_audio(
            args.audio, transcript, steps=args.steps, cfg_strength=args.cfg_strength
        )
    else:
        if args.target_wav is None:
            raise SystemExit("--target-wav is required alongside --prompt-wav.")
        prompt_text = read_transcript(args.prompt_text, None, "--prompt-wav")
        target_text = read_transcript(args.target_text, None, "--target-wav")
        print("Aligning the prompt and target references with MFA …")
        prompt_annotation = annotate_audio(args.prompt_wav, prompt_text, tts.aligner)
        target_annotation = annotate_audio(args.target_wav, target_text, tts.aligner)
        print("Generating the baseline …")
        baseline = tts.generate(
            args.prompt_wav, prompt_annotation, target_annotation,
            steps=args.steps, cfg_strength=args.cfg_strength,
            align=wants_control,
        )

    print(
        f"Baseline: {len(baseline.gen_np) / SAMPLE_RATE:.2f}s, "
        f"{len(baseline.word_data)} words aligned."
    )
    if args.save_baseline:
        baseline.generation.save(args.save_baseline)
        print(f"  wrote {args.save_baseline}")

    if not wants_control:
        baseline.generation.save(args.out)
        print(f"No control requested; wrote the baseline to {args.out}")
        return 0

    pitch = np.asarray(baseline.gen_f0, dtype=float)
    loudness = np.asarray(baseline.gen_loud, dtype=float)
    phonemes = baseline.gen_phoneme_data

    if args.pitch_shift:
        pitch = shift_pitch_semitones(pitch, args.pitch_shift)
        print(f"Pitch: {args.pitch_shift:+g} semitones")
    if args.loudness_shift:
        loudness = shift_loudness_db(loudness, args.loudness_shift)
        print(f"Loudness: {args.loudness_shift:+g} dB")

    if args.stretch_word:
        if not baseline.word_data:
            raise SystemExit("Duration editing needs an MFA alignment.")
        idx = find_word(baseline.word_data, args.stretch_word)
        word = baseline.word_data[idx]
        old = float(word["end"] - word["start"])
        if args.stretch_seconds is not None:
            new = args.stretch_seconds
        elif args.stretch_ratio is not None:
            new = old * args.stretch_ratio
        else:
            raise SystemExit(
                "--stretch-word needs --stretch-ratio or --stretch-seconds."
            )
        pitch, loudness, phonemes, words, summary = retime_word_curves(
            pitch, loudness, phonemes, baseline.word_data, idx, new,
            hop_length=HOP_LENGTH, sample_rate=SAMPLE_RATE,
        )
        print(
            f"Duration: {summary['label']} {summary['old_duration']:.2f}s -> "
            f"{summary['new_duration']:.2f}s ({summary['scale']:.2f}x), "
            f"timeline now {summary['total_frames']} frames"
        )

    print("Regenerating under the edited controls …")
    result = tts.regenerate(baseline, pitch=pitch, loudness=loudness, phonemes=phonemes)
    result.save(args.out)
    print(f"Wrote {args.out} ({result.duration:.2f}s)")

    if args.stretch_word:
        achieved = tts.align(result.audio, baseline.target_words)
        got = achieved["word_data"][idx]
        print(
            f"Verified: {got['label']} lasts {got['end'] - got['start']:.2f}s "
            f"(requested {summary['new_duration']:.2f}s)"
        )
    elif args.pitch_shift:
        voiced = lambda c: float(np.mean([v for v in c if v > 1]) or 0)
        print(
            f"Voiced pitch mean: {voiced(baseline.gen_f0):.1f} -> "
            f"{voiced(result.pitch):.1f} bins (requested {voiced(pitch):.1f})"
        )
    elif args.loudness_shift:
        print(
            f"Loudness mean: {np.mean(baseline.gen_loud):.1f} -> "
            f"{np.mean(result.loudness):.1f} bins "
            f"(requested {np.mean(loudness):.1f})"
        )
    return 0
