"""Montreal Forced Aligner wrapper.

CtrlSpeech conditions every *phoneme token* on a pitch, loudness and duration
value, so editing prosody means knowing where each phoneme starts and ends.
That is what MFA provides here: given a waveform and its transcript, it returns
phone-level boundaries in the same 4-line layout the training annotations use::

    words   : If you dream a thing more than once ...
    phones  : IH F | Y UW | D R IY M | ...
    starts  : 0.040000 0.120000 | 0.210000 0.270000 | ...
    ends    : 0.120000 0.210000 | 0.270000 0.300000 | ...

MFA is only needed when boundaries have to be *derived* from audio. If you
already have a 4-line annotation, ``read_four_line_annotation`` is enough and
MFA never runs.

Speed note: aligning against the full 200k-word ``english_us_arpa`` dictionary
recompiles a ~500 MB lexicon FST on every call (~3 minutes). This module writes
a sentence-level mini dictionary instead and falls back to the full one only
when a word is out of vocabulary, which takes alignment down to ~15 seconds.
"""

import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torchaudio

SAMPLE_RATE = 16000

# Punctuation stripped from the *edges* of a word before dictionary lookup.
# The apostrophe is included so 'urgent' (wrapped in straight quotes) resolves,
# while word-internal apostrophes in storm's / don't survive.
STRIP_PUNCT = ".,!?;:\"'()[]{}…—-–「」『』“”‘’"

DEFAULT_CACHE = Path(
    os.environ.get(
        "CTRLSPEECH_MFA_CACHE", Path.home() / ".cache" / "ctrlspeech" / "mfa"
    )
)
# Where `mfa model download dictionary english_us_arpa` puts the lexicon.
DEFAULT_DICT = Path(
    os.environ.get(
        "CTRLSPEECH_MFA_DICT",
        Path.home() / "Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict",
    )
)


def read_four_line_annotation(path):
    """Read a words / phones / starts / ends annotation file."""
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    if len(lines) != 4:
        raise ValueError(f"Expected 4 lines in {path}, got {len(lines)}")
    words, phones, starts, ends = lines
    return {"words": words, "phones": phones, "starts": starts, "ends": ends}


def normalize_word(word):
    return word.lower().strip().strip(STRIP_PUNCT)


def annotate_audio(wav_path, transcript, aligner=None, cache=True):
    """Turn a clip plus a plain transcript into a 4-line annotation.

    The alignment runs on the *trimmed* waveform, because the pipeline trims
    prompts the same way; otherwise the phone timings and the extracted control
    curves would sit on two different time bases.

    The result is cached next to the clip as ``<name>.aligned.txt`` and reused
    while both the audio and the transcript are older than the cache.
    """
    import librosa
    import torch

    wav_path = Path(wav_path)
    transcript = " ".join(transcript.split())
    cache_path = wav_path.with_suffix(".aligned.txt")

    if cache and cache_path.exists():
        fresh = cache_path.stat().st_mtime >= wav_path.stat().st_mtime
        if fresh:
            annotation = read_four_line_annotation(cache_path)
            if annotation["words"] == transcript:
                return annotation

    audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=40)
    alignment = (aligner or MFAAligner()).align(
        torch.from_numpy(audio).float(), transcript
    )
    annotation = {
        "words": transcript,
        "phones": alignment["phones"],
        "starts": alignment["starts"],
        "ends": alignment["ends"],
    }
    if cache:
        cache_path.write_text(
            "\n".join(annotation[k] for k in ("words", "phones", "starts", "ends"))
            + "\n",
            encoding="utf-8",
        )
    return annotation


def read_mfa_csv(csv_path, words_text=None, expected_phones=None):
    """Parse an MFA CSV export into the 4-line layout plus editing metadata.

    Times stay in seconds here; quantisation to the model's 100 fps grid happens
    at the last moment, in the pipeline. ``target_token_refs`` lines up 1:1 with
    the target phone tokens, with ``None`` marking a ``|`` word separator.
    """
    df = pd.read_csv(csv_path)
    words_df = df[df["Type"] == "words"].sort_values("Begin").reset_index(drop=True)
    phones_df = df[df["Type"] == "phones"].sort_values("Begin").reset_index(drop=True)
    if words_df.empty or phones_df.empty:
        raise RuntimeError(f"MFA alignment empty: {csv_path}")

    # Keep the original capitalisation for display when the word count agrees;
    # awkward punctuation makes that unreliable, so fall back to MFA's labels.
    display_words = []
    if words_text:
        candidates = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)*", words_text)
        if len(candidates) == len(words_df):
            display_words = candidates

    phone_begins = phones_df["Begin"].to_numpy(dtype=float)
    phone_ends = phones_df["End"].to_numpy(dtype=float)
    phone_midpoints = (phone_begins + phone_ends) / 2.0
    phone_labels = phones_df["Label"].astype(str).to_numpy()

    phoneme_data = []
    word_data = []
    target_token_refs = []
    phone_groups, begin_groups, end_groups = [], [], []
    used = np.zeros(len(phones_df), dtype=bool)

    for word_idx, row in words_df.iterrows():
        wb, we = float(row["Begin"]), float(row["End"])
        # Assigning by midpoint sidesteps the float rounding on CSV boundaries.
        indices = np.flatnonzero(
            (phone_midpoints >= wb - 1e-6) & (phone_midpoints <= we + 1e-6) & ~used
        )
        if len(indices) == 0:
            raise RuntimeError(f"MFA word {row['Label']!r} has no matching phoneme")
        used[indices] = True

        phone_start = len(phoneme_data)
        labels, begins, ends = [], [], []
        for phone_row_idx in indices:
            label = re.sub(r"\d", "", str(phone_labels[phone_row_idx]))
            begin = float(phone_begins[phone_row_idx])
            end = float(phone_ends[phone_row_idx])
            labels.append(label)
            begins.append(begin)
            ends.append(end)
            phoneme_data.append([label, begin, end])
            target_token_refs.append(len(phoneme_data) - 1)

        word_data.append(
            {
                "label": display_words[word_idx] if display_words else str(row["Label"]),
                "start": wb,
                "end": we,
                "phone_start": phone_start,
                "phone_end": len(phoneme_data),
            }
        )
        phone_groups.append(" ".join(labels))
        begin_groups.append(" ".join(f"{v:.6f}" for v in begins))
        end_groups.append(" ".join(f"{v:.6f}" for v in ends))
        if word_idx < len(words_df) - 1:
            target_token_refs.append(None)

    # Token order has to match the model input, or a duration edit lands on the
    # wrong phoneme.
    if expected_phones:
        expected_groups = [
            [re.sub(r"\d", "", phone) for phone in group.split()]
            for group in expected_phones.split(" | ")
        ]
        actual_groups = [group.split() for group in phone_groups]
        if len(expected_groups) != len(actual_groups):
            raise RuntimeError(
                "MFA-aligned word count does not match the target annotations: "
                f"{len(actual_groups)} vs {len(expected_groups)}"
            )
        count_mismatches = [
            i
            for i, (expected, actual) in enumerate(zip(expected_groups, actual_groups))
            if len(expected) != len(actual)
        ]
        if count_mismatches:
            first = count_mismatches[0]
            raise RuntimeError(
                "Per-word MFA phoneme counts do not match the target annotations; "
                "duration cannot be edited safely: "
                f"word {first + 1}, expected={expected_groups[first]}, "
                f"aligned={actual_groups[first]}"
            )
        pronunciation_mismatches = [
            i
            for i, (expected, actual) in enumerate(zip(expected_groups, actual_groups))
            if expected != actual
        ]
    else:
        pronunciation_mismatches = []

    return {
        "phones": " | ".join(phone_groups),
        "starts": " | ".join(begin_groups),
        "ends": " | ".join(end_groups),
        "phoneme_data": phoneme_data,
        "word_data": word_data,
        "target_token_refs": target_token_refs,
        # Same phone count but different labels is usually a legal pronunciation
        # variant (AA vs AO); positional mapping still holds.
        "pronunciation_mismatches": pronunciation_mismatches,
    }


class MFAAligner:
    """Force-align 16 kHz waveforms with MFA's ``english_us_arpa`` models."""

    def __init__(self, cache_dir=None, dictionary_path=None, acoustic_model="english_us_arpa"):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE)
        self.dictionary_path = Path(dictionary_path or DEFAULT_DICT)
        self.acoustic_model = acoustic_model
        self._full_dict = None
        self._lock = threading.Lock()

    # -- dictionary ------------------------------------------------------
    def _load_full_dict(self):
        """english_us_arpa.dict -> {word: [pronunciation, ...]}.

        The phones are always the final tab-separated field; the optional
        columns in between are pronunciation probabilities.
        """
        if not self.dictionary_path.exists():
            raise FileNotFoundError(
                f"MFA dictionary not found at {self.dictionary_path}. Run "
                "`mfa model download dictionary english_us_arpa`, or set "
                "CTRLSPEECH_MFA_DICT to its location."
            )
        table = {}
        with open(self.dictionary_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                table.setdefault(parts[0], []).append(parts[-1])
        return table

    def _build_mini_dict(self, words_text, out_path):
        """Write a dictionary holding only this sentence's words.

        Returns the out-of-vocabulary words; a non-empty list means the caller
        must fall back to the full dictionary.
        """
        if self._full_dict is None:
            self._full_dict = self._load_full_dict()
        seen, lines, missing = set(), [], []
        for raw in words_text.split():
            word = normalize_word(raw)
            if not word or word in seen:
                continue
            seen.add(word)
            pronunciations = self._full_dict.get(word)
            if not pronunciations:
                missing.append(word)
                continue
            lines.extend(f"{word}\t{pron}" for pron in pronunciations)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return missing

    # -- alignment -------------------------------------------------------
    def align(self, waveform, transcript, expected_phones=None):
        """Align a mono 16 kHz ``torch.Tensor`` against ``transcript``."""
        mfa_bin = shutil.which("mfa")
        if mfa_bin is None:
            raise RuntimeError(
                "The `mfa` command was not found. Install it with "
                "`conda install -c conda-forge montreal-forced-aligner`, then "
                "`mfa model download acoustic english_us_arpa` and "
                "`mfa model download dictionary english_us_arpa`."
            )

        wav = waveform
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        with self._lock:
            corpus = self.cache_dir / "corpus"
            temp_root = self.cache_dir / "tmp"
            aligned = self.cache_dir / "aligned"
            mini_dict = self.cache_dir / "mini.dict"
            shutil.rmtree(corpus, ignore_errors=True)
            shutil.rmtree(aligned, ignore_errors=True)
            corpus.mkdir(parents=True, exist_ok=True)
            temp_root.mkdir(parents=True, exist_ok=True)

            torchaudio.save(str(corpus / "audio.wav"), wav, SAMPLE_RATE)
            (corpus / "audio.txt").write_text(transcript, encoding="utf-8")

            missing = self._build_mini_dict(transcript, mini_dict)
            dictionary = self.acoustic_model if missing else str(mini_dict)

            cmd = [
                mfa_bin, "align", "--clean", "-j", "1", "--output_format", "csv",
                str(corpus), dictionary, self.acoustic_model, str(aligned),
                "-t", str(temp_root),
            ]
            # MFA prints non-ASCII progress characters; an ASCII locale would
            # otherwise raise UnicodeDecodeError before we ever see the error.
            proc = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            if proc.returncode != 0:
                hint = ""
                if "FstIOError" in proc.stderr or "Write failed" in proc.stderr:
                    hint = (
                        "\n(The disk holding CTRLSPEECH_MFA_CACHE may be full; the "
                        "full-dictionary FST needs ~500 MB.)"
                    )
                raise RuntimeError(f"MFA alignment failed:{hint}\n{proc.stderr[-1500:]}")

            return read_mfa_csv(
                aligned / "audio.csv",
                words_text=transcript,
                expected_phones=expected_phones,
            )
