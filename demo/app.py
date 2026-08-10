"""CtrlSpeech second-pass control demo (Panel + Bokeh).

    panel serve demo/app.py --show --port 5006

    # over SSH, with `ssh -L 5006:localhost:5006 host` on your laptop:
    panel serve demo/app.py --port 5006 --allow-websocket-origin=localhost:5006

The flow:

1. Get a **baseline** — either synthesise one from the bundled prompt voice and
   target text, or upload your own clip and adopt it directly.
2. The app extracts the baseline's frame-level pitch and loudness curves and
   force-aligns it with MFA, so every phoneme has a time boundary.
3. Edit one control: drag the sliders, draw on the contour, or stretch a word.
4. **Regenerate** — the edited curves become the conditioning, and the result is
   plotted against both the baseline and what you asked for.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
# `panel serve` adds the script's directory itself, but importing this module
# any other way (tests, `python -m`) would not find interactive_plot.
for path in (REPO_ROOT, DEMO_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import panel as pn
import torch
from bokeh.plotting import figure

from ctrlspeech import CtrlSpeech, shift_loudness_db, shift_pitch_semitones
from ctrlspeech.align import annotate_audio, normalize_word
from ctrlspeech.pipeline import (
    FPS,
    HOP_LENGTH,
    MAX_CONTROL_FRAMES,
    MAX_PHONE_FRAMES,
    SAMPLE_RATE,
)

from interactive_plot import WORD_COLORS, interactive_dual_plot_bokeh

pn.extension(notifications=True, sizing_mode="stretch_width")

ASSET_DIR = Path(__file__).resolve().parent / "assets"
MODEL_NAME = os.environ.get("CTRLSPEECH_DEMO_MODEL", "control-600m")

DEFAULT_STEPS = 32
DEFAULT_CFG_STRENGTH = 1.5
DURATION_STEP_SECONDS = 1.0 / FPS

# An uploaded clip is used as prompt *and* target, so the sequence is twice its
# length; this ceiling is therefore stricter than MAX_CONTROL_FRAMES.
UPLOAD_MIN_SECONDS = 0.5
UPLOAD_MAX_SECONDS = 10.0

CONTROL_LABELS = {"🎵 Pitch": "pitch", "🔊 Loudness": "loudness", "⏱ Duration": "duration"}


# ─────────────────────────────────────────────────────────────────────────────
# Examples
# ─────────────────────────────────────────────────────────────────────────────
def load_examples():
    """Read demo/assets/examples.json, keeping only entries whose files exist.

    Adding an example means dropping a wav plus a plain-text transcript into
    demo/assets and listing it here; the phoneme timings are force-aligned on
    first use and cached next to the clip.
    """
    manifest = ASSET_DIR / "examples.json"
    if not manifest.exists():
        return []
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    available = []
    for entry in entries:
        paths = {
            key: ASSET_DIR / entry[key]
            for key in ("prompt_wav", "prompt_txt", "target_wav", "target_txt")
        }
        if all(path.exists() for path in paths.values()):
            available.append({**entry, **paths})
    return available


EXAMPLES = load_examples()


# ─────────────────────────────────────────────────────────────────────────────
# Model (loaded once per process, shared across browser sessions)
# ─────────────────────────────────────────────────────────────────────────────
def get_model():
    if "ctrlspeech" not in pn.state.cache:
        pn.state.cache["ctrlspeech"] = CtrlSpeech.from_pretrained(MODEL_NAME)
    return pn.state.cache["ctrlspeech"]


# ─────────────────────────────────────────────────────────────────────────────
# Duration helpers — every target lands on the model's 10 ms grid
# ─────────────────────────────────────────────────────────────────────────────
def _duration_grid_bounds(lower, upper):
    """Return the inclusive 10 ms bounds that fit inside a continuous range."""
    lower, upper = float(lower), float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("Invalid duration bounds")
    # The tolerance stops values already on the grid from being pulled inward by
    # their binary floating-point representation.
    lower_frame = int(np.ceil(lower * FPS - 1e-9))
    upper_frame = int(np.floor(upper * FPS + 1e-9))
    if lower_frame > upper_frame:
        raise ValueError("No 10 ms duration is available inside the allowed range")
    return (
        round(lower_frame * DURATION_STEP_SECONDS, 2),
        round(upper_frame * DURATION_STEP_SECONDS, 2),
    )


def _normalize_duration_target(value, lower, upper):
    """Round a duration onto the model grid, then clamp it."""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Target duration must be finite")
    lower, upper = _duration_grid_bounds(lower, upper)
    # Durations are positive, so floor(x + 0.5) reads as ordinary half-up
    # rounding instead of Python's banker's rounding at half-frame values.
    target_frame = int(np.floor(value * FPS + 0.5 + 1e-9))
    target = target_frame * DURATION_STEP_SECONDS
    return round(float(np.clip(target, lower, upper)), 2), lower, upper


def _duration_nudge_disabled(requested, delta, lower, upper):
    candidate, _, _ = _normalize_duration_target(requested + delta, lower, upper)
    return bool(np.isclose(candidate, requested, atol=1e-9))


# ─────────────────────────────────────────────────────────────────────────────
# Comparison plots
# ─────────────────────────────────────────────────────────────────────────────
def _style_figure(fig):
    fig.background_fill_color = "#FBFCFE"
    fig.border_fill_color = "#FFFFFF"
    fig.outline_line_color = None
    fig.grid.grid_line_color = "#E8ECF3"
    fig.grid.grid_line_alpha = 0.8
    fig.axis.axis_line_color = "#CBD2E0"
    fig.axis.major_tick_line_color = "#CBD2E0"
    fig.axis.minor_tick_line_color = None
    fig.title.text_color = "#20283A"
    fig.title.text_font_size = "13pt"


def comparison_figure(title, baseline, requested, achieved, y_range, y_label,
                      accent="#6C5CE7"):
    base = np.asarray(baseline, dtype=float)
    req = np.asarray(requested, dtype=float)
    ach = np.asarray(achieved, dtype=float)
    fig = figure(
        height=280, title=title, x_axis_label="Time (s)", y_axis_label=y_label,
        y_range=y_range, sizing_mode="stretch_width",
        tools="xpan,xwheel_zoom,reset,save", toolbar_location="right",
    )
    fig.line(np.arange(len(base)) / FPS, base, color="#A7B0C0", line_width=1.7,
             line_alpha=0.8, legend_label="Baseline")
    fig.line(np.arange(len(req)) / FPS, req, color=accent, line_width=2.8,
             legend_label="Requested")
    fig.line(np.arange(len(ach)) / FPS, ach, color="#172033", line_width=2.0,
             line_dash="dashed", legend_label="Achieved")
    _style_figure(fig)
    fig.legend.location = "top_left"
    fig.legend.orientation = "horizontal"
    fig.legend.label_text_font_size = "9pt"
    fig.legend.background_fill_alpha = 0.0
    fig.legend.border_line_alpha = 0.0
    return fig


def duration_comparison_figure(baseline_words, requested_words, achieved_words=None,
                               highlight_idx=None):
    """Word durations as colour-coded blocks whose widths encode seconds."""
    rows = [("Baseline", baseline_words), ("Requested", requested_words)]
    if achieved_words:
        rows.append(("Achieved", achieved_words))

    fig = figure(
        height=110 + 54 * len(rows),
        title="Word duration · block width encodes time",
        x_axis_label="Time (s)", y_range=[name for name, _ in rows],
        sizing_mode="stretch_width", tools="xpan,xwheel_zoom,reset,save",
        toolbar_location="right",
    )
    for row_name, words in rows:
        left = [float(word["start"]) for word in words]
        right = [float(word["end"]) for word in words]
        widths = [end - start for start, end in zip(left, right)]
        colors = [WORD_COLORS[i % len(WORD_COLORS)] for i in range(len(words))]
        labels = [
            str(word["label"]) if width >= 0.12 else ""
            for word, width in zip(words, widths)
        ]
        fig.hbar(y=[row_name] * len(words), left=left, right=right, height=0.58,
                 color=colors, line_color="#FFFFFF", line_width=1.2, alpha=0.88)
        fig.text(
            x=[(start + end) / 2 for start, end in zip(left, right)],
            y=[row_name] * len(words), text=labels, text_align="center",
            text_baseline="middle", text_font_size="9px", text_color="#253047",
        )
        if highlight_idx is not None and 0 <= highlight_idx < len(words):
            word = words[highlight_idx]
            fig.hbar(y=[row_name], left=[float(word["start"])],
                     right=[float(word["end"])], height=0.68, fill_alpha=0,
                     line_color="#172033", line_width=2.5)

    _style_figure(fig)
    fig.ygrid.grid_line_color = None
    fig.yaxis.axis_line_color = None
    fig.yaxis.major_tick_line_color = None
    fig.xgrid.grid_line_alpha = 0.7
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Blocking work, run off the event loop
# ─────────────────────────────────────────────────────────────────────────────
def synthesize_baseline(example, steps, cfg_strength):
    tts = get_model()
    prompt_text = " ".join(example["prompt_txt"].read_text(encoding="utf-8").split())
    target_text = " ".join(example["target_txt"].read_text(encoding="utf-8").split())
    prompt_annotation = annotate_audio(example["prompt_wav"], prompt_text, tts.aligner)
    target_annotation = annotate_audio(example["target_wav"], target_text, tts.aligner)
    return tts.generate(
        example["prompt_wav"], prompt_annotation, target_annotation,
        steps=steps, cfg_strength=cfg_strength,
    )


def adopt_upload(audio_bytes, filename, transcript, steps, cfg_strength):
    """Validate and decode an upload, then hand it to CtrlSpeech.from_audio."""
    import librosa

    transcript = " ".join((transcript or "").split())
    if not transcript:
        raise ValueError("Type the transcript of the uploaded audio first.")
    if not audio_bytes:
        raise ValueError("Choose an audio file to upload first.")

    tts = get_model()
    scratch = Path(os.environ.get("TMPDIR", "/tmp")) / "ctrlspeech-upload"
    scratch.mkdir(parents=True, exist_ok=True)
    upload_path = scratch / f"upload{Path(filename or 'audio.wav').suffix.lower() or '.wav'}"
    upload_path.write_bytes(audio_bytes)
    try:
        audio_np, _ = librosa.load(str(upload_path), sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise ValueError(f"Could not decode the uploaded audio: {exc}") from exc
    finally:
        upload_path.unlink(missing_ok=True)

    return tts.from_audio(
        audio_np, transcript, steps=steps, cfg_strength=cfg_strength,
        min_seconds=UPLOAD_MIN_SECONDS, max_seconds=UPLOAD_MAX_SECONDS,
    )


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
def build_app():
    if not EXAMPLES:
        example_note = (
            "No bundled example was found under `demo/assets`. "
            "Upload your own clip below to try the controls."
        )
    else:
        example_note = ""

    source_select = pn.widgets.RadioButtonGroup(
        name="Source",
        options={"Bundled example": "example", "Upload your own audio": "upload"},
        value="example" if EXAMPLES else "upload",
        button_type="primary", button_style="outline", width=460,
    )
    example_select = pn.widgets.Select(
        name="Example",
        options={entry["name"]: idx for idx, entry in enumerate(EXAMPLES)},
        value=0 if EXAMPLES else None, width=460,
    )
    example_copy = pn.pane.Markdown(example_note)
    example_focus = pn.widgets.RadioButtonGroup(
        name="Control to edit", options=CONTROL_LABELS, value="pitch",
        button_type="light", width=340,
    )
    gen1_btn = pn.widgets.Button(
        name="Generate baseline & align", button_type="primary", width=220,
        disabled=not EXAMPLES,
    )

    upload_input = pn.widgets.FileInput(
        accept=".wav,.mp3,.flac,.ogg,.m4a", multiple=False, width=460
    )
    transcript_input = pn.widgets.TextAreaInput(
        name="Transcript · exactly the words spoken in the clip",
        placeholder="If you dream a thing more than once, it's sure to come true.",
        height=88, width=460,
    )
    upload_focus = pn.widgets.RadioButtonGroup(
        name="Control to edit", options=CONTROL_LABELS, value="pitch",
        button_type="light", width=340,
    )
    upload_btn = pn.widgets.Button(
        name="Use uploaded audio as baseline", button_type="primary", width=260
    )

    spinner = pn.indicators.LoadingSpinner(value=False, size=28, visible=False)
    status = pn.pane.Markdown("")

    baseline_container = pn.Column()
    control_container = pn.Column()
    plot_container = pn.Column()
    regeneration_container = pn.Column()

    session = {
        "state": None,
        "editor": None,
        "focus": "pitch",
        "duration_edit": None,
        "upload_mode": not EXAMPLES,
        "example": EXAMPLES[0] if EXAMPLES else None,
    }

    # -- shared UI helpers ------------------------------------------------
    def clear_generated(msg=""):
        session["state"] = None
        session["editor"] = None
        session["duration_edit"] = None
        baseline_container.clear()
        control_container.clear()
        plot_container.clear()
        regeneration_container.clear()
        if msg:
            status.object = msg

    def describe_example(idx):
        if idx is None or not EXAMPLES:
            return
        entry = EXAMPLES[idx]
        session["example"] = entry
        target = " ".join(entry["target_txt"].read_text(encoding="utf-8").split())
        prompt = " ".join(entry["prompt_txt"].read_text(encoding="utf-8").split())
        example_copy.object = (
            f"**Target text** · {target}  \n"
            f"**Prompt voice** · {entry['prompt_wav'].name} — _{prompt}_"
        )

    def set_busy(busy, msg=""):
        spinner.value = spinner.visible = busy
        for widget in (gen1_btn, example_select, example_focus, source_select,
                       upload_btn, upload_focus):
            widget.disabled = busy or (widget is gen1_btn and not EXAMPLES)
        status.object = msg

    def audio_pane(np_audio, label):
        wav = np.clip(np_audio, -1.0, 1.0).astype(np.float32)
        return pn.Column(
            pn.pane.Markdown(f"**{label}**", margin=(0, 0, -4, 0)),
            pn.pane.Audio(wav, sample_rate=SAMPLE_RATE, name=label),
            min_width=380,
        )

    def _write_curve(src, values):
        values = np.asarray(values, dtype=float)
        data = dict(src.data)
        data["xs"] = [(np.arange(len(values)) * HOP_LENGTH / SAMPLE_RATE).tolist()]
        data["ys"] = [values.tolist()]
        src.data = data

    def curve_mean(values, voiced_only=False):
        values = np.asarray(values, dtype=float)
        if voiced_only:
            values = values[values > 1]
        return float(values.mean()) if len(values) else float("nan")

    # -- step 2: the control panel ---------------------------------------
    def build_control_panel():
        editor = session["editor"]
        state = session["state"]
        focus = session["focus"]
        # Uploads have no preset, so sliders and the stretch word fall back to
        # neutral defaults.
        defaults = {} if session["upload_mode"] else (
            session["example"].get("defaults", {}) if session["example"] else {}
        )
        note = pn.pane.Markdown("")
        reset_btn = pn.widgets.Button(name="Reset", button_type="default", width=100)

        if focus == "pitch":
            value = pn.widgets.FloatSlider(
                name="Pitch shift (semitones)", start=-12, end=12, step=1,
                value=float(defaults.get("semitones", 5)), width=360,
            )
            apply_btn = pn.widgets.Button(name="Apply", button_type="primary", width=100)

            def apply_pitch(_):
                _write_curve(
                    editor._pitch_src, shift_pitch_semitones(state.gen_f0, value.value)
                )
                note.object = f"Applied **{value.value:+g} semitones**."

            def reset_edits(_):
                editor.reset_timeline()
                session["duration_edit"] = None
                note.object = "Reset to baseline."

            apply_btn.on_click(apply_pitch)
            reset_btn.on_click(reset_edits)
            return pn.Column(
                "## 2. Pitch control",
                "Use the slider, or draw on the contour below and press Finish "
                "Pitch. Slider shifts leave unvoiced frames untouched.",
                pn.Row(value, apply_btn, reset_btn, align="end"), note,
            )

        if focus == "loudness":
            value = pn.widgets.FloatSlider(
                name="Loudness shift (dB)", start=-20, end=20, step=1,
                value=float(defaults.get("db", 8)), width=360,
            )
            apply_btn = pn.widgets.Button(name="Apply", button_type="primary", width=100)

            def apply_loudness(_):
                _write_curve(
                    editor._loudness_src, shift_loudness_db(state.gen_loud, value.value)
                )
                note.object = f"Applied **{value.value:+g} dB**."

            def reset_edits(_):
                editor.reset_timeline()
                session["duration_edit"] = None
                note.object = "Reset to baseline."

            apply_btn.on_click(apply_loudness)
            reset_btn.on_click(reset_edits)
            return pn.Column(
                "## 2. Loudness control",
                "Use the slider, or draw on the contour below and press Finish "
                "Loudness.",
                pn.Row(value, apply_btn, reset_btn, align="end"), note,
            )

        # -- duration --------------------------------------------------
        words = editor.get_words() or []
        word_select = pn.widgets.Select(name="Word", width=270)
        ratios = pn.widgets.RadioButtonGroup(
            name="Ratio", options={"0.75×": 0.75, "1×": 1.0, "1.5×": 1.5, "2×": 2.0},
            value=1.0, button_type="light", width=300,
        )
        target_duration = pn.widgets.FloatInput(
            name="Target duration (s)", value=1.0, step=0.01, width=155
        )
        nudge_buttons = {
            delta: pn.widgets.Button(name=f"{delta:+.2f} s", button_type="light", width=76)
            for delta in (-0.10, -0.05, 0.05, 0.10)
        }
        apply_btn = pn.widgets.Button(name="Apply", button_type="primary", width=100)
        summary = pn.pane.Markdown("")
        updating = [False]

        def word_options():
            return {
                f"{idx + 1:02d} · {word['label']} ({word['end'] - word['start']:.2f}s)": idx
                for idx, word in enumerate(editor.get_words() or [])
            }

        def duration_bounds(word_idx):
            word = state.word_data[word_idx]
            old_duration = word["end"] - word["start"]
            phone_durations = [
                state.gen_phoneme_data[i][2] - state.gen_phoneme_data[i][1]
                for i in range(word["phone_start"], word["phone_end"])
            ]
            min_ratio = max(0.25, max(1.0 / max(1, round(d * FPS)) for d in phone_durations))
            max_ratio = min(
                3.0,
                min(MAX_PHONE_FRAMES / max(1, round(d * FPS)) for d in phone_durations),
            )
            curve_frames = len(state.gen_f0)
            word_frames = max(1, round(old_duration * FPS))
            max_ratio = min(
                max_ratio,
                max(0.0, (MAX_CONTROL_FRAMES - curve_frames + word_frames) / word_frames),
            )
            return min_ratio, max_ratio

        def target_bounds(word_idx):
            word = state.word_data[word_idx]
            baseline = float(word["end"] - word["start"])
            lower_ratio, upper_ratio = duration_bounds(word_idx)
            return _duration_grid_bounds(baseline * lower_ratio, baseline * upper_ratio)

        def update_summary(*_):
            if updating[0] or word_select.value is None:
                return
            word = state.word_data[word_select.value]
            baseline = word["end"] - word["start"]
            lower, upper = target_bounds(word_select.value)
            requested, _, _ = _normalize_duration_target(target_duration.value, lower, upper)
            count = word["phone_end"] - word["phone_start"]
            summary.object = (
                f"**{word['label']}** · {baseline:.2f}s → {requested:.2f}s · "
                f"**{requested / baseline:.2f}×** · {count} phonemes scaled "
                f"proportionally  \nAllowed target range: {lower:.2f}s–{upper:.2f}s"
            )
            for delta, button in nudge_buttons.items():
                button.disabled = _duration_nudge_disabled(requested, delta, lower, upper)

        def set_target(value, sync_ratio=True):
            """Set an absolute target and keep the quick ratios consistent."""
            if word_select.value is None:
                return
            idx = word_select.value
            word = state.word_data[idx]
            baseline = float(word["end"] - word["start"])
            lower, upper = target_bounds(idx)
            value, lower, upper = _normalize_duration_target(value, lower, upper)

            updating[0] = True
            try:
                target_duration.value = value
                if sync_ratio:
                    ratios.value = next(
                        (
                            candidate
                            for candidate in ratios.options.values()
                            if np.isclose(
                                value,
                                _normalize_duration_target(
                                    baseline * candidate, lower, upper
                                )[0],
                                atol=1e-9,
                            )
                        ),
                        None,
                    )
            finally:
                updating[0] = False
            update_summary()

        def select_word(word_idx, suggested_ratio=1.0, suggested_duration=None):
            if word_idx is None:
                return
            updating[0] = True
            try:
                word = state.word_data[word_idx]
                baseline = float(word["end"] - word["start"])
                lower_ratio, upper_ratio = duration_bounds(word_idx)
                lower, upper = target_bounds(word_idx)
                allowed = {
                    label: ratio
                    for label, ratio in {"0.75×": 0.75, "1×": 1.0, "1.5×": 1.5, "2×": 2.0}.items()
                    if lower_ratio - 1e-9 <= ratio <= upper_ratio + 1e-9
                }
                allowed.setdefault("1×", 1.0)
                ratios.options = allowed
                requested = (
                    float(suggested_duration)
                    if suggested_duration is not None
                    else baseline * float(suggested_ratio)
                )
                requested, lower, upper = _normalize_duration_target(requested, lower, upper)
                ratios.value = next(
                    (
                        candidate
                        for candidate in allowed.values()
                        if np.isclose(
                            requested,
                            _normalize_duration_target(baseline * candidate, lower, upper)[0],
                            atol=1e-9,
                        )
                    ),
                    None,
                )
                target_duration.start = lower
                target_duration.end = upper
                target_duration.value = requested
                if hasattr(editor, "select_word"):
                    editor.select_word(word_idx)
            finally:
                updating[0] = False
            update_summary()

        def on_ratio(event):
            if updating[0] or word_select.value is None or event.new is None:
                return
            word = state.word_data[word_select.value]
            set_target(float(word["end"] - word["start"]) * float(event.new),
                       sync_ratio=False)

        def on_target(event):
            if updating[0] or word_select.value is None:
                return
            if event.new is None or not np.isfinite(event.new):
                word = state.word_data[word_select.value]
                set_target(float(word["end"] - word["start"]))
                return
            set_target(event.new)

        def apply_duration(_):
            idx = word_select.value
            before_word = dict(state.word_data[idx])
            before = before_word["end"] - before_word["start"]
            lower, upper = target_bounds(idx)
            requested_target, _, _ = _normalize_duration_target(
                target_duration.value, lower, upper
            )
            try:
                # One word edit at a time: starting from the immutable baseline
                # makes Apply idempotent and replaces any earlier edit.
                editor.reset_timeline()
                result = editor.apply_word_duration(idx, requested_target)
                requested = float(result["new_duration"])
                session["duration_edit"] = {
                    "word_idx": idx, "label": before_word["label"],
                    "before": before, "requested": requested,
                    "ratio": requested / before,
                }
                word_select.options = word_options()
                word_select.value = idx
                note.object = (
                    f"Applied **{before_word['label']} {before:.2f}s → "
                    f"{requested:.2f}s**. This replaces the previous duration "
                    "edit; later words move together and inter-word pauses are "
                    "preserved."
                )
                select_word(idx, suggested_duration=requested)
            except Exception as exc:
                note.object = f"Error: {exc}"
                pn.state.notifications.error(str(exc)[:200])

        def reset_duration(_):
            selected_idx = word_select.value
            editor.reset_timeline()
            session["duration_edit"] = None
            word_select.options = word_options()
            idx = selected_idx
            if idx is None:
                wanted = normalize_word(str(defaults.get("word", "")))
                idx = next(
                    (i for i, word in enumerate(editor.get_words())
                     if normalize_word(str(word["label"])) == wanted),
                    0,
                )
            word_select.value = idx
            select_word(idx, 1.0)
            note.object = "Reset to baseline."

        reset_btn.on_click(reset_duration)
        apply_btn.on_click(apply_duration)
        word_select.param.watch(lambda event: select_word(event.new), "value")
        ratios.param.watch(on_ratio, "value")
        target_duration.param.watch(on_target, "value")
        for delta, button in nudge_buttons.items():
            button.on_click(
                lambda _, amount=delta: set_target(float(target_duration.value) + amount)
            )

        word_select.options = word_options()
        if word_select.options:
            wanted = normalize_word(str(defaults.get("word", "")))
            selected = next(
                (i for i, word in enumerate(words)
                 if normalize_word(str(word["label"])) == wanted),
                0,
            )
            word_select.value = selected
            select_word(selected, float(defaults.get("ratio", 1.0)))

        return pn.Column(
            "## 2. Word duration control",
            "Select a word, then use a quick ratio or set its exact duration.",
            pn.Row(word_select, ratios, align="end"),
            pn.Row(target_duration, *nudge_buttons.values(), apply_btn, reset_btn,
                   align="end"),
            summary, note,
        )

    # -- installing a baseline (shared by both sources) -------------------
    def install_baseline(state, audio_panes):
        session["state"] = state
        editor = interactive_dual_plot_bokeh(
            state.gen_f0, state.gen_loud,
            phoneme_data=state.gen_phoneme_data,
            word_data=state.word_data,
            focus=session["focus"],
            waveform_data=state.gen_np,
            waveform_sample_rate=SAMPLE_RATE,
            hop_length=HOP_LENGTH, sample_rate=SAMPLE_RATE,
            pitch_title="Pitch contour", loudness_title="Loudness contour",
        )
        session["editor"] = editor

        baseline_container[:] = [
            pn.pane.Markdown("### Baseline"),
            pn.Row(*audio_panes),
            pn.pane.Markdown(f"**Target text** · {state.target_words}"),
        ]
        control_container[:] = [pn.layout.Divider(), build_control_panel()]
        plot_container[:] = [editor]
        regeneration_container[:] = [pn.layout.Divider(), build_regen_section()]

    async def on_gen1(_):
        set_busy(True, "Generating the baseline and aligning words / phonemes with MFA…")
        clear_generated()
        session["upload_mode"] = False
        session["focus"] = example_focus.value
        try:
            state = await asyncio.get_event_loop().run_in_executor(
                None, synthesize_baseline, session["example"],
                DEFAULT_STEPS, DEFAULT_CFG_STRENGTH,
            )
            install_baseline(state, [
                audio_pane(state.prompt_np, "Prompt · Voice reference"),
                audio_pane(state.gen_np, "Baseline · First generation"),
            ])
            set_busy(False, "Baseline generation and MFA alignment complete.")
        except Exception as exc:
            set_busy(False, f"Error: {exc}")
            pn.state.notifications.error(str(exc)[:200])

    async def on_upload(_):
        set_busy(True, "Analysing the uploaded audio and aligning it with MFA…")
        clear_generated()
        session["upload_mode"] = True
        session["focus"] = upload_focus.value
        try:
            state = await asyncio.get_event_loop().run_in_executor(
                None, adopt_upload, upload_input.value, upload_input.filename,
                transcript_input.value, DEFAULT_STEPS, DEFAULT_CFG_STRENGTH,
            )
            install_baseline(state, [
                audio_pane(state.gen_np, "Uploaded audio · Voice + baseline prosody"),
            ])
            set_busy(
                False,
                f"Uploaded audio ready · {state.prompt_duration:.2f}s · "
                f"{len(state.word_data)} words aligned.",
            )
        except Exception as exc:
            set_busy(False, f"Error: {exc}")
            pn.state.notifications.error(str(exc)[:200])

    # -- step 3: regenerate -----------------------------------------------
    def build_regen_section():
        gen2_btn = pn.widgets.Button(
            name="Regenerate with current control", button_type="success", width=250
        )
        gen2_spinner = pn.indicators.LoadingSpinner(value=False, size=24, visible=False)
        gen2_status = pn.pane.Markdown("")
        gen2_out = pn.Column()

        async def on_gen2(_):
            for widget in (gen2_btn, gen1_btn, example_select, source_select, upload_btn):
                widget.disabled = True
            gen2_spinner.value = gen2_spinner.visible = True
            gen2_status.object = "Generating controlled audio…"
            gen2_out.clear()
            try:
                editor = session["editor"]
                state = session["state"]
                focus = session["focus"]
                edited_f0 = editor.get_pitch()
                edited_loud = editor.get_loudness()
                edited_phonemes = editor.get_phonemes()
                tts = get_model()
                res = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: tts.regenerate(
                        state, pitch=edited_f0, loudness=edited_loud,
                        phonemes=edited_phonemes,
                    ),
                )

                achieved_words = None
                duration_message = None
                duration_edit = session["duration_edit"]
                if focus == "duration" and duration_edit:
                    gen2_status.object = (
                        "Audio generated. Verifying the target word's actual "
                        "duration with MFA…"
                    )
                    try:
                        achieved = await asyncio.get_event_loop().run_in_executor(
                            None, tts.align, torch.from_numpy(res.audio).float(),
                            state.target_words, state.target_phones,
                        )
                        achieved_words = achieved["word_data"]
                        got = achieved_words[duration_edit["word_idx"]]
                        duration_message = (
                            f"**{duration_edit['label']}** · baseline "
                            f"{duration_edit['before']:.2f}s → requested "
                            f"{duration_edit['requested']:.2f}s → achieved "
                            f"{got['end'] - got['start']:.2f}s"
                        )
                    except Exception as align_exc:
                        duration_message = (
                            f"**{duration_edit['label']}** · requested "
                            f"{duration_edit['requested']:.2f}s · MFA verification failed"
                        )
                        pn.state.notifications.warning(str(align_exc)[:200])

                output = [
                    pn.Row(
                        audio_pane(state.gen_np, "Baseline"),
                        audio_pane(res.audio, "Controlled · Second generation"),
                    )
                ]
                if focus == "pitch":
                    output += [
                        pn.pane.Markdown(
                            "**Voiced pitch mean** · "
                            f"{curve_mean(state.gen_f0, True):.1f} → "
                            f"{curve_mean(res.pitch, True):.1f} bins"
                        ),
                        pn.pane.Bokeh(comparison_figure(
                            "Pitch · baseline vs requested vs achieved",
                            state.gen_f0, edited_f0, res.pitch,
                            (0, 128), "Pitch (bin)", "#6C5CE7",
                        )),
                    ]
                elif focus == "loudness":
                    output += [
                        pn.pane.Markdown(
                            "**Loudness mean** · "
                            f"{curve_mean(state.gen_loud):.1f} → "
                            f"{curve_mean(res.loudness):.1f} bins"
                        ),
                        pn.pane.Bokeh(comparison_figure(
                            "Loudness · baseline vs requested vs achieved",
                            state.gen_loud, edited_loud, res.loudness,
                            (0, 64), "Loudness (bin)", "#00A6A6",
                        )),
                    ]
                else:
                    if duration_message:
                        output.append(pn.pane.Markdown(duration_message))
                    output.append(pn.pane.Bokeh(duration_comparison_figure(
                        state.word_data, editor.get_words(), achieved_words,
                        duration_edit["word_idx"] if duration_edit else None,
                    )))

                gen2_out[:] = output
                gen2_status.object = "Controlled generation and comparison complete."
            except Exception as exc:
                gen2_status.object = f"Error: {exc}"
                pn.state.notifications.error(str(exc)[:200])
            finally:
                for widget in (gen2_btn, gen1_btn, example_select, source_select,
                               upload_btn):
                    widget.disabled = False
                gen1_btn.disabled = not EXAMPLES
                gen2_spinner.value = gen2_spinner.visible = False

        gen2_btn.on_click(on_gen2)
        return pn.Column(
            "## 3. Regenerate and compare",
            pn.Row(gen2_btn, gen2_spinner), gen2_status, gen2_out,
        )

    # -- wiring ------------------------------------------------------------
    example_panel = pn.Column(
        example_select, example_copy,
        pn.pane.Markdown("**Control to edit**", margin=(4, 0, -8, 0)),
        example_focus, pn.Row(gen1_btn),
        visible=bool(EXAMPLES),
    )
    upload_panel = pn.Column(
        pn.pane.Markdown(
            "Upload a clip and type its transcript. That clip becomes the voice "
            "**and** the baseline prosody — pitch, loudness and word durations "
            "are read straight from it, so no first generation is needed. "
            "Whatever you do not edit is reused from the upload as-is. Clips "
            f"must be {UPLOAD_MIN_SECONDS:g}–{UPLOAD_MAX_SECONDS:g}s."
        ),
        upload_input, transcript_input,
        pn.pane.Markdown("**Control to edit**", margin=(4, 0, -8, 0)),
        upload_focus, pn.Row(upload_btn),
        visible=not EXAMPLES,
    )

    def on_source_change(event):
        is_upload = event.new == "upload"
        example_panel.visible = not is_upload
        upload_panel.visible = is_upload
        clear_generated(
            "Upload a clip and type its transcript." if is_upload
            else "Example loaded. Click **Generate baseline & align**."
        )

    source_select.param.watch(on_source_change, "value")
    example_select.param.watch(
        lambda event: (describe_example(event.new),
                       clear_generated("Example loaded. Click "
                                       "**Generate baseline & align**.")),
        "value",
    )
    gen1_btn.on_click(on_gen1)
    upload_btn.on_click(on_upload)
    describe_example(0 if EXAMPLES else None)

    return pn.Column(
        pn.pane.Markdown(
            "# CtrlSpeech Control Demo\n"
            "Generate or upload a baseline, edit one prosody control, and hear "
            f"what the model actually does with it. Model: `{MODEL_NAME}`."
        ),
        pn.Column(
            "## 1. Pick a baseline",
            source_select, example_panel, upload_panel,
            pn.Row(spinner, align="center"), status,
        ),
        baseline_container, control_container, plot_container, regeneration_container,
        max_width=1050, margin=20,
    )


build_app().servable(title="CtrlSpeech Control Demo")
