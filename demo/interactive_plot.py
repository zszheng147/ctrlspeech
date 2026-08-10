"""
Bokeh + Panel interactive pitch / loudness / phoneme editor.

Zero-lag drawing because:
  - FreehandDrawTool  → runs in the browser, no Python round-trips during stroke
  - PolyEditTool      → vertex dragging is fully browser-side
  - PointDrawTool     → phoneme boundary handles are browser-side
  - Python callbacks  → only fire on discrete button clicks (not every pixel)

Usage (Jupyter):
    pn.extension()
    layout = interactive_dual_plot_bokeh(pitch, loudness, phonemes)
    layout          # renders inline in Jupyter

Retrieve results after editing:
    pitch_arr    = layout.get_pitch()
    loudness_arr = layout.get_loudness()
    phoneme_list = layout.get_phonemes()   # None if no phonemes
"""

import numpy as np
import panel as pn
from bokeh.plotting import figure
from bokeh.models import (
    BoxAnnotation, ColumnDataSource, Span,
    FreehandDrawTool, PointDrawTool,
    Range1d, CrosshairTool, CustomJS, HoverTool,
)
from bokeh.events import MouseMove, MouseLeave
from scipy.interpolate import interp1d as scipy_interp1d

from ctrlspeech.retime import (
    MAX_DURATION_FRAMES,
    MAX_TIMELINE_FRAMES,
    _resample_segment,
    _retime_waveform_segment,
    retime_word_curves,
    time_to_frame_index,
)


PHONEME_COLORS = [
    '#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4', '#FFEAA7',
    '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
]
WORD_COLORS = [
    '#C4B5FD', '#BFDBFE', '#A7F3D0', '#FDE68A', '#FBCFE8',
    '#BAE6FD', '#FED7AA', '#DDD6FE', '#BBF7D0', '#FECACA',
    '#A5F3FC', '#C7D2FE', '#F5D0FE', '#D9F99D', '#FDE2E4',
    '#BFE3D0', '#FFE0B5', '#CDE7F0', '#E2D3F5', '#F8D7A7',
]
PITCH_COLOR = '#6C5CE7'
LOUDNESS_COLOR = '#16A394'
PLOT_TEXT_COLOR = '#253047'
PLOT_MUTED_COLOR = '#667085'
PLOT_GRID_COLOR = '#E9EDF5'
MIN_PHONEME_DURATION = 0.02
EDIT_POINT_TARGET_COUNT = 96
EDIT_POINT_NEIGHBOR_RADIUS = 6


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _waveform_envelope(waveform, sample_rate, max_columns=2400):
    """Return a compact min/max envelope suitable for a Bokeh waveform track."""
    waveform = np.asarray(waveform, dtype=float)
    if len(waveform) == 0:
        return {"x": [], "low": [], "high": []}

    bucket_size = max(1, int(np.ceil(len(waveform) / max_columns)))
    starts = np.arange(0, len(waveform), bucket_size, dtype=int)
    ends = np.minimum(starts + bucket_size, len(waveform))
    low = np.minimum.reduceat(waveform, starts)
    high = np.maximum.reduceat(waveform, starts)
    x = ((starts + ends - 1) * 0.5) / float(sample_rate)
    return {
        "x": x.tolist(),
        "low": low.tolist(),
        "high": high.tolist(),
    }


def apply_phoneme_edit_to_data(pitch_data, loudness_data, original_phonemes,
                               new_phonemes, hop_length=256, sample_rate=24000,
                               changed_idx=None):
    """
    Apply phoneme duration changes to pitch and loudness data.

    When a phoneme's duration changes, the corresponding pitch/loudness segment
    is resampled (stretched or compressed) to match the new duration.

    Returns
    -------
    new_pitch_data, new_loudness_data, new_time_axis
    """
    if len(original_phonemes) == 0 or len(new_phonemes) == 0:
        time_axis = np.arange(len(pitch_data), dtype=float) * hop_length / sample_rate
        return pitch_data.copy(), loudness_data.copy(), time_axis

    new_pitch_segments = []
    new_loudness_segments = []
    frame_sec = hop_length / sample_rate

    # Preserve non-phoneme region before the first phoneme.
    first_start = time_to_frame_index(original_phonemes[0][1], hop_length, sample_rate)
    first_start = max(0, min(first_start, len(pitch_data)))
    if first_start > 0:
        new_pitch_segments.append(pitch_data[:first_start].copy())
        new_loudness_segments.append(loudness_data[:first_start].copy())

    for i, (orig, new) in enumerate(zip(original_phonemes, new_phonemes)):
        orig_start_frame = time_to_frame_index(orig[1], hop_length, sample_rate)
        orig_end_frame   = time_to_frame_index(orig[2], hop_length, sample_rate)
        new_n = (time_to_frame_index(new[2], hop_length, sample_rate)
                 - time_to_frame_index(new[1], hop_length, sample_rate))

        orig_start_frame = max(0, min(orig_start_frame, len(pitch_data)))
        orig_end_frame   = max(0, min(orig_end_frame,   len(pitch_data)))

        orig_n = orig_end_frame - orig_start_frame

        # When we know which boundary moved, only that phoneme should be
        # stretched/compressed. All other phonemes must keep their original
        # frame sequence untouched.
        if changed_idx is not None and i != changed_idx:
            new_n = orig_n
        else:
            # Keep numerically unchanged durations frame-stable:
            # dragging one boundary shifts later phonemes in time, and floor-based
            # sec->frame conversion can otherwise introduce +/-1 frame jitter.
            orig_dur = orig[2] - orig[1]
            new_dur = new[2] - new[1]
            if abs(new_dur - orig_dur) <= 0.5 * frame_sec:
                new_n = orig_n

        if new_n == 0 and (new[2] - new[1]) > 0:
            new_n = 1

        p_seg = pitch_data[orig_start_frame:orig_end_frame].copy()
        l_seg = loudness_data[orig_start_frame:orig_end_frame].copy()

        if orig_n != new_n and len(p_seg) > 0:
            p_seg = _resample_segment(p_seg, orig_n, new_n)
            l_seg = _resample_segment(l_seg, orig_n, new_n)
        elif len(p_seg) == 0 and new_n > 0:
            p_seg = np.zeros(new_n)
            l_seg = np.zeros(new_n)

        new_pitch_segments.append(p_seg)
        new_loudness_segments.append(l_seg)

    # Preserve non-phoneme region after the last phoneme.
    last_end = time_to_frame_index(original_phonemes[-1][2], hop_length, sample_rate)
    last_end = max(0, min(last_end, len(pitch_data)))
    if last_end < len(pitch_data):
        new_pitch_segments.append(pitch_data[last_end:].copy())
        new_loudness_segments.append(loudness_data[last_end:].copy())

    new_pitch = np.concatenate(new_pitch_segments) if new_pitch_segments else np.array([])
    new_loud = np.concatenate(new_loudness_segments) if new_loudness_segments else np.array([])

    total_frames = len(new_pitch)
    new_time = np.arange(total_frames, dtype=float) * hop_length / sample_rate

    return new_pitch, new_loud, new_time


def _resample_to_time_axis(drawn_xs, drawn_ys, time_axis, y_min, y_max):
    """
    Resample a freehand-drawn curve (arbitrary x spacing) onto a fixed
    time_axis.  Called after FreehandDrawTool completes a stroke.
    """
    drawn_xs = np.asarray(drawn_xs, dtype=float)
    drawn_ys = np.asarray(drawn_ys, dtype=float)
    if len(drawn_xs) < 2:
        return None

    # Sort and deduplicate by x
    order = np.argsort(drawn_xs)
    drawn_xs, drawn_ys = drawn_xs[order], drawn_ys[order]
    _, ui = np.unique(drawn_xs, return_index=True)
    drawn_xs, drawn_ys = drawn_xs[ui], drawn_ys[ui]
    if len(drawn_xs) < 2:
        return None

    f = scipy_interp1d(drawn_xs, drawn_ys, bounds_error=False,
                       fill_value=(drawn_ys[0], drawn_ys[-1]), kind='linear')
    return np.clip(f(time_axis), y_min, y_max)


def _latest_valid_multiline(data):
    """Return the newest usable ``MultiLine`` row from a CDS data mapping.

    ``FreehandDrawTool`` appends every stroke as a new row in the renderer's
    ``xs``/``ys`` columns.  The source normally starts with the current edited
    contour in row zero, so the newly drawn contour is the *last* row, not the
    first one.  Scan backwards as a small guard against an incomplete final
    row (for example, a click without a drag).
    """
    xs_rows = data.get('xs', [])
    ys_rows = data.get('ys', [])
    for xs, ys in reversed(list(zip(xs_rows, ys_rows))):
        try:
            x = np.asarray(xs, dtype=float)
            y = np.asarray(ys, dtype=float)
        except (TypeError, ValueError):
            continue
        if (
            x.ndim == 1
            and y.ndim == 1
            and len(x) == len(y)
            and len(x) >= 2
            and len(np.unique(x)) >= 2
            and np.all(np.isfinite(x))
            and np.all(np.isfinite(y))
        ):
            return x, y
    return None


def _sample_sparse_control_points(curve_x, curve_y, target_points=EDIT_POINT_TARGET_COUNT):
    """
    Downsample a dense curve into sparse, draggable control points.
    Keeps first/last point so the reconstructed curve remains anchored.
    """
    x = np.asarray(curve_x, dtype=float)
    y = np.asarray(curve_y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return x, y

    order = np.argsort(x)
    x, y = x[order], y[order]
    _, ui = np.unique(x, return_index=True)
    x, y = x[ui], y[ui]
    if len(x) <= 2:
        return x, y

    step = max(1, int(np.ceil(len(x) / max(2, target_points))))
    idx = np.arange(0, len(x), step, dtype=int)
    if idx[-1] != len(x) - 1:
        idx = np.append(idx, len(x) - 1)
    return x[idx], y[idx]


def _apply_local_delta(old_y, moved_idx, delta, radius=EDIT_POINT_NEIGHBOR_RADIUS):
    """
    Apply one control-point edit to nearby points with Gaussian falloff.
    This creates a smoother, Bezier-like local deformation.
    """
    old_y = np.asarray(old_y, dtype=float)
    if len(old_y) == 0:
        return old_y
    if moved_idx < 0 or moved_idx >= len(old_y):
        return old_y.copy()

    if abs(delta) < 1e-8:
        return old_y.copy()

    dist = np.abs(np.arange(len(old_y)) - moved_idx)
    sigma = max(1.0, radius / 2.0)
    weights = np.exp(-0.5 * (dist / sigma) ** 2)
    weights[dist > radius] = 0.0
    return old_y + delta * weights


def _reconstruct_curve_from_controls(ctrl_x, ctrl_y, target_x, y_min, y_max):
    """
    Reconstruct a dense curve from sparse control points.
    Uses cubic interpolation when possible for smoother shape.
    """
    ctrl_x = np.asarray(ctrl_x, dtype=float)
    ctrl_y = np.asarray(ctrl_y, dtype=float)
    target_x = np.asarray(target_x, dtype=float)
    if len(ctrl_x) < 2 or len(ctrl_y) < 2 or len(target_x) == 0:
        return None

    order = np.argsort(ctrl_x)
    ctrl_x, ctrl_y = ctrl_x[order], ctrl_y[order]
    _, ui = np.unique(ctrl_x, return_index=True)
    ctrl_x, ctrl_y = ctrl_x[ui], ctrl_y[ui]
    if len(ctrl_x) < 2:
        return None

    kind = 'cubic' if len(ctrl_x) >= 4 else 'linear'
    f = scipy_interp1d(
        ctrl_x, ctrl_y,
        bounds_error=False,
        fill_value=(ctrl_y[0], ctrl_y[-1]),
        kind=kind,
    )
    return np.clip(f(target_x), y_min, y_max)


def _restore_pan(fig):
    """Switch the active drag tool back to Pan after editing mode is turned off."""
    from bokeh.models import PanTool
    for t in fig.tools:
        if isinstance(t, PanTool):
            fig.toolbar.active_drag = t
            return


def _style_plot(fig, *, show_y_grid=True):
    """Apply the shared light visual treatment used by all control plots."""
    fig.background_fill_color = '#FFFFFF'
    fig.border_fill_color = '#FFFFFF'
    fig.outline_line_color = None
    fig.toolbar.autohide = True
    fig.toolbar.logo = None

    fig.title.text_color = PLOT_TEXT_COLOR
    fig.title.text_font_size = '14px'
    fig.title.text_font_style = 'bold'
    fig.title.align = 'left'

    fig.axis.axis_line_color = '#D6DCE8'
    fig.axis.major_tick_line_color = '#D6DCE8'
    fig.axis.minor_tick_line_color = None
    fig.axis.major_label_text_color = PLOT_MUTED_COLOR
    fig.axis.major_label_text_font_size = '9pt'
    fig.axis.axis_label_text_color = PLOT_MUTED_COLOR
    fig.axis.axis_label_text_font_size = '10pt'
    fig.axis.axis_label_text_font_style = 'normal'

    fig.xgrid.grid_line_color = PLOT_GRID_COLOR
    fig.xgrid.grid_line_alpha = 0.8
    if show_y_grid:
        fig.ygrid.grid_line_color = PLOT_GRID_COLOR
        fig.ygrid.grid_line_alpha = 0.8
    else:
        fig.ygrid.visible = False


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def interactive_dual_plot_bokeh(
    pitch_data,
    loudness_data,
    phoneme_data=None,
    hop_length: int = 256,
    sample_rate: int = 24000,
    pitch_title: str = "Pitch",
    loudness_title: str = "Loudness",
    plot_width: int = 900,
    word_data=None,
    focus=None,
    waveform_data=None,
    waveform_sample_rate=None,
):
    """
    Parameters
    ----------
    pitch_data     : array-like, shape (N,), values 0–128
    loudness_data  : array-like, shape (N,), values 0–64
    phoneme_data   : list of [phoneme_str, start_sec, end_sec], optional
    word_data      : list of dictionaries with label/start/end/phone_start/
                     phone_end (exclusive), optional. When supplied, a
                     read-only word timeline replaces phoneme drag controls.
    hop_length     : int, default 256
    sample_rate    : int, default 24_000
    pitch_title    : str
    loudness_title : str
    plot_width     : int, default 900
    focus          : {None, "pitch", "loudness", "duration"}, optional
                     Limit the visible layout to one control visualization.
                     ``None`` preserves the full editor layout.
    waveform_data  : array-like, shape (samples,), optional
                     Audio waveform shown above the word-duration track.
    waveform_sample_rate : int, optional
                     Sample rate of ``waveform_data``. Defaults to
                     ``sample_rate`` when omitted.

    Returns
    -------
    layout : pn.Column
        Call layout.get_pitch() / .get_loudness() / .get_phonemes() /
        .get_words() to retrieve current data. Word-aware layouts additionally
        expose .apply_word_duration(index, seconds) and .select_word(index).
    """
    pitch_data    = np.asarray(pitch_data,    dtype=float)
    loudness_data = np.asarray(loudness_data, dtype=float)

    has_waveform = waveform_data is not None
    if has_waveform:
        waveform_data = np.asarray(waveform_data, dtype=float)
        if waveform_data.ndim != 1:
            raise ValueError("waveform_data must be one-dimensional")
        if len(waveform_data) == 0:
            raise ValueError("waveform_data must not be empty")
        if waveform_sample_rate is None:
            waveform_sample_rate = sample_rate
        try:
            waveform_sample_rate = int(waveform_sample_rate)
        except (TypeError, ValueError) as exc:
            raise TypeError("waveform_sample_rate must be an integer") from exc
        if waveform_sample_rate <= 0:
            raise ValueError("waveform_sample_rate must be positive")
        waveform_data = np.nan_to_num(waveform_data, copy=True)
    elif waveform_sample_rate is not None:
        raise ValueError("waveform_sample_rate requires waveform_data")

    valid_focuses = {None, "pitch", "loudness", "duration"}
    if focus not in valid_focuses:
        raise ValueError(
            "focus must be one of None, 'pitch', 'loudness', or 'duration'"
        )

    n_frames       = len(pitch_data)
    total_duration = n_frames * hop_length / sample_rate
    time_axis      = np.arange(n_frames, dtype=float) * hop_length / sample_rate

    if pitch_data.ndim != 1 or loudness_data.ndim != 1:
        raise ValueError("pitch_data and loudness_data must be one-dimensional")
    if len(pitch_data) != len(loudness_data):
        raise ValueError("pitch_data and loudness_data must have equal length")

    has_phonemes = phoneme_data is not None and len(phoneme_data) > 0
    if has_phonemes:
        phoneme_data = [list(p) for p in phoneme_data]   # mutable deep copy
    else:
        phoneme_data = []

    has_words = word_data is not None and len(word_data) > 0
    if has_words:
        if not has_phonemes:
            raise ValueError("word_data requires phoneme_data")
        word_data = [dict(word) for word in word_data]
        required_word_fields = {"label", "start", "end", "phone_start", "phone_end"}
        for idx, word in enumerate(word_data):
            missing = required_word_fields.difference(word)
            if missing:
                raise ValueError(f"word_data[{idx}] is missing: {sorted(missing)}")
    else:
        word_data = []

    if focus == "duration" and not has_words:
        raise ValueError("focus='duration' requires word_data")

    has_phoneme_editor = has_phonemes and not has_words

    initial_pitch = pitch_data.copy()
    initial_loudness = loudness_data.copy()
    initial_phonemes = [list(p) for p in phoneme_data]
    initial_words = [dict(word) for word in word_data]
    initial_waveform = waveform_data.copy() if has_waveform else None
    current_waveform = waveform_data.copy() if has_waveform else None

    W = plot_width
    # Shared x_range: pan/zoom is automatically synced across all three plots
    waveform_duration = (
        len(current_waveform) / waveform_sample_rate if has_waveform else 0.0
    )
    shared_x = Range1d(0, max(total_duration, waveform_duration))

    # ══════════════════════════════════════════════════════════════════════════
    # PHONEME / WORD TRACK
    # ══════════════════════════════════════════════════════════════════════════
    phoneme_panel = None
    phoneme_rect_src = None
    phoneme_bdry_src = None
    word_panel = None
    word_rect_src = None
    waveform_panel = None
    waveform_src = None
    waveform_selection = None
    p_ph = None
    p_word = None
    p_waveform = None
    _spans = []
    _selected_word_idx = [None]
    # Keep a snapshot of boundaries BEFORE each edit session so we can compute
    # the delta when "Finish Phonemes" is clicked.
    _orig_phonemes = [list(p) for p in phoneme_data] if has_phonemes else []
    _orig_pitch = pitch_data.copy()
    _orig_loudness = loudness_data.copy()

    if has_phoneme_editor:
        p_ph = figure(
            width=W, height=120,
            title="Phonemes  ─  Click 'Edit Phonemes' then drag ▼ handles",
            x_range=shared_x, y_range=(-0.05, 1.05),
            tools="reset,save",
            toolbar_location="right",
        )
        p_ph.yaxis.visible = False
        p_ph.ygrid.visible = False
        p_ph.xgrid.visible = False
        p_ph.outline_line_color = None
        p_ph.background_fill_color = "#fafafa"
        p_ph.min_border_left = 80

        # ── Phoneme rectangles (dynamic ColumnDataSource) ─────────────────
        n_ph = len(phoneme_data)
        phoneme_rect_src = ColumnDataSource(dict(
            cx   = [(p[1]+p[2])/2 for p in phoneme_data],
            cw   = [p[2]-p[1]     for p in phoneme_data],
            cy   = [0.625]        * n_ph,
            ch   = [0.65]         * n_ph,
            color= [PHONEME_COLORS[i % len(PHONEME_COLORS)] for i in range(n_ph)],
            txt  = [p[0]          for p in phoneme_data],
        ))
        p_ph.rect(
            x='cx', y='cy', width='cw', height='ch',
            fill_color='color', line_color='white', line_width=0.5, alpha=0.85,
            source=phoneme_rect_src,
        )
        p_ph.text(
            x='cx', y=0.62, text='txt',
            text_align='center', text_baseline='middle',
            text_font_size='10px', text_color='#222222',
            source=phoneme_rect_src,
        )

        # ── Boundary drag handles (N red triangles) ─────────────────────
        # One handle per phoneme end; user drags horizontally.
        bx = [phoneme_data[i][2] for i in range(n_ph)]
        phoneme_bdry_src = ColumnDataSource({'x': bx, 'y': [0.08] * len(bx)})
        b_glyphs = p_ph.scatter(
            x='x', y='y', size=11, marker='inverted_triangle',
            color='#FF4444', alpha=0.9,
            source=phoneme_bdry_src,
        )

        # PointDrawTool: drag existing handles, no adding new ones
        pdt = PointDrawTool(renderers=[b_glyphs], add=False)
        p_ph.add_tools(pdt)
        p_ph.toolbar.active_tap = pdt

        # Red vertical spans (purely visual, stay in sync via Python callback)
        _spans = [
            Span(location=x, dimension='height',
                 line_color='#FF4444', line_width=1.5, line_alpha=0.5)
            for x in bx
        ]
        for s in _spans:
            p_ph.add_layout(s)

        _updating_phonemes = [False]

        def _sync_phoneme_rects(attr, old, new):
            """
            Triggered when boundary handles are moved (after mouse release).
            Only the phoneme before the dragged boundary changes duration;
            all subsequent phonemes shift by the same delta, preserving
            their original durations.  Total duration changes accordingly.
            """
            if _updating_phonemes[0]:
                return
            _updating_phonemes[0] = True
            try:
                raw_bx = list(phoneme_bdry_src.data['x'])
                sorted_bx = sorted(raw_bx)
                n = len(phoneme_data)

                cur_bx = [phoneme_data[i][2] for i in range(n)]

                moved_idx = -1
                delta = 0.0
                for i in range(len(cur_bx)):
                    d = sorted_bx[i] - cur_bx[i]
                    if abs(d) > abs(delta):
                        moved_idx = i
                        delta = d

                if moved_idx < 0 or abs(delta) < 1e-6:
                    return

                min_end = phoneme_data[moved_idx][1] + MIN_PHONEME_DURATION
                new_boundary = max(sorted_bx[moved_idx], min_end)
                delta = new_boundary - cur_bx[moved_idx]

                phoneme_data[moved_idx][2] = new_boundary
                for j in range(moved_idx + 1, n):
                    phoneme_data[j][1] += delta
                    phoneme_data[j][2] += delta

                new_bx = [phoneme_data[i][2] for i in range(n)]
                phoneme_bdry_src.data = {'x': new_bx, 'y': [0.08] * len(new_bx)}

                phoneme_rect_src.data.update(dict(
                    cx = [(p[1]+p[2])/2 for p in phoneme_data],
                    cw = [p[2]-p[1]     for p in phoneme_data],
                ))
                for span, x in zip(_spans, new_bx):
                    span.location = x

                new_p, new_l, new_t = apply_phoneme_edit_to_data(
                    _orig_pitch, _orig_loudness,
                    _orig_phonemes, phoneme_data,
                    hop_length, sample_rate,
                    changed_idx=moved_idx,
                )
                pitch_src.data    = {'xs': [new_t.tolist()], 'ys': [new_p.tolist()]}
                loudness_src.data = {'xs': [new_t.tolist()], 'ys': [new_l.tolist()]}
                shared_x.end = phoneme_data[-1][2]
            finally:
                _updating_phonemes[0] = False

        phoneme_bdry_src.on_change('data', _sync_phoneme_rects)

        finish_ph_btn = pn.widgets.Button(
            name='✓ Finish Phonemes  (sync Pitch / Loudness)',
            button_type='danger', width=310,
        )
        phoneme_panel = pn.Column(
            pn.pane.Bokeh(p_ph),
            pn.Row(finish_ph_btn),
            margin=(0, 0, 4, 0),
        )

    elif has_words:
        def _word_source_data(words):
            durations = [float(word["end"]) - float(word["start"]) for word in words]
            selected_idx = _selected_word_idx[0]
            return dict(
                cx=[(float(word["start"]) + float(word["end"])) / 2 for word in words],
                cw=durations,
                cy=[0.52] * len(words),
                ch=[0.72] * len(words),
                color=[WORD_COLORS[i % len(WORD_COLORS)] for i in range(len(words))],
                alpha=[1.0 if i == selected_idx else 0.88 for i in range(len(words))],
                line_color=[
                    PITCH_COLOR if i == selected_idx else '#FFFFFF'
                    for i in range(len(words))
                ],
                line_width=[3.0 if i == selected_idx else 1.5 for i in range(len(words))],
                label=[str(word["label"]) for word in words],
                duration=durations,
                duration_text=[f'{duration:.2f} s' for duration in durations],
            )

        word_rect_src = ColumnDataSource(_word_source_data(word_data))
        p_word = figure(
            width=W,
            height=148,
            title="Word duration",
            x_range=shared_x,
            y_range=(-0.02, 1.02),
            tools="xpan,xwheel_zoom,reset",
            toolbar_location="right",
        )
        _style_plot(p_word, show_y_grid=False)
        p_word.yaxis.visible = False
        p_word.xaxis.axis_label = "Time (s)"
        p_word.min_border_left = 80
        word_glyph = p_word.rect(
            x="cx",
            y="cy",
            width="cw",
            height="ch",
            fill_color="color",
            fill_alpha="alpha",
            line_color="line_color",
            line_width="line_width",
            source=word_rect_src,
        )
        p_word.text(
            x="cx",
            y=0.63,
            text="label",
            text_align="center",
            text_baseline="middle",
            text_font_size="10px",
            text_font_style="bold",
            text_color=PLOT_TEXT_COLOR,
            source=word_rect_src,
        )
        p_word.text(
            x="cx",
            y=0.37,
            text="duration_text",
            text_align="center",
            text_baseline="middle",
            text_font_size="8px",
            text_color=PLOT_MUTED_COLOR,
            source=word_rect_src,
        )
        p_word.add_tools(HoverTool(
            renderers=[word_glyph],
            tooltips=[("Word", "@label"), ("Duration", "@duration{0.00}s")],
        ))
        word_panel = pn.Column(
            pn.pane.Bokeh(p_word),
            margin=(0, 0, 4, 0),
        )

        if has_waveform:
            waveform_src = ColumnDataSource(
                _waveform_envelope(current_waveform, waveform_sample_rate)
            )
            waveform_peak = max(
                0.05,
                float(np.max(np.abs(current_waveform))),
            )
            p_waveform = figure(
                width=W,
                height=164,
                title="Requested timing preview · waveform (not regenerated audio)",
                x_range=shared_x,
                y_range=Range1d(-waveform_peak * 1.08, waveform_peak * 1.08),
                tools="xpan,xwheel_zoom,reset",
                toolbar_location="right",
            )
            _style_plot(p_waveform, show_y_grid=False)
            p_waveform.xaxis.visible = False
            p_waveform.yaxis.axis_label = "Amplitude"
            p_waveform.min_border_left = 80
            p_waveform.segment(
                x0="x",
                y0="low",
                x1="x",
                y1="high",
                source=waveform_src,
                line_color="#526D82",
                line_alpha=0.88,
                line_width=1.0,
            )
            waveform_selection = BoxAnnotation(
                left=0,
                right=0,
                fill_color=PITCH_COLOR,
                fill_alpha=0.10,
                line_color=PITCH_COLOR,
                line_alpha=0.75,
                line_width=2,
                visible=False,
                level="underlay",
            )
            p_waveform.add_layout(waveform_selection)
            waveform_panel = pn.Column(
                pn.pane.Bokeh(p_waveform),
                margin=(0, 0, 0, 0),
            )

    # ══════════════════════════════════════════════════════════════════════════
    # PITCH PLOT
    # ══════════════════════════════════════════════════════════════════════════
    # We use multi_line (xs=[[...]], ys=[[...]]) so FreehandDrawTool can
    # replace the curve in-place.  PolyEditTool also works on multi_line.
    pitch_src = ColumnDataSource({
        'xs': [time_axis.tolist()],
        'ys': [pitch_data.tolist()],
    })

    p_pitch = figure(
        width=W, height=270, title=pitch_title,
        x_range=shared_x, y_range=(-5, 133),
        tools="pan,wheel_zoom,reset",
        toolbar_location="right",
    )
    _style_plot(p_pitch)
    p_pitch.xaxis.axis_label = "Time (s)"
    p_pitch.yaxis.axis_label = "Pitch (coarse bins)"
    p_pitch.min_border_left = 80
    p_pitch.add_tools(CrosshairTool())

    p_pitch.line(
        time_axis,
        pitch_data,
        color="#AAB3C2",
        line_width=1.8,
        line_alpha=0.85,
        line_dash="dashed",
        legend_label="Baseline",
    )
    pitch_ml   = p_pitch.multi_line('xs', 'ys', source=pitch_src,
                                    color=PITCH_COLOR, line_width=3.0,
                                    line_cap='round', line_join='round',
                                    legend_label='Edited')
    p_pitch.legend.location = 'top_left'
    p_pitch.legend.orientation = 'horizontal'
    p_pitch.legend.background_fill_alpha = 0.0
    p_pitch.legend.border_line_alpha = 0.0
    p_pitch.legend.label_text_font_size = '9pt'
    # Keep the current contour while drawing. Bokeh appends the new stroke as
    # the last MultiLine row; Finish collapses the source back to one row.
    fhd_pitch  = FreehandDrawTool(renderers=[pitch_ml], num_objects=2)
    # PointDrawTool fallback: explicitly visible control points for robust editing.
    pitch_pts_src = ColumnDataSource({'x': time_axis.tolist(), 'y': pitch_data.tolist()})
    pitch_pts = p_pitch.scatter(
        x='x', y='y', source=pitch_pts_src, size=7, marker='circle',
        color=PITCH_COLOR, line_color='white', line_width=1.0,
        alpha=0.95, visible=False
    )
    pdt_pitch = PointDrawTool(renderers=[pitch_pts], add=False)
    if focus is None:
        p_pitch.add_tools(fhd_pitch, pdt_pitch)
    elif focus == "pitch":
        p_pitch.add_tools(fhd_pitch)

    _updating_pitch_points = [False]

    def _sync_pitch_points_to_line(attr, old, new):
        if _updating_pitch_points[0]:
            return
        _updating_pitch_points[0] = True
        try:
            x = np.asarray(pitch_pts_src.data.get('x', []), dtype=float)
            y = np.asarray(pitch_pts_src.data.get('y', []), dtype=float)
            if len(x) < 2:
                return

            old_x = np.asarray((old or {}).get('x', []), dtype=float)
            old_y = np.asarray((old or {}).get('y', []), dtype=float)
            if len(old_x) == len(x) and len(old_y) == len(y) and len(y) > 0:
                # Lock x on edit so "Edit Points" only changes values, not timing.
                x = old_x.copy()
                dy = y - old_y
                moved_idx = int(np.argmax(np.abs(dy)))
                y = _apply_local_delta(old_y, moved_idx, float(dy[moved_idx]))

            y = np.clip(y, 0, 128)
            pitch_pts_src.data = {'x': x.tolist(), 'y': y.tolist()}

            base_xs = np.asarray(pitch_src.data.get('xs', [[]])[0], dtype=float)
            if len(base_xs) == 0:
                base_xs = x
            dense_y = _reconstruct_curve_from_controls(x, y, base_xs, 0, 128)
            if dense_y is None:
                return
            pitch_src.data = {'xs': [base_xs.tolist()], 'ys': [dense_y.tolist()]}
        finally:
            _updating_pitch_points[0] = False

    pitch_pts_src.on_change('data', _sync_pitch_points_to_line)

    pitch_draw_btn   = pn.widgets.Toggle(name='✏ Draw Pitch',   button_type='success', width=210)
    pitch_edit_btn   = pn.widgets.Toggle(name='⊕ Edit Points',  button_type='primary',  width=150)
    pitch_finish_btn = pn.widgets.Button(name='✓ Finish Pitch', button_type='danger',   width=140)

    def _sync_pitch_draw_label():
        """Show whether a drag currently draws, so the toggle is never a guess."""
        if pitch_draw_btn.value:
            pitch_draw_btn.name = '✏ Drawing… ✓ Finish to lock'
            pitch_draw_btn.button_type = 'success'
        else:
            pitch_draw_btn.name = '✏ Draw Pitch'
            pitch_draw_btn.button_type = 'default'

    def _on_pitch_draw(event):
        if event.new:
            p_pitch.toolbar.active_drag = fhd_pitch
            pitch_edit_btn.value = False
        else:
            _restore_pan(p_pitch)
        _sync_pitch_draw_label()

    def _on_pitch_edit(event):
        if event.new:
            latest = _latest_valid_multiline(pitch_src.data)
            if latest is not None:
                latest_x, latest_y = latest
                ctrl_x, ctrl_y = _sample_sparse_control_points(latest_x, latest_y)
                pitch_pts_src.data = {'x': ctrl_x.tolist(), 'y': ctrl_y.tolist()}
            pitch_pts.visible = True
            p_pitch.toolbar.active_drag = pdt_pitch
            pitch_draw_btn.value = False
        else:
            pitch_pts.visible = False
            _restore_pan(p_pitch)

    def _arm_pitch_draw():
        """Keep freehand drawing active so the next sketch needs no re-arming.

        On the focused pitch page drawing is the whole point, and handing the
        drag gesture back to PanTool silently turns the next stroke into a pan:
        nothing reaches the server and Finish just re-aligns the same contour.
        """
        pitch_draw_btn.value = True
        pitch_edit_btn.value = False
        pitch_pts.visible = False
        p_pitch.toolbar.active_drag = fhd_pitch
        _sync_pitch_draw_label()

    def _on_pitch_finish(event):
        nonlocal _orig_pitch
        pitch_edit_btn.value = False
        pitch_pts.visible = False
        latest = _latest_valid_multiline(pitch_src.data)
        rows = len(pitch_src.data.get('xs', []))
        if latest is not None:
            drawn_x, drawn_y = latest
            aligned = _resample_to_time_axis(drawn_x, drawn_y, time_axis, 0, 128)
            if aligned is not None:
                pitch_src.data = {'xs': [time_axis.tolist()], 'ys': [aligned.tolist()]}
                _orig_pitch = aligned.copy()
                print(f"[Pitch] {rows} row(s) in source, stroke {len(drawn_x)} pts "
                      f"→ aligned {len(aligned)} frames, "
                      f"range [{aligned.min():.1f}, {aligned.max():.1f}]")
        # Finish locks the contour in and hands the drag gesture back, until
        # the user explicitly clicks Draw again.
        pitch_draw_btn.value = False
        _restore_pan(p_pitch)
        _sync_pitch_draw_label()

    pitch_draw_btn.param.watch(_on_pitch_draw,   'value')
    pitch_edit_btn.param.watch(_on_pitch_edit,   'value')
    pitch_finish_btn.on_click(_on_pitch_finish)

    # ══════════════════════════════════════════════════════════════════════════
    # LOUDNESS PLOT
    # ══════════════════════════════════════════════════════════════════════════
    loudness_src = ColumnDataSource({
        'xs': [time_axis.tolist()],
        'ys': [loudness_data.tolist()],
    })

    p_loudness = figure(
        width=W, height=270, title=loudness_title,
        x_range=shared_x, y_range=(-2, 66),
        tools="pan,wheel_zoom,reset",
        toolbar_location="right",
    )
    _style_plot(p_loudness)
    p_loudness.xaxis.axis_label = "Time (s)"
    p_loudness.yaxis.axis_label = "Loudness (coarse bins)"
    p_loudness.min_border_left = 80
    p_loudness.add_tools(CrosshairTool())

    p_loudness.line(
        time_axis,
        loudness_data,
        color="#AAB3C2",
        line_width=1.8,
        line_alpha=0.85,
        line_dash="dashed",
        legend_label="Baseline",
    )
    loudness_ml   = p_loudness.multi_line('xs', 'ys', source=loudness_src,
                                          color=LOUDNESS_COLOR, line_width=3.0,
                                          line_cap='round', line_join='round',
                                          legend_label='Edited')
    p_loudness.legend.location = 'top_left'
    p_loudness.legend.orientation = 'horizontal'
    p_loudness.legend.background_fill_alpha = 0.0
    p_loudness.legend.border_line_alpha = 0.0
    p_loudness.legend.label_text_font_size = '9pt'
    fhd_loudness  = FreehandDrawTool(renderers=[loudness_ml], num_objects=2)
    loudness_pts_src = ColumnDataSource({'x': time_axis.tolist(), 'y': loudness_data.tolist()})
    loudness_pts = p_loudness.scatter(
        x='x', y='y', source=loudness_pts_src, size=7, marker='triangle',
        color=LOUDNESS_COLOR, line_color='white', line_width=1.0,
        alpha=0.95, visible=False
    )
    pdt_loudness = PointDrawTool(renderers=[loudness_pts], add=False)
    if focus is None:
        p_loudness.add_tools(fhd_loudness, pdt_loudness)
    elif focus == "loudness":
        p_loudness.add_tools(fhd_loudness)

    _updating_loudness_points = [False]

    def _sync_loudness_points_to_line(attr, old, new):
        if _updating_loudness_points[0]:
            return
        _updating_loudness_points[0] = True
        try:
            x = np.asarray(loudness_pts_src.data.get('x', []), dtype=float)
            y = np.asarray(loudness_pts_src.data.get('y', []), dtype=float)
            if len(x) < 2:
                return

            old_x = np.asarray((old or {}).get('x', []), dtype=float)
            old_y = np.asarray((old or {}).get('y', []), dtype=float)
            if len(old_x) == len(x) and len(old_y) == len(y) and len(y) > 0:
                x = old_x.copy()
                dy = y - old_y
                moved_idx = int(np.argmax(np.abs(dy)))
                y = _apply_local_delta(old_y, moved_idx, float(dy[moved_idx]))

            y = np.clip(y, 0, 64)
            loudness_pts_src.data = {'x': x.tolist(), 'y': y.tolist()}

            base_xs = np.asarray(loudness_src.data.get('xs', [[]])[0], dtype=float)
            if len(base_xs) == 0:
                base_xs = x
            dense_y = _reconstruct_curve_from_controls(x, y, base_xs, 0, 64)
            if dense_y is None:
                return
            loudness_src.data = {'xs': [base_xs.tolist()], 'ys': [dense_y.tolist()]}
        finally:
            _updating_loudness_points[0] = False

    loudness_pts_src.on_change('data', _sync_loudness_points_to_line)

    loudness_draw_btn   = pn.widgets.Toggle(name='✏ Draw Loudness', button_type='success', width=210)
    loudness_edit_btn   = pn.widgets.Toggle(name='⊕ Edit Points',   button_type='primary',  width=150)
    loudness_finish_btn = pn.widgets.Button(name='✓ Finish Loudness', button_type='danger', width=155)

    def _sync_loudness_draw_label():
        """See ``_sync_pitch_draw_label``."""
        if loudness_draw_btn.value:
            loudness_draw_btn.name = '✏ Drawing… ✓ Finish to lock'
            loudness_draw_btn.button_type = 'success'
        else:
            loudness_draw_btn.name = '✏ Draw Loudness'
            loudness_draw_btn.button_type = 'default'

    def _on_loudness_draw(event):
        if event.new:
            p_loudness.toolbar.active_drag = fhd_loudness
            loudness_edit_btn.value = False
        else:
            _restore_pan(p_loudness)
        _sync_loudness_draw_label()

    def _on_loudness_edit(event):
        if event.new:
            latest = _latest_valid_multiline(loudness_src.data)
            if latest is not None:
                latest_x, latest_y = latest
                ctrl_x, ctrl_y = _sample_sparse_control_points(latest_x, latest_y)
                loudness_pts_src.data = {'x': ctrl_x.tolist(), 'y': ctrl_y.tolist()}
            loudness_pts.visible = True
            p_loudness.toolbar.active_drag = pdt_loudness
            loudness_draw_btn.value = False
        else:
            loudness_pts.visible = False
            _restore_pan(p_loudness)

    def _arm_loudness_draw():
        """Keep freehand drawing active; see ``_arm_pitch_draw``."""
        loudness_draw_btn.value = True
        loudness_edit_btn.value = False
        loudness_pts.visible = False
        p_loudness.toolbar.active_drag = fhd_loudness
        _sync_loudness_draw_label()

    def _on_loudness_finish(event):
        nonlocal _orig_loudness
        loudness_edit_btn.value = False
        loudness_pts.visible = False
        latest = _latest_valid_multiline(loudness_src.data)
        rows = len(loudness_src.data.get('xs', []))
        if latest is not None:
            drawn_x, drawn_y = latest
            aligned = _resample_to_time_axis(drawn_x, drawn_y, time_axis, 0, 64)
            if aligned is not None:
                loudness_src.data = {'xs': [time_axis.tolist()], 'ys': [aligned.tolist()]}
                _orig_loudness = aligned.copy()
                print(f"[Loudness] {rows} row(s) in source, stroke {len(drawn_x)} pts "
                      f"→ aligned {len(aligned)} frames, "
                      f"range [{aligned.min():.1f}, {aligned.max():.1f}]")
        # Finish locks the contour in; see ``_on_pitch_finish``.
        loudness_draw_btn.value = False
        _restore_pan(p_loudness)
        _sync_loudness_draw_label()

    loudness_draw_btn.param.watch(_on_loudness_draw,   'value')
    loudness_edit_btn.param.watch(_on_loudness_edit,   'value')
    loudness_finish_btn.on_click(_on_loudness_finish)

    # ══════════════════════════════════════════════════════════════════════════
    # GLOBAL VERTICAL REFERENCE LINE  (synced across all tracks)
    # ══════════════════════════════════════════════════════════════════════════
    ref_line_color = '#666666'
    ref_line_kwargs = dict(
        dimension='height',
        line_color=ref_line_color,
        line_width=1.2,
        line_alpha=0.9,
        line_dash='dotted',
    )

    p_pitch_ref = Span(location=np.nan, **ref_line_kwargs)
    p_loudness_ref = Span(location=np.nan, **ref_line_kwargs)
    p_pitch.add_layout(p_pitch_ref)
    p_loudness.add_layout(p_loudness_ref)

    ref_spans = [p_pitch_ref, p_loudness_ref]
    timeline_figure = p_word if has_words else p_ph
    if timeline_figure is not None:
        timeline_ref = Span(location=np.nan, **ref_line_kwargs)
        timeline_figure.add_layout(timeline_ref)
        ref_spans.append(timeline_ref)
    if p_waveform is not None:
        waveform_ref = Span(location=np.nan, **ref_line_kwargs)
        p_waveform.add_layout(waveform_ref)
        ref_spans.append(waveform_ref)

    sync_ref_line_js = CustomJS(args=dict(spans=ref_spans), code="""
        const x = cb_obj.x;
        if (x === null || Number.isNaN(x)) return;
        for (const span of spans) {
            span.location = x;
        }
    """)
    clear_ref_line_js = CustomJS(args=dict(spans=ref_spans), code="""
        for (const span of spans) {
            span.location = NaN;
        }
    """)

    figures_with_reference = [p_pitch, p_loudness]
    if timeline_figure is not None:
        figures_with_reference.append(timeline_figure)
    if p_waveform is not None:
        figures_with_reference.append(p_waveform)
    for fig in figures_with_reference:
        fig.js_on_event(MouseMove, sync_ref_line_js)
        fig.js_on_event(MouseLeave, clear_ref_line_js)

    # ══════════════════════════════════════════════════════════════════════════
    # PHONEME → PITCH / LOUDNESS SYNC  (triggered by "Finish Phonemes")
    # ══════════════════════════════════════════════════════════════════════════
    if has_phoneme_editor:
        def _on_finish_phonemes(event):
            nonlocal _orig_phonemes, _orig_pitch, _orig_loudness
            nonlocal time_axis, n_frames

            _orig_phonemes = [list(p) for p in phoneme_data]
            _orig_pitch    = np.array(pitch_src.data['ys'][0])
            _orig_loudness = np.array(loudness_src.data['ys'][0])

            new_time   = np.array(pitch_src.data['xs'][0])
            time_axis  = new_time
            n_frames   = len(new_time)

            new_dur = phoneme_data[-1][2]
            shared_x.end = new_dur

            print(f"[Phonemes] confirmed: {n_frames} frames, "
                  f"total duration {new_dur:.3f}s")

        finish_ph_btn.on_click(_on_finish_phonemes)

    # ══════════════════════════════════════════════════════════════════════════
    # DATA ACCESSORS
    # ══════════════════════════════════════════════════════════════════════════
    def get_pitch() -> np.ndarray:
        """Return the current pitch array, aligned to the current time axis."""
        latest = _latest_valid_multiline(pitch_src.data)
        if latest is None:
            return _orig_pitch.copy()
        curve_x, curve_y = latest
        if len(curve_x) == n_frames and np.allclose(curve_x, time_axis, atol=1e-6):
            return curve_y.copy()
        aligned = _resample_to_time_axis(curve_x, curve_y, time_axis, 0, 128)
        return aligned if aligned is not None else _orig_pitch.copy()

    def get_loudness() -> np.ndarray:
        """Return the current loudness array, aligned to the current time axis."""
        latest = _latest_valid_multiline(loudness_src.data)
        if latest is None:
            return _orig_loudness.copy()
        curve_x, curve_y = latest
        if len(curve_x) == n_frames and np.allclose(curve_x, time_axis, atol=1e-6):
            return curve_y.copy()
        aligned = _resample_to_time_axis(curve_x, curve_y, time_axis, 0, 64)
        return aligned if aligned is not None else _orig_loudness.copy()

    def get_phonemes():
        """Return current phoneme list [[phoneme, start, end], ...]."""
        return [list(phone) for phone in phoneme_data] if has_phonemes else None

    def get_words():
        """Return current word alignment dictionaries, or ``None``."""
        return [dict(word) for word in word_data] if has_words else None

    def get_waveform():
        """Return the current duration-preview waveform, or ``None``."""
        return current_waveform.copy() if has_waveform else None

    def _update_waveform_selection():
        """Keep the waveform highlight aligned with the selected word."""
        if waveform_selection is None:
            return
        selected_idx = _selected_word_idx[0]
        if selected_idx is None:
            waveform_selection.visible = False
            return
        word = word_data[selected_idx]
        waveform_selection.left = float(word["start"])
        waveform_selection.right = float(word["end"])
        waveform_selection.visible = True

    def select_word(word_idx):
        """Highlight one word in the duration track, or clear with ``None``."""
        if not has_words or word_rect_src is None:
            raise RuntimeError("word selection requires word_data")
        if word_idx is None:
            _selected_word_idx[0] = None
            word_rect_src.data = _word_source_data(word_data)
            _update_waveform_selection()
            return None

        try:
            word_idx = int(word_idx)
        except (TypeError, ValueError) as exc:
            raise TypeError("word_idx must be an integer or None") from exc
        if not 0 <= word_idx < len(word_data):
            raise IndexError(f"word_idx {word_idx} is out of range")

        _selected_word_idx[0] = word_idx
        word_rect_src.data = _word_source_data(word_data)
        _update_waveform_selection()
        return dict(word_data[word_idx])

    def _replace_timeline(
        new_pitch,
        new_loudness,
        new_phonemes,
        new_words,
        new_waveform=None,
    ):
        """Atomically replace all Python-side timeline state and Bokeh sources."""
        nonlocal time_axis, n_frames, total_duration
        nonlocal phoneme_data, word_data
        nonlocal _orig_pitch, _orig_loudness, _orig_phonemes
        nonlocal current_waveform

        new_pitch = np.asarray(new_pitch, dtype=float)
        new_loudness = np.asarray(new_loudness, dtype=float)
        if new_pitch.ndim != 1 or new_loudness.ndim != 1:
            raise ValueError("replacement curves must be one-dimensional")
        if len(new_pitch) != len(new_loudness):
            raise ValueError("replacement pitch/loudness lengths differ")

        new_time_axis = (
            np.arange(len(new_pitch), dtype=float) * hop_length / sample_rate
        )
        new_phonemes = [list(phone) for phone in new_phonemes]
        new_words = [dict(word) for word in new_words]
        if has_waveform:
            if new_waveform is None:
                raise ValueError("replacement waveform is required")
            new_waveform = np.asarray(new_waveform, dtype=float)
            if new_waveform.ndim != 1 or len(new_waveform) == 0:
                raise ValueError("replacement waveform must be a non-empty 1-D array")

        # Update closure state first so any subsequent server callback sees a
        # fully consistent timeline rather than a mixture of old/new lengths.
        phoneme_data = new_phonemes
        word_data = new_words
        time_axis = new_time_axis
        n_frames = len(new_pitch)
        total_duration = n_frames * hop_length / sample_rate
        _orig_pitch = new_pitch.copy()
        _orig_loudness = new_loudness.copy()
        _orig_phonemes = [list(phone) for phone in new_phonemes]
        if has_waveform:
            current_waveform = new_waveform.copy()

        # Reset only restores the curves; it must not change the drag mode.
        # Flipping the toggle off here would hand the gesture back to PanTool
        # mid-session, so the user's next stroke would pan instead of draw and
        # never reach the server. Only Draw and Finish change this mode.
        was_pitch_drawing = bool(pitch_draw_btn.value)
        was_loudness_drawing = bool(loudness_draw_btn.value)

        pitch_draw_btn.value = False
        pitch_edit_btn.value = False
        loudness_draw_btn.value = False
        loudness_edit_btn.value = False
        pitch_pts.visible = False
        loudness_pts.visible = False

        pitch_src.data = {
            'xs': [new_time_axis.tolist()],
            'ys': [new_pitch.tolist()],
        }
        loudness_src.data = {
            'xs': [new_time_axis.tolist()],
            'ys': [new_loudness.tolist()],
        }
        _updating_pitch_points[0] = True
        _updating_loudness_points[0] = True
        try:
            pitch_pts_src.data = {
                'x': new_time_axis.tolist(),
                'y': new_pitch.tolist(),
            }
            loudness_pts_src.data = {
                'x': new_time_axis.tolist(),
                'y': new_loudness.tolist(),
            }
        finally:
            _updating_pitch_points[0] = False
            _updating_loudness_points[0] = False
        if word_rect_src is not None:
            word_rect_src.data = _word_source_data(word_data)
        if waveform_src is not None:
            waveform_src.data = _waveform_envelope(
                current_waveform,
                waveform_sample_rate,
            )
        _update_waveform_selection()
        shared_x.start = 0
        current_waveform_duration = (
            len(current_waveform) / waveform_sample_rate if has_waveform else 0.0
        )
        last_word_end = float(word_data[-1]["end"]) if word_data else 0.0
        shared_x.end = max(
            total_duration,
            current_waveform_duration,
            last_word_end,
            hop_length / sample_rate,
        )

        # Re-arm freehand drawing on the refreshed curves so the user can keep
        # sketching straight after Reset instead of panning by accident.
        if was_pitch_drawing:
            _arm_pitch_draw()
        if was_loudness_drawing:
            _arm_loudness_draw()

    def apply_word_duration(word_idx, seconds):
        """Apply one word duration edit and return a machine/UI-friendly summary."""
        if not has_words:
            raise RuntimeError("word duration editing requires word_data")
        current_pitch = get_pitch()
        current_loudness = get_loudness()
        result = retime_word_curves(
            current_pitch,
            current_loudness,
            phoneme_data,
            word_data,
            word_idx,
            seconds,
            hop_length=hop_length,
            sample_rate=sample_rate,
        )
        new_pitch, new_loudness, new_phonemes, new_words, summary = result
        new_waveform = None
        if has_waveform:
            edited_word = word_data[summary["word_idx"]]
            new_waveform = _retime_waveform_segment(
                current_waveform,
                waveform_sample_rate,
                float(edited_word["start"]),
                float(edited_word["end"]),
                float(summary["new_duration"]),
            )
        _replace_timeline(
            new_pitch,
            new_loudness,
            new_phonemes,
            new_words,
            new_waveform,
        )
        return summary

    def reset_timeline():
        """Restore the curves and word/phone boundaries supplied at construction."""
        if not has_words:
            raise RuntimeError("reset_timeline is available on word-aware layouts")
        _replace_timeline(
            initial_pitch,
            initial_loudness,
            initial_phonemes,
            initial_words,
            initial_waveform,
        )
        return {
            "reset": True,
            "total_frames": len(initial_pitch),
            "total_duration": len(initial_pitch) * hop_length / sample_rate,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # ASSEMBLE LAYOUT
    # ══════════════════════════════════════════════════════════════════════════
    pitch_plot_pane = pn.pane.Bokeh(p_pitch)
    loudness_plot_pane = pn.pane.Bokeh(p_loudness)
    pitch_panel = pn.Column(
        pitch_plot_pane,
        pn.Row(pitch_draw_btn, pitch_edit_btn, pitch_finish_btn,
               margin=(2, 0, 8, 0)),
        margin=0,
    )
    loudness_panel = pn.Column(
        loudness_plot_pane,
        pn.Row(loudness_draw_btn, loudness_edit_btn, loudness_finish_btn,
               margin=(2, 0, 4, 0)),
        margin=0,
    )
    focused_pitch_panel = pn.Column(
        pitch_plot_pane,
        pn.Row(pitch_draw_btn, pitch_finish_btn, margin=(2, 0, 8, 0)),
        margin=0,
    )
    focused_loudness_panel = pn.Column(
        loudness_plot_pane,
        pn.Row(loudness_draw_btn, loudness_finish_btn, margin=(2, 0, 4, 0)),
        margin=0,
    )
    duration_panel = pn.Column(
        *([waveform_panel] if waveform_panel is not None else []),
        word_panel,
        margin=0,
    )

    if focus == "pitch":
        items = [focused_pitch_panel]
    elif focus == "loudness":
        items = [focused_loudness_panel]
    elif focus == "duration":
        items = [duration_panel]
    else:
        items = []
        if word_panel is not None:
            items.append(word_panel)
            items.append(pn.pane.HTML(
                '<hr style="border-top:1px solid #ddd;margin:4px 0">'
            ))
        elif has_phoneme_editor and phoneme_panel is not None:
            items.append(phoneme_panel)
            items.append(pn.pane.HTML(
                '<hr style="border-top:1px solid #ddd;margin:4px 0">'
            ))
        items.extend([pitch_panel, loudness_panel])

    layout = pn.Column(*items, sizing_mode='stretch_width')

    # Drawing starts off: dragging pans until the user clicks Draw.
    _sync_pitch_draw_label()
    _sync_loudness_draw_label()

    # Attach accessors and raw sources as attributes
    layout.get_pitch     = get_pitch
    layout.get_loudness  = get_loudness
    layout.get_phonemes  = get_phonemes
    layout.get_words     = get_words
    layout.get_waveform  = get_waveform
    layout.select_word   = select_word
    layout.apply_word_duration = apply_word_duration
    layout.reset_timeline = reset_timeline
    layout._pitch_src    = pitch_src
    layout._loudness_src = loudness_src
    layout._word_src     = word_rect_src
    layout._waveform_src = waveform_src
    layout._waveform_sample_rate = waveform_sample_rate
    layout.focus         = focus

    return layout


# ─────────────────────────────────────────────────────────────────────────────
# Demo (run as script: python -m inference.interactive_plot_bokeh)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    hop_length, sample_rate = 256, 24_000
    n = int(2.0 * sample_rate / hop_length)
    t = np.linspace(0, 4 * np.pi, n)

    pitch    = np.clip(np.sin(t) * 40 + 64 + np.random.randn(n) * 5,  0, 128)
    loudness = np.clip(np.cos(t) * 15 + 32 + np.random.randn(n) * 3,  0,  64)

    phonemes = [
        ["SIL", 0.00, 0.5],  ["F",  0.5, 1.0],
        ["AA",  1.0, 1.50], ["K",   1.50, 2.00]
    ]

    pn.extension()
    layout = interactive_dual_plot_bokeh(pitch, loudness, phonemes, hop_length, sample_rate)
    pn.serve(layout, show=True, title="DiTAR Interactive Editor")
