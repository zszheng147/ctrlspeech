#!/usr/bin/env python3
"""Headless end-to-end check of the demo's three control paths.

Drives the Panel app the way a browser would — Generate, apply a control,
Regenerate — and reads the app's own comparison output back to confirm the
model moved the quantity that was edited. Needs a GPU, the checkpoints and MFA.

    CTRLSPEECH_ASSETS=/path/to/staged/repo python tests/test_demo_flow.py
"""

import asyncio
import inspect
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "demo"))

import panel as pn  # noqa: E402

import app as demo  # noqa: E402

LOOP = asyncio.new_event_loop()


def click(button):
    """Invoke a Button's own handler.

    ``on_click`` registers through ``param.watch`` on ``clicks``; index 0 is
    always Panel's internal ``_param_change``, so filter for the app's function.
    """
    handlers = [
        w.fn for w in button.param.watchers["clicks"]["value"]
        if inspect.iscoroutinefunction(w.fn)
        or getattr(w.fn, "__name__", "").startswith(("on_", "apply_", "reset_"))
    ]
    if not handlers:
        raise AssertionError(f"no app handler on button {button.name!r}")
    result = handlers[0](None)
    if inspect.isawaitable(result):
        LOOP.run_until_complete(result)


def walk(node):
    yield node
    for child in getattr(node, "objects", []) or []:
        yield from walk(child)


def find(container, predicate):
    return next((node for node in walk(container) if predicate(node)), None)


def button_named(container, name):
    found = find(
        container,
        lambda o: isinstance(o, pn.widgets.Button) and o.name == name,
    )
    assert found is not None, f"button {name!r} not found"
    return found


def markdown_matching(container, pattern):
    regex = re.compile(pattern)
    return [
        node.object for node in walk(container)
        if isinstance(node, pn.pane.Markdown)
        and isinstance(node.object, str)
        and regex.search(node.object)
    ]


def run_case(focus):
    layout = demo.build_app()

    focus_group = find(
        layout,
        lambda o: isinstance(o, pn.widgets.RadioButtonGroup)
        and o.options == demo.CONTROL_LABELS,
    )
    assert focus_group is not None, "control-to-edit selector not found"
    focus_group.value = focus

    click(button_named(layout, "Generate baseline & align"))

    editor = find(layout, lambda o: hasattr(o, "get_pitch") and hasattr(o, "get_words"))
    assert editor is not None, "baseline generation produced no editor"

    click(button_named(layout, "Apply"))

    # The regeneration section is created by install_baseline, so it only exists
    # once the baseline is in place.
    regen_btn = button_named(layout, "Regenerate with current control")
    click(regen_btn)

    errors = markdown_matching(layout, r"^Error:")
    assert not errors, f"app reported: {errors[0]}"

    done = markdown_matching(layout, r"Controlled generation and comparison complete")
    assert done, "regeneration never reported completion"

    # Read the app's own before/after numbers rather than trusting the status text.
    if focus == "pitch":
        measured = markdown_matching(layout, r"Voiced pitch mean")
        assert measured, "no pitch comparison was rendered"
        before, after = _two_numbers(measured[0])
        assert after > before, f"pitch did not rise: {before} -> {after}"
    elif focus == "loudness":
        measured = markdown_matching(layout, r"Loudness mean")
        assert measured, "no loudness comparison was rendered"
        before, after = _two_numbers(measured[0])
        assert after > before, f"loudness did not rise: {before} -> {after}"
    else:
        measured = markdown_matching(layout, r"baseline .* → requested .* → achieved")
        assert measured, "duration was never verified with MFA"
        before, requested, achieved = (
            float(v) for v in re.findall(r"([\d.]+)s", measured[0])[:3]
        )
        assert achieved > before * 1.4, (
            f"word did not stretch: {before}s -> {achieved}s "
            f"(requested {requested}s)"
        )
    print(f"  {focus:9s} OK — {measured[0]}")


def _two_numbers(text):
    values = [float(v) for v in re.findall(r"[\d.]+(?=\s*(?:→|bins))", text)]
    assert len(values) >= 2, f"could not read two numbers from {text!r}"
    return values[0], values[1]


def main():
    failures = []
    for focus in ("pitch", "loudness", "duration"):
        print(f"[{focus}]")
        try:
            run_case(focus)
        except Exception as exc:  # noqa: BLE001 - report all three, not just the first
            failures.append(f"{focus}: {exc}")
            print(f"  {focus:9s} FAILED — {exc}")
    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\nAll three control paths verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
