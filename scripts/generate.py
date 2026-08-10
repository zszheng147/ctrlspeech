#!/usr/bin/env python3
"""Run the CtrlSpeech CLI straight from a checkout, without installing.

`pip install -e .` gives you the same thing as a `ctrlspeech` command.
Run with --help for the full option list.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctrlspeech.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
