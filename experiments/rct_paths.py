"""Resolve the artifacts output folder for experiment scripts.

Order: explicit ``cli_out`` (an --out flag), then the ``RCT_ARTIFACTS``
environment variable, then ``./artifacts`` relative to the current folder.
The folder is created if missing.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "RCT_ARTIFACTS"


def artifacts_dir(cli_out: str | os.PathLike | None = None) -> Path:
    raw = cli_out or os.environ.get(ENV_VAR) or "artifacts"
    out = Path(raw).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    return out
