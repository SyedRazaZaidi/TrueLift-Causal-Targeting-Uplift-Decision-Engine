from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PREPARED = DATA / "prepared"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def ensure_dirs() -> None:
    for p in (RAW, PREPARED, ARTIFACTS, REPORTS):
        p.mkdir(parents=True, exist_ok=True)
