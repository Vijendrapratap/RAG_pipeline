"""Robust text decoding for the long tail of transcript encodings.

Per PRD §6 Phase 4 deliverable 2: try utf-8, utf-8-sig, latin-1 in order.
Returns the first that decodes without UnicodeDecodeError. Raises if all fail.
"""
from __future__ import annotations

from pathlib import Path

ENCODINGS = ("utf-8", "utf-8-sig", "latin-1")


def read_text_robust(path: Path) -> str:
    """Decode `path` using the first encoding that succeeds. Raises the last
    UnicodeDecodeError if none work."""
    last: UnicodeDecodeError | None = None
    raw = path.read_bytes()
    for enc in ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError as e:
            last = e
            continue
    assert last is not None
    raise last
