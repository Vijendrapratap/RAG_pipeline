"""Serve one track's transcript and its isolated-vocals audio.

The archive already knows where everything is; nothing here discovers anything.

  * ``file_meta.source_file`` is the identity key *and* the folder path:
    ``Dagshai 2002/03 MAR - 2002/…/02 OM GURUVE NAMAH.json``
  * the transcript on disk is that path with ``_isolation`` on every folder and
    ``.raw.json`` on the leaf
  * the isolated WAV's absolute path is recorded *inside* the transcript, as
    ``audio_file`` — written there by ``transcribe_whisperx.write_raw_outputs``

So the WAV is never guessed. It is read.

**The inverse is lossy by construction.** ``chunker_cleaned._rel_parents``
*deletes* any folder named ``turbo`` or containing ``_model-``, and
``path_parser._strip_folder_suffix`` collapses ``_isolation`` and ``_model-*``
onto the same output. Two different disk trees can therefore produce the same
key. It happens to be exact for the tree that was actually ingested
(``D:\\Transcription whisperx\\Output``: 9,335 ``.raw.json``, zero scaffolding
folders, uniform ``_isolation``), and ``tests/unit/test_rag_api_tracks.py``
pins that with a parity check over every real key. If the convention drifts,
the test goes red before the archive does.

**Path traversal is structurally impossible, not merely filtered.** A caller
cannot name a file that is not a row: ``/api/track/transcript`` looks the key
up in ``file_meta`` before touching disk, and ``/api/track/audio`` accepts only
an HMAC ticket that route minted for that exact key. ``_safe_join`` re-checks
containment anyway.

**Why a ticket at all.** Auth is the ``X-Dashboard-Password`` header, and an
``<audio src>`` element issues a bare GET — it cannot carry one. Putting the
password in the query string would write it into every access log. The ticket
is bound to one ``source_file``, lives 5 minutes, and is keyed by a per-process
random secret: no password material in the HMAC, and tickets simply die on
restart (the panel re-mints).
"""
from __future__ import annotations

import glob as _glob
import hashlib
import hmac
import json
import logging
import secrets
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("rag_api.tracks")

TICKET_TTL_S = 300

# Per-process, never persisted, never derived from the dashboard password.
_TICKET_SECRET = secrets.token_bytes(32)


# --------------------------------------------------------------------------
# Path inverse
# --------------------------------------------------------------------------

def split_key(source_file: str) -> tuple[list[str], str]:
    """``a/b/track.json`` -> ``(["a", "b"], "track")``.

    Raises ValueError on anything that could escape the root. Folder names in
    this archive really do contain ``$``, ``&``, ``#`` and spaces, so the check
    is a segment blacklist, not a character whitelist.
    """
    if not source_file or source_file != source_file.strip():
        raise ValueError("empty or padded key")
    if "\\" in source_file or source_file.startswith("/"):
        raise ValueError("backslash or absolute path in key")
    parts = source_file.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ValueError("empty or relative segment in key")
    leaf = parts[-1]
    if not leaf.endswith(".json"):
        raise ValueError("key does not name a .json transcript")
    return parts[:-1], leaf[: -len(".json")]


def _safe_join(root: Path, *parts: str) -> Path:
    """Join under `root` and prove the result stayed there.

    Belt-and-braces: `split_key` already refused `..`, and the key had to match
    a `file_meta` row. This catches the symlink case those two cannot see.
    """
    p = (root / Path(*parts)).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError(f"path escaped root: {p}")
    return p


def transcript_path(source_file: str, root: Path) -> Path:
    """Disk path of the raw transcript. The `_isolation` suffix that the
    isolation pipeline appended to every folder was stripped to build the key;
    put it back."""
    folders, stem = split_key(source_file)
    return _safe_join(root, *[f + "_isolation" for f in folders], f"{stem}.raw.json")


def cleaned_path(source_file: str, root: Path) -> Path:
    folders, stem = split_key(source_file)
    return _safe_join(root, *[f + "_isolation" for f in folders], f"{stem}.cleaned.json")


def audio_path(audio_file: str | None, source_file: str, root: Path) -> Path | None:
    """Locate the isolated WAV under `root` — ``GuruAudio/Output``, the folder
    that *holds* the per-model `.ckpt` trees, not one of them.

    That distinction is load-bearing. The three tracks `transcribe_local.py` left
    with degenerate keys were transcribed from the **second** isolation model's
    tree (`melband_roformer_big_beta4.ckpt`), not the one every other track came
    from. Root at a single model and their audio is silently unreachable, even
    though the files are right there. So the `.ckpt` component stays in the path.

    Prefers `audio_file` from the transcript — the path the transcriber actually
    opened, recorded at the time. It is a Windows absolute path, so we keep the
    `.ckpt` folder and everything under it, and re-root that at the mount.

    Falls back to globbing `*.ckpt/<mirrored folders>/<stem>_model-*.wav`: the
    model number is not in the key, and hardcoding `_model-1_…` would 404 every
    track the day a second model is enabled.
    """
    rel: list[str] | None = None

    if audio_file:
        segs = [s for s in audio_file.replace("\\", "/").split("/") if s]
        for i, seg in enumerate(segs):
            if seg.endswith(".ckpt"):
                rel = segs[i:]  # keep the model folder — see the docstring
                break
        if rel is None:
            log.warning("audio_file names no .ckpt tree, falling back to glob: %r",
                        audio_file)
        elif any(p in (".", "..") for p in rel):
            log.warning("audio_file has a relative segment, ignoring it: %r", audio_file)
            rel = None

    if rel is not None:
        try:
            p = _safe_join(root, *rel)
        except ValueError as e:
            log.error("audio_file escaped the isolated root (%s): %r", e, audio_file)
            return None
        if p.is_file():
            return p
        log.warning("audio_file names a WAV that is not on disk: %s", p)

    folders, stem = split_key(source_file)
    pattern = "/".join(
        ["*.ckpt", *[_glob.escape(f + "_isolation") for f in folders],
         f"{_glob.escape(stem)}_model-*.wav"]
    )
    try:
        matches = sorted(m for m in root.glob(pattern) if m.is_file())
    except OSError as e:
        log.error("cannot scan isolated root for %r: %s: %s", source_file,
                  type(e).__name__, e)
        return None
    if not matches:
        return None
    if len(matches) > 1:
        log.info("%d isolated WAVs for %r; serving %s", len(matches), source_file,
                 matches[0].name)
    try:
        return _safe_join(root, *matches[0].relative_to(root).parts)
    except ValueError as e:
        log.error("globbed WAV escaped root for %r: %s", source_file, e)
        return None


# --------------------------------------------------------------------------
# Ticket
# --------------------------------------------------------------------------

def mint_ticket(source_file: str, *, now: float | None = None,
                secret: bytes | None = None) -> str:
    exp = int((now if now is not None else time.time()) + TICKET_TTL_S)
    return f"{exp}.{_sign(source_file, exp, secret or _TICKET_SECRET)}"


def check_ticket(source_file: str, ticket: str, *, now: float | None = None,
                 secret: bytes | None = None) -> bool:
    exp_s, _, sig = (ticket or "").partition(".")
    if not sig:
        return False
    try:
        exp = int(exp_s)
    except ValueError:
        return False
    if exp < (now if now is not None else time.time()):
        return False
    expected = _sign(source_file, exp, secret or _TICKET_SECRET)
    return hmac.compare_digest(sig, expected)


def _sign(source_file: str, exp: int, secret: bytes) -> str:
    msg = f"{exp}:{source_file}".encode("utf-8")
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()[:32]


# --------------------------------------------------------------------------
# Transcript shaping
# --------------------------------------------------------------------------

def slim(raw: dict[str, Any]) -> dict[str, Any]:
    """Segment-level view of a raw transcript: ~118 KB of JSON down to ~20 KB.

    `words` (with per-word start/end/score) is dropped — segment granularity is
    enough to click a line and seek to it. The word timings stay on disk for
    whenever someone wants a karaoke view.
    """
    segments = []
    for s in raw.get("segments") or []:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        try:
            start, end = float(s["start"]), float(s["end"])
        except (KeyError, TypeError, ValueError):
            log.warning("segment without usable start/end, skipped: %r", s.get("id"))
            continue
        segments.append({"start": round(start, 3), "end": round(end, 3), "text": text})

    return {
        "duration": raw.get("duration"),
        "language": raw.get("language"),
        "model": raw.get("model"),
        "segments": segments,
    }


def read_json(path: Path) -> dict[str, Any] | None:
    """Read a transcript. Returns None (logged) rather than raising: a truncated
    `.raw.json` is a known hazard — `transcribe_whisperx` writes it without the
    `.tmp` + `os.replace` the isolation pipeline uses, so a killed run can leave
    one behind, and `is_transcribed()` will call it complete forever."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        log.error("corrupt transcript %s: %s (line %d col %d)", path, e.msg,
                  e.lineno, e.colno)
        return None
    except OSError as e:
        log.error("cannot read transcript %s: %s: %s", path, type(e).__name__, e)
        return None
    if not isinstance(doc, dict):
        log.error("transcript %s is a %s, not an object", path, type(doc).__name__)
        return None
    return doc


def tree_is_mounted(root: Path) -> bool:
    """True when the bind mount actually landed. Docker happily creates an empty
    directory for a missing bind source, so `exists()` alone proves nothing."""
    try:
        return root.is_dir() and next(root.iterdir(), None) is not None
    except OSError as e:
        log.error("cannot stat media root %s: %s: %s", root, type(e).__name__, e)
        return False
