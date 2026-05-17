"""Phase 10 — pick a random subset of files from a source tree whose total
size approximates --target-gb, and materialize them into --dst.

Per PRD §6 Phase 10: "random sampling utility (uses os.walk + reservoir
sampling on file sizes)". Reservoir-style sampling under a *byte* budget
(rather than a count budget) is implemented by uniformly shuffling the
candidate list with a seeded RNG, then taking the prefix whose cumulative
size first reaches the target. That yields a uniformly random subset
constrained by size — the spirit of reservoir sampling for this problem.

The destination is flat (chunker_text scans non-recursively). Basename
collisions are resolved by prefixing with an 8-char hash of the source path."""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
import sys
from pathlib import Path

DEFAULT_EXTS = (".txt", ".json")


def walk_with_sizes(root: Path, exts: tuple[str, ...]) -> list[tuple[Path, int]]:
    files: list[tuple[Path, int]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if not name.lower().endswith(exts):
                continue
            p = Path(dirpath) / name
            try:
                size = p.stat().st_size
            except OSError as e:
                print(f"WARN: stat failed for {p}: {e}", file=sys.stderr)
                continue
            files.append((p, size))
    return files


def select_slice(
    candidates: list[tuple[Path, int]],
    target_bytes: int,
    seed: int,
) -> list[tuple[Path, int]]:
    """Shuffle deterministically with `seed`, then take the prefix whose
    cumulative byte count first reaches or exceeds `target_bytes`. Returns
    the entire shuffled corpus if it sums to less than the target."""
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    chosen: list[tuple[Path, int]] = []
    total = 0
    for path, size in shuffled:
        chosen.append((path, size))
        total += size
        if total >= target_bytes:
            break
    return chosen


def dst_name(src: Path, used: set[str]) -> str:
    name = src.name
    if name not in used:
        return name
    h = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:8]
    return f"{h}_{name}"


def materialize(
    chosen: list[tuple[Path, int]],
    dst: Path,
    mode: str = "hardlink",
) -> tuple[int, int]:
    """Place chosen files in `dst`. Returns (n_placed, bytes_placed).
    `mode` is one of {hardlink, copy, symlink}; hardlink falls back to copy
    if the platform/filesystem rejects it (e.g. cross-device)."""
    dst.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    placed = 0
    total = 0
    for src, size in chosen:
        name = dst_name(src, used)
        used.add(name)
        out = dst / name
        if out.exists():
            out.unlink()
        try:
            if mode == "hardlink":
                os.link(src, out)
            elif mode == "symlink":
                os.symlink(src, out)
            else:
                shutil.copy2(src, out)
        except OSError as e:
            if mode == "hardlink":
                try:
                    shutil.copy2(src, out)
                except OSError as e2:
                    print(f"WARN: copy fallback failed for {src}: {e2}", file=sys.stderr)
                    continue
            else:
                print(f"WARN: failed to place {src} -> {out}: {e}", file=sys.stderr)
                continue
        placed += 1
        total += size
    return placed, total


def _fmt_gb(n: int) -> str:
    return f"{n / 1024 / 1024 / 1024:.2f} GB"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Random size-bounded slice of a transcript tree.")
    p.add_argument("--src", required=True, help="Root directory to scan recursively.")
    p.add_argument("--dst", required=True, help="Destination directory for the slice (flat).")
    p.add_argument("--target-gb", type=float, default=100.0, help="Target slice size in GB.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible selection.")
    p.add_argument("--ext", action="append", default=None,
                   help="File extension to include (repeatable). Default: .txt and .json")
    p.add_argument("--mode", choices=("hardlink", "copy", "symlink"), default="hardlink",
                   help="How to place files in --dst (default: hardlink).")
    p.add_argument("--dry-run", action="store_true",
                   help="Select and report only; do not place any files.")
    args = p.parse_args(argv)

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.is_dir():
        print(f"ERROR: --src is not a directory: {src}", file=sys.stderr)
        return 2

    raw_exts = args.ext or list(DEFAULT_EXTS)
    exts = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in raw_exts)
    target_bytes = int(args.target_gb * 1024 * 1024 * 1024)

    print(f"Scanning {src} for {exts} ...", flush=True)
    candidates = walk_with_sizes(src, exts)
    if not candidates:
        print("ERROR: no candidate files found", file=sys.stderr)
        return 1
    corpus_bytes = sum(s for _, s in candidates)
    print(f"Found {len(candidates):,} files, total {_fmt_gb(corpus_bytes)}.", flush=True)
    print(f"Target slice: {_fmt_gb(target_bytes)} (seed={args.seed}).", flush=True)

    chosen = select_slice(candidates, target_bytes, args.seed)
    chosen_bytes = sum(s for _, s in chosen)
    print(f"Selected {len(chosen):,} files, ~{_fmt_gb(chosen_bytes)}.", flush=True)

    if args.dry_run:
        print("Dry run — not placing files.", flush=True)
        return 0

    placed, placed_bytes = materialize(chosen, dst, mode=args.mode)
    print(f"Placed {placed:,} files ({_fmt_gb(placed_bytes)}) in {dst} via {args.mode}.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
