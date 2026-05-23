"""Unit tests for scripts/select_random_slice.py — exercises the pure
helpers (walk_with_sizes, select_slice, dst_name, materialize) and the
CLI's dry-run path. The full preflight pipeline is exercised by
scripts/preflight.sh against real data; we don't try to mock that."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# scripts/ is not a package — load select_random_slice directly.
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import select_random_slice as srs  # noqa: E402


def _populate(root: Path, files: dict[str, int]) -> None:
    """Create files at `root / relpath` with `size` bytes each."""
    for rel, size in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * size)


def test_walk_with_sizes_filters_by_extension(tmp_path):
    _populate(tmp_path, {
        "a.txt": 100,
        "b.json": 200,
        "sub/c.md": 300,
        "sub/d.txt": 400,
        "sub/deeper/e.bin": 500,
    })
    out = srs.walk_with_sizes(tmp_path, (".txt", ".json"))
    by_name = {p.name: s for p, s in out}
    assert by_name == {"a.txt": 100, "b.json": 200, "d.txt": 400}


def test_walk_with_sizes_case_insensitive_extensions(tmp_path):
    _populate(tmp_path, {"A.TXT": 10, "B.Json": 20, "c.txt": 30})
    out = srs.walk_with_sizes(tmp_path, (".txt", ".json"))
    assert sorted(p.name for p, _ in out) == ["A.TXT", "B.Json", "c.txt"]


def test_select_slice_stops_at_target_bytes():
    items = [(Path(f"f{i}"), 100) for i in range(10)]
    chosen = srs.select_slice(items, target_bytes=350, seed=1)
    # Accumulates 100,200,300,400 — stops at the first ≥350.
    assert len(chosen) == 4
    assert sum(s for _, s in chosen) == 400


def test_select_slice_returns_all_when_corpus_smaller_than_target():
    items = [(Path(f"f{i}"), 50) for i in range(4)]
    chosen = srs.select_slice(items, target_bytes=10_000, seed=1)
    assert len(chosen) == 4


def test_select_slice_deterministic_with_same_seed():
    items = [(Path(f"f{i}"), 100) for i in range(20)]
    a = srs.select_slice(items, target_bytes=500, seed=42)
    b = srs.select_slice(items, target_bytes=500, seed=42)
    assert [p.name for p, _ in a] == [p.name for p, _ in b]


def test_select_slice_differs_with_different_seeds():
    items = [(Path(f"f{i}"), 100) for i in range(50)]
    a = srs.select_slice(items, target_bytes=500, seed=1)
    b = srs.select_slice(items, target_bytes=500, seed=2)
    assert [p.name for p, _ in a] != [p.name for p, _ in b]


def test_dst_name_returns_original_when_free():
    used: set[str] = set()
    assert srs.dst_name(Path("/x/y/a.txt"), used) == "a.txt"


def test_dst_name_prefixes_hash_on_collision():
    used = {"a.txt"}
    name = srs.dst_name(Path("/different/dir/a.txt"), used)
    assert name.endswith("_a.txt")
    assert len(name) == len("a.txt") + 1 + 8


def test_materialize_copy_places_all_files(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _populate(src, {"a.txt": 10, "b.txt": 20, "sub/c.txt": 30})
    chosen = srs.walk_with_sizes(src, (".txt",))
    placed, total = srs.materialize(chosen, dst, mode="copy")
    assert placed == 3
    assert total == 60
    placed_files = sorted(p.name for p in dst.iterdir())
    assert placed_files == ["a.txt", "b.txt", "c.txt"]


def test_materialize_copy_resolves_basename_collision(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _populate(src, {"x/dup.txt": 10, "y/dup.txt": 20})
    chosen = srs.walk_with_sizes(src, (".txt",))
    placed, total = srs.materialize(chosen, dst, mode="copy")
    assert placed == 2
    assert total == 30
    names = sorted(p.name for p in dst.iterdir())
    # one original, one hash-prefixed
    assert "dup.txt" in names
    assert any(n.endswith("_dup.txt") and n != "dup.txt" for n in names)


def test_main_dry_run_reports_and_does_not_materialize(tmp_path, capsys):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _populate(src, {"a.txt": 1024, "b.txt": 2048, "c.json": 4096})
    rc = srs.main([
        "--src", str(src), "--dst", str(dst),
        "--target-gb", "0.000005",  # ~5 KB
        "--seed", "7", "--dry-run",
    ])
    assert rc == 0
    assert not dst.exists() or not any(dst.iterdir())
    out = capsys.readouterr().out
    assert "Found 3 files" in out
    assert "Selected" in out
    assert "Dry run" in out


def test_main_errors_on_missing_src(tmp_path, capsys):
    rc = srs.main([
        "--src", str(tmp_path / "nope"), "--dst", str(tmp_path / "out"),
        "--target-gb", "1",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not a directory" in err


def test_main_errors_on_empty_corpus(tmp_path, capsys):
    src = tmp_path / "src"
    src.mkdir()
    rc = srs.main([
        "--src", str(src), "--dst", str(tmp_path / "out"),
        "--target-gb", "1",
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "no candidate files" in err
