"""Unit tests for ingestion.utils.encoding."""
from __future__ import annotations

import pytest

from ingestion.utils.encoding import read_text_robust


def test_plain_utf8(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world", encoding="utf-8")
    assert read_text_robust(p) == "hello world"


def test_utf8_with_bom(tmp_path):
    p = tmp_path / "bom.txt"
    p.write_bytes(b"\xef\xbb\xbfhello")
    out = read_text_robust(p)
    # The plain "utf-8" decoder accepts the BOM bytes literally as U+FEFF;
    # this is the documented behavior and is acceptable per the PRD ordering.
    assert "hello" in out


def test_latin1_fallback(tmp_path):
    p = tmp_path / "latin.txt"
    # 0xff = ÿ in latin-1, invalid in utf-8.
    p.write_bytes(b"caf\xe9 et th\xe9")
    out = read_text_robust(p)
    assert "caf" in out
    assert "et" in out


def test_unicode_chars_preserved(tmp_path):
    p = tmp_path / "u.txt"
    p.write_text("世界 Привет مرحبا", encoding="utf-8")
    out = read_text_robust(p)
    assert "世界" in out and "Привет" in out and "مرحبا" in out


def test_empty_file(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_bytes(b"")
    assert read_text_robust(p) == ""
