"""Track media: the path inverse, the audio ticket, and transcript shaping.

The inverse in `rag_api.tracks` is a *reimplementation* of what
`ingestion.chunker_cleaned.qualified_source` undoes — it has to be, because
`services/rag_api/Dockerfile` copies only `rag_api/` into the image and
`ingestion/` is not importable there.

So the two can drift, and the archive would go quietly 404 everywhere. The
parity test at the bottom of this file is the guard: it walks the real
transcript tree and asserts that for every one of the 9,335 indexed keys,
`transcript_path(qualified_source(p)) == p`. It skips when the tree is not
mounted, so CI stays green off the ingest box — but on this box it must pass.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from rag_api.tracks import (
    audio_path, check_ticket, cleaned_path, mint_ticket, read_json, slim,
    split_key, transcript_path, tree_is_mounted,
)

KEY = ("Dagshai 2002/03 MAR - 2002/DAGSHAI 26 - 29 MAR 2002 HOLI CAMP/"
       "26 MAR - 1$ - 7 PM/02 OM GURUVE NAMAH.json")
SECRET = b"a" * 32


# --- split_key -------------------------------------------------------------


def test_a_real_key_splits_into_folders_and_a_stem() -> None:
    folders, stem = split_key(KEY)
    assert folders == ["Dagshai 2002", "03 MAR - 2002",
                       "DAGSHAI 26 - 29 MAR 2002 HOLI CAMP", "26 MAR - 1$ - 7 PM"]
    assert stem == "02 OM GURUVE NAMAH"


def test_the_degenerate_flat_key_has_no_folders() -> None:
    """transcribe_local.py's flat out-dir left 3 of these in the index."""
    assert split_key("03 AA HIMMAT KA EK KADAM.json") == ([], "03 AA HIMMAT KA EK KADAM")


@pytest.mark.parametrize("bad", [
    "../../../etc/passwd.json",
    "a/../../etc/passwd.json",
    "/etc/passwd.json",
    "a\\..\\b.json",
    "a//b.json",
    "a/./b.json",
    "",
    " a.json",
    "no-extension",
    "a/b.txt",
])
def test_a_key_that_could_escape_the_root_is_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        split_key(bad)


def test_a_dollar_sign_is_a_folder_name_not_a_metacharacter() -> None:
    folders, _ = split_key("Live Masters 2010/7 JAN - 1$ - 6 PM/04 PRAVACHAN.json")
    assert folders[1] == "7 JAN - 1$ - 6 PM"


# --- transcript_path -------------------------------------------------------


def test_isolation_goes_back_onto_every_folder_but_not_the_leaf(tmp_path: Path) -> None:
    p = transcript_path(KEY, tmp_path)
    assert p.relative_to(tmp_path).parts == (
        "Dagshai 2002_isolation", "03 MAR - 2002_isolation",
        "DAGSHAI 26 - 29 MAR 2002 HOLI CAMP_isolation", "26 MAR - 1$ - 7 PM_isolation",
        "02 OM GURUVE NAMAH.raw.json",
    )


def test_cleaned_path_differs_only_in_the_infix(tmp_path: Path) -> None:
    raw, clean = transcript_path(KEY, tmp_path), cleaned_path(KEY, tmp_path)
    assert raw.parent == clean.parent
    assert (raw.name, clean.name) == ("02 OM GURUVE NAMAH.raw.json",
                                      "02 OM GURUVE NAMAH.cleaned.json")


def test_a_flat_key_resolves_directly_under_the_root(tmp_path: Path) -> None:
    p = transcript_path("03 AA HIMMAT KA EK KADAM.json", tmp_path)
    assert p == tmp_path / "03 AA HIMMAT KA EK KADAM.raw.json"


# --- audio_path ------------------------------------------------------------


def _mk(root: Path, rel: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"RIFF")
    return p


def test_the_wav_is_read_from_the_transcript_not_guessed(tmp_path: Path) -> None:
    """`audio_file` is the path the transcriber actually opened. Trust it."""
    want = _mk(tmp_path, "kim.ckpt/Dagshai 2002_isolation/x_model-1_kim.wav")
    audio_file = r"D:\GuruAudio\Output\kim.ckpt\Dagshai 2002_isolation\x_model-1_kim.wav"
    assert audio_path(audio_file, "Dagshai 2002/x.json", tmp_path) == want


def test_the_ckpt_folder_stays_in_the_path_so_a_second_model_is_reachable(
    tmp_path: Path,
) -> None:
    """The degenerate tracks were transcribed from `melband_roformer_big_beta4`,
    not `kim_ft`. Rooting at one model's `.ckpt` folder strands their audio while
    the files sit right there — measured, not hypothetical."""
    want = _mk(tmp_path, "big_beta4.ckpt/a_isolation/x_model-2_big.wav")
    audio_file = r"D:\GuruAudio\Output\big_beta4.ckpt\a_isolation\x_model-2_big.wav"
    assert audio_path(audio_file, "a/x.json", tmp_path) == want
    # ...and the glob fallback finds it too, without knowing the model folder.
    assert audio_path(None, "a/x.json", tmp_path) == want


def test_a_windows_audio_file_pointing_outside_the_root_is_refused(tmp_path: Path) -> None:
    evil = r"D:\GuruAudio\Output\x.ckpt\..\..\..\Windows\System32\config\SAM"
    assert audio_path(evil, "Dagshai 2002/x.json", tmp_path) is None


def test_an_audio_file_with_no_ckpt_component_falls_back_to_the_glob(tmp_path: Path) -> None:
    want = _mk(tmp_path, "kim.ckpt/Dagshai 2002_isolation/x_model-1_kim.wav")
    assert audio_path("/somewhere/else/x.wav", "Dagshai 2002/x.json", tmp_path) == want


def test_an_audio_file_naming_a_deleted_wav_falls_back_to_the_glob(tmp_path: Path) -> None:
    """A WAV re-isolated into another model tree must not strand its transcript."""
    want = _mk(tmp_path, "big.ckpt/a_isolation/x_model-2_big.wav")
    gone = r"D:\GuruAudio\Output\kim.ckpt\a_isolation\x_model-1_kim.wav"
    assert audio_path(gone, "a/x.json", tmp_path) == want


def test_the_glob_does_not_hardcode_the_model_number(tmp_path: Path) -> None:
    """Enabling a second isolation model must not silently 404 every track."""
    _mk(tmp_path, "big.ckpt/a_isolation/x_model-2_melband_roformer_big_beta4.wav")
    got = audio_path(None, "a/x.json", tmp_path)
    assert got is not None and got.name.startswith("x_model-2_")


def test_a_stem_with_glob_metacharacters_is_matched_literally(tmp_path: Path) -> None:
    """Track titles contain '[', '(' and '$'. fnmatch would eat them."""
    want = _mk(tmp_path, "kim.ckpt/a_isolation/1 MEDITATION [ONLY VOICE]_model-1_x.wav")
    _mk(tmp_path, "kim.ckpt/a_isolation/1 MEDITATION O_model-1_x.wav")
    assert audio_path(None, "a/1 MEDITATION [ONLY VOICE].json", tmp_path) == want


def test_a_flat_key_globs_across_every_model_tree(tmp_path: Path) -> None:
    """The 3 degenerate keys have no folders; their WAVs sit at a model root."""
    want = _mk(tmp_path, "kim.ckpt/03 PRAVACHAN IN MEDITATION_model-1_kim.wav")
    assert audio_path(None, "03 PRAVACHAN IN MEDITATION.json", tmp_path) == want


def test_a_missing_wav_is_none_not_an_exception(tmp_path: Path) -> None:
    assert audio_path(None, "a/b.json", tmp_path) is None
    assert audio_path(None, KEY, tmp_path) is None


# --- ticket ----------------------------------------------------------------


def test_a_fresh_ticket_verifies_for_its_own_track() -> None:
    t = mint_ticket(KEY, now=1000.0, secret=SECRET)
    assert check_ticket(KEY, t, now=1000.0, secret=SECRET)


def test_a_ticket_does_not_travel_to_another_track() -> None:
    """Otherwise one authorized play unlocks the whole 5 TB archive."""
    t = mint_ticket(KEY, now=1000.0, secret=SECRET)
    assert not check_ticket("Live Masters 2010/x.json", t, now=1000.0, secret=SECRET)


def test_a_ticket_expires() -> None:
    t = mint_ticket(KEY, now=1000.0, secret=SECRET)
    assert check_ticket(KEY, t, now=1299.0, secret=SECRET)
    assert not check_ticket(KEY, t, now=1301.0, secret=SECRET)


def test_a_ticket_from_another_process_is_worthless() -> None:
    t = mint_ticket(KEY, now=1000.0, secret=b"b" * 32)
    assert not check_ticket(KEY, t, now=1000.0, secret=SECRET)


def test_a_forged_expiry_does_not_extend_the_ticket() -> None:
    t = mint_ticket(KEY, now=1000.0, secret=SECRET)
    _, _, sig = t.partition(".")
    assert not check_ticket(KEY, f"999999999.{sig}", now=1000.0, secret=SECRET)


@pytest.mark.parametrize("junk", ["", ".", "abc", "1000.", ".deadbeef", "1000.zz"])
def test_malformed_tickets_are_rejected_without_raising(junk: str) -> None:
    assert not check_ticket(KEY, junk, now=1000.0, secret=SECRET)


# --- slim / read_json ------------------------------------------------------


def test_slim_keeps_timings_and_drops_the_word_array() -> None:
    raw = {
        "duration": 1064.802, "language": "hi", "model": "whisperx-large-v3",
        "segments": [
            {"id": 0, "start": 8.333, "end": 36.565, "text": " ओम् गुरुवे नमः ",
             "words": [{"word": "ओम्", "start": 8.333, "end": 13.8, "score": 0.0}]},
            {"id": 1, "start": 36.6, "end": 40.0, "text": "   "},
        ],
    }
    out = slim(raw)
    assert out["duration"] == 1064.802 and out["language"] == "hi"
    assert out["segments"] == [{"start": 8.333, "end": 36.565, "text": "ओम् गुरुवे नमः"}]
    assert "words" not in out["segments"][0]


def test_a_segment_without_usable_timings_is_skipped_not_fatal() -> None:
    out = slim({"segments": [{"id": 0, "text": "hi"},
                             {"id": 1, "start": 1, "end": 2, "text": "ok"}]})
    assert [s["text"] for s in out["segments"]] == ["ok"]


def test_an_empty_transcript_slims_to_no_segments() -> None:
    assert slim({})["segments"] == []


def test_a_truncated_transcript_reads_as_none_not_a_crash(tmp_path: Path) -> None:
    """`transcribe_whisperx` writes .raw.json without .tmp + os.replace, so a
    killed run really does leave these behind."""
    p = tmp_path / "t.raw.json"
    p.write_text('{"segments": [{"start": 1,', encoding="utf-8")
    assert read_json(p) is None
    assert read_json(tmp_path / "absent.raw.json") is None


def test_a_json_array_is_not_a_transcript(tmp_path: Path) -> None:
    p = tmp_path / "t.raw.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    assert read_json(p) is None


def test_an_empty_dir_is_not_a_mounted_tree(tmp_path: Path) -> None:
    assert not tree_is_mounted(tmp_path)
    assert not tree_is_mounted(tmp_path / "nope")
    (tmp_path / "x").mkdir()
    assert tree_is_mounted(tmp_path)


# --- parity: the guard that keeps the inverse honest ------------------------
# Selected with `-k parity`. No custom marker: it would need pytest config for
# one nicety, and these skip themselves off the ingest box anyway.


def _transcript_root() -> Path | None:
    """`.resolve()` matters: `rag_api.config` loads `.env` on import, and its
    `TRANSCRIPTS_DIR` is relative to the compose file. `transcript_path` resolves
    its result, so an unresolved root makes every comparison below fail."""
    for c in (os.environ.get("TRANSCRIPTS_DIR"),
              "/mnt/d/Transcription whisperx/Output",
              "D:/Transcription whisperx/Output"):
        if c and Path(c).is_dir():
            return Path(c).resolve()
    return None


def test_the_inverse_round_trips_every_indexed_key_on_disk() -> None:
    """`transcript_path(qualified_source(p)) == p`, for all 9,335.

    `qualified_source` is lossy — it deletes folders named `turbo` or containing
    `_model-`. `transcript_path` cannot restore those. This asserts the ingested
    tree contains none, which is the only reason the track panel can find a file
    at all. It is a property of the *filesystem*, not of the code, so it is
    checked against the filesystem.
    """
    root = _transcript_root()
    if root is None:
        pytest.skip("transcript tree not mounted on this host")

    from ingestion.chunker_cleaned import qualified_source

    checked = 0
    mismatches: list[tuple[str, str]] = []
    for p in root.rglob("*.raw.json"):
        key = qualified_source(p, root)
        try:
            back = transcript_path(key, root)
        except ValueError as e:
            mismatches.append((key, f"unbuildable: {e}"))
            continue
        if back != p:
            mismatches.append((key, str(back)))
        checked += 1
        if len(mismatches) > 5:
            break

    assert checked > 0, f"no transcripts under {root}"
    assert not mismatches, (
        f"{len(mismatches)} of {checked} keys do not invert. The track panel "
        f"will 404 on these. First: {mismatches[0]}"
    )


_AUDIO_FILE_RE = re.compile(r'"audio_file"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _audio_file_of(p: Path) -> str | None:
    """Pull `audio_file` out of a raw transcript without parsing 118 KB of it.

    It is the first key `write_raw_outputs` emits, so the head of the file is
    enough. Reading all 9,335 in full would be ~1 GB and the test would get
    quietly capped to a sample — which reads as coverage it does not have.
    """
    with p.open("r", encoding="utf-8") as fh:
        head = fh.read(2048)
    m = _AUDIO_FILE_RE.search(head)
    return json.loads(f'"{m.group(1)}"') if m else None


def test_every_indexed_key_that_claims_audio_can_find_it() -> None:
    """`audio_file` inside every transcript must resolve under the isolated root.

    The other half of the inverse: a key can reach its transcript and that
    transcript can still name a WAV we cannot serve. It found a real one —
    the tracks `transcribe_local.py` handled were isolated by the *second*
    model, so the root must be `GuruAudio/Output`, not one `.ckpt` folder.

    All 9,335. No sampling.
    """
    root = _transcript_root()
    iso = None
    for c in (os.environ.get("ISOLATED_DIR"),
              "/mnt/d/GuruAudio/Output", "D:/GuruAudio/Output"):
        if c and Path(c).is_dir():
            iso = Path(c).resolve()
            break
    if root is None or iso is None:
        pytest.skip("media trees not mounted on this host")

    from ingestion.chunker_cleaned import qualified_source

    checked = 0
    missing: list[str] = []
    for p in root.rglob("*.raw.json"):
        checked += 1
        key = qualified_source(p, root)
        if audio_path(_audio_file_of(p), key, iso) is None:
            missing.append(key)

    assert checked > 0, f"no transcripts under {root}"
    assert not missing, (
        f"{len(missing)} of {checked} indexed tracks have no reachable isolated "
        f"audio. The panel will offer no player for them. First 3: {missing[:3]}"
    )
