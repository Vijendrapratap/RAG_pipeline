"""Corpus tree aggregation. Pure functions, no Postgres.

The shapes below are real: Dagshai buries a track four folders deep behind a
month bucket, Live Masters buries it three, and `transcribe_local.py` left three
tracks with no folder at all. Any code that infers a node's *level* from its
depth gets one of those three wrong.
"""
from __future__ import annotations

from datetime import date

import pytest

from rag_api.corpus import build_nodes, summarize, track_state, version_key

D = date(2002, 3, 26)

ROWS = [
    # collection / month bucket / event / session / track   (Dagshai: 4 folders)
    ("Dagshai 2002/03 MAR - 2002/DAGSHAI 26 - 29 MAR 2002 HOLI CAMP/"
     "26 MAR - 1$ - 7 PM/02 OM GURUVE NAMAH.json", 3, 610.0, D),
    ("Dagshai 2002/03 MAR - 2002/DAGSHAI 26 - 29 MAR 2002 HOLI CAMP/"
     "26 MAR - 1$ - 7 PM/03 SAMBODHAN.json", 5, 900.0, D),
    # an undated sibling session in the same event
    ("Dagshai 2002/03 MAR - 2002/DAGSHAI 26 - 29 MAR 2002 HOLI CAMP/"
     "EVENING $/01 PRAVACHAN.json", 4, 700.0, None),
    # collection / event / session / track                  (Live Masters: 3)
    ("Live Masters 2010/01 NOIDA 7 - 10 JAN 2010/7 JAN - 1$ - 6 PM/"
     "04 PRAVACHAN.json", 7, 1200.0, date(2010, 1, 7)),
    # degenerate: no folder at all
    ("03 AA HIMMAT KA EK KADAM.json", 2, 300.0, None),
]


def _by_path(nodes: list[dict]) -> dict[str, dict]:
    return {n["path"]: n for n in nodes}


# --- track_state ----------------------------------------------------------


def test_a_key_with_no_folder_is_failed_however_many_chunks_it_has() -> None:
    assert track_state("03 AA HIMMAT.json", 99, D) == "failed"


def test_indexed_and_dated_is_remembered() -> None:
    assert track_state("a/b.json", 1, D) == "remembered"


@pytest.mark.parametrize("chunks,sdate", [(0, D), (3, None), (0, None)])
def test_missing_chunks_or_missing_date_is_written_not_remembered(chunks, sdate) -> None:
    assert track_state("a/b.json", chunks, sdate) == "written"


# --- build_nodes ----------------------------------------------------------


def test_root_depth_1_returns_collections_and_the_degenerate_track() -> None:
    nodes = _by_path(build_nodes(ROWS, "", 1))
    assert set(nodes) == {"Dagshai 2002", "Live Masters 2010",
                          "03 AA HIMMAT KA EK KADAM.json"}
    assert nodes["Dagshai 2002"]["n_files"] == 3
    assert nodes["Dagshai 2002"]["n_chunks"] == 12
    assert nodes["Live Masters 2010"]["n_files"] == 1


def test_a_collection_is_a_folder_and_the_degenerate_key_is_a_leaf() -> None:
    """is_leaf comes from what follows in the key, never from depth. The
    degenerate track and a whole collection sit at the same depth."""
    nodes = _by_path(build_nodes(ROWS, "", 1))
    assert nodes["Dagshai 2002"]["is_leaf"] is False
    assert nodes["03 AA HIMMAT KA EK KADAM.json"]["is_leaf"] is True
    assert nodes["03 AA HIMMAT KA EK KADAM.json"]["failed"] == 1


def test_depth_3_from_root_spans_trees_of_different_shapes() -> None:
    """Live Masters' session sits at depth 3; Dagshai's event does. Both appear,
    and neither is mistaken for the other's level."""
    nodes = _by_path(build_nodes(ROWS, "", 3))
    assert "Dagshai 2002/03 MAR - 2002/DAGSHAI 26 - 29 MAR 2002 HOLI CAMP" in nodes
    lm = nodes["Live Masters 2010/01 NOIDA 7 - 10 JAN 2010/7 JAN - 1$ - 6 PM"]
    assert lm["is_leaf"] is False and lm["depth"] == 3
    # Depth 3 must not yet expose the Dagshai tracks, which live at depth 5.
    assert not any(n["path"].endswith("02 OM GURUVE NAMAH.json") for n in nodes.values())


def test_drilling_a_session_yields_its_tracks_with_dates() -> None:
    prefix = ("Dagshai 2002/03 MAR - 2002/DAGSHAI 26 - 29 MAR 2002 HOLI CAMP/"
              "26 MAR - 1$ - 7 PM")
    nodes = build_nodes(ROWS, prefix, 1)
    assert [n["name"] for n in nodes] == ["02 OM GURUVE NAMAH.json", "03 SAMBODHAN.json"]
    assert all(n["is_leaf"] for n in nodes)
    assert all(n["session_date"] == "2002-03-26" for n in nodes)
    assert all(n["remembered"] == 1 for n in nodes)


def test_a_prefix_containing_dollar_and_underscore_is_matched_literally() -> None:
    """'26 MAR - 1$ - 7 PM' would be a LIKE metacharacter minefield. Nothing here
    reaches SQL, so it is matched as bytes."""
    prefix = ("Dagshai 2002/03 MAR - 2002/DAGSHAI 26 - 29 MAR 2002 HOLI CAMP/"
              "EVENING $")
    nodes = build_nodes(ROWS, prefix, 1)
    assert [n["name"] for n in nodes] == ["01 PRAVACHAN.json"]


def test_prefix_is_normalized_so_a_missing_slash_does_not_match_a_sibling() -> None:
    """Without the trailing '/', prefix 'Dagshai 200' would swallow 'Dagshai 2002'."""
    assert build_nodes(ROWS, "Dagshai 200", 1) == []
    assert len(build_nodes(ROWS, "Dagshai 2002", 1)) == 1


def test_an_event_aggregates_its_dated_and_undated_sessions_separately() -> None:
    """The colour of a container is a ratio, not a guess: this event holds two
    remembered tracks and one written one."""
    event = ("Dagshai 2002/03 MAR - 2002/DAGSHAI 26 - 29 MAR 2002 HOLI CAMP")
    node = _by_path(build_nodes(ROWS, "", 3))[event]
    assert (node["remembered"], node["written"], node["failed"]) == (2, 1, 0)
    assert node["n_files"] == 3
    assert node["duration_sec"] == pytest.approx(2210.0)


def test_unknown_prefix_returns_empty_not_the_whole_tree() -> None:
    assert build_nodes(ROWS, "Nonexistent 1999", 2) == []


# --- summarize ------------------------------------------------------------


def test_summarize_counts_states_and_names_the_broken_files() -> None:
    s = summarize(ROWS)
    assert s["n_files"] == 5
    assert s["n_chunks"] == 21
    assert (s["n_remembered"], s["n_written"], s["n_failed"]) == (3, 1, 1)
    assert s["degenerate_files"] == ["03 AA HIMMAT KA EK KADAM.json"]
    assert s["min_session_date"] == "2002-03-26"
    assert s["max_session_date"] == "2010-01-07"


def test_summarize_of_an_empty_corpus_does_not_crash_on_min_of_nothing() -> None:
    s = summarize([])
    assert s["n_files"] == 0 and s["min_session_date"] is None


def test_summarize_tolerates_a_null_duration() -> None:
    s = summarize([("a/b.json", 1, None, D)])
    assert s["duration_sec"] == 0.0


# --- version_key ----------------------------------------------------------


def test_version_moves_when_a_reingest_replaces_chunks_one_for_one() -> None:
    """n_files and n_chunks unchanged; only ingested_at moved. The map must
    still notice, or a re-ingest is invisible."""
    a = version_key(9335, 24567, "2026-07-08T15:00:00")
    b = version_key(9335, 24567, "2026-07-08T15:10:00")
    assert a != b


def test_version_is_stable_for_identical_inputs() -> None:
    assert version_key(1, 2, "x") == version_key(1, 2, "x")
