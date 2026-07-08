"""The backfill's safety argument, pinned.

`backfill_path_meta` writes corrected dates straight into file_meta and Qdrant
without re-chunking. That is only sound because a `source_file` identity key
parses to exactly the metadata the chunker derived from the full disk path.
If `qualified_source()` ever drops a level, or `parse_path()` ever grows a
dependency on the base_dir, these tests fail before 9,335 rows get bad dates.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.backfill_path_meta import QDRANT_FIELDS, meta_for_key
from ingestion.chunker_cleaned import qualified_source
from ingestion.utils.path_parser import parse_path

BASE = Path("/mnt/d/Transcription whisperx/Output")

# Verbatim shapes from the real archive: the 4-level Dagshai tree with a month
# bucket, the 3-level Live Masters tree, a cross-year camp, and the degenerate
# flat key transcribe_local.py left behind.
REAL_PATHS = [
    "Dagshai 2002_isolation/03 MAR - 2002/DAGSHAI 26 - 29 MAR 2002 HOLI CAMP/"
    "26 MAR - 1$ - 7 PM/02 OM GURUVE NAMAH.raw.json",
    "Dagshai 2001_isolation/01 JAN - 2001/10 JAN - 6 PM/01 OM GURUVE NAMAH.raw.json",
    "Live Masters 2010_isolation/01 NOIDA 7 - 10 JAN 2010/7 JAN - 1$ - 6 PM/"
    "04 PRAVACHAN.raw.json",
    "Live Masters 2016_isolation/05 CHHATTARPUR DELHI 30 DEC 2016 - 1 JAN 2017/"
    "31 DEC - 3$ - 615 PM/06 JAG MEIN DO DIN KA.raw.json",
    "03 AA HIMMAT KA EK KADAM.raw.json",
]


def _truth(raw_path: Path):
    """What chunker_cleaned.process_file computes at ingest time: parse_path of
    the disk path, with the '.raw' infix normalized off the track name."""
    key = qualified_source(raw_path, BASE)
    return parse_path(raw_path.with_name(key.rsplit("/", 1)[-1]), base_dir=BASE)


@pytest.mark.parametrize("rel", REAL_PATHS)
def test_source_file_parses_identically_to_the_disk_path(rel: str) -> None:
    raw_path = BASE / rel
    key = qualified_source(raw_path, BASE)
    truth, from_key = _truth(raw_path), meta_for_key(key)
    assert truth.to_payload() == from_key.to_payload(), (
        f"{key!r} does not round-trip; the backfill would write wrong metadata"
    )


def test_the_round_trip_actually_recovers_a_date() -> None:
    """A vacuous round-trip (both sides None) would pass the test above."""
    key = qualified_source(BASE / REAL_PATHS[0], BASE)
    assert meta_for_key(key).session_date is not None


def test_cross_year_camp_dates_from_the_span_not_the_collection_year() -> None:
    """'30 DEC 2016 - 1 JAN 2017', session '31 DEC' -> 2016, not 2017."""
    meta = meta_for_key(qualified_source(BASE / REAL_PATHS[3], BASE))
    assert meta.session_date is not None
    assert (meta.session_date.year, meta.session_date.month) == (2016, 12)


def test_degenerate_key_yields_no_date_rather_than_a_guess() -> None:
    meta = meta_for_key(qualified_source(BASE / REAL_PATHS[4], BASE))
    assert meta.session_date is None
    assert meta.event_id is None


def test_qdrant_fields_match_the_ingest_writers_field_list() -> None:
    """A field indexed for filtering but absent from QDRANT_FIELDS would keep
    answering date queries from the old parser's payload forever."""
    from ingestion.bulk_ingest_hardened import _PATH_FIELDS

    assert set(QDRANT_FIELDS) == set(_PATH_FIELDS)
