"""Curated-catalog ingestion (the 'Camp Record Export' spreadsheet).

The spreadsheet is a 34-year human-curated index of the same discourse corpus
the RAG pipeline transcribes. It carries ground-truth metadata the audio paths
do not (performers, release references) plus partial hand transcriptions.

This package is purely additive: it normalises the sheet into structured
records keyed identically to `ingestion.utils.path_parser`, so a catalog row
and its matching audio folder collapse onto the same `join_key`. Nothing here
mutates the running Qdrant / Postgres / Tantivy state — it produces clean
tables and a coverage report; downstream phases decide what to load.
"""
from __future__ import annotations

from .normalize import (
    CatalogTrack,
    clean_text,
    join_key_from_fields,
    join_key_from_path,
    normalize_location,
    parse_duration_to_seconds,
    parse_sitting,
    rows_to_tracks,
    split_performers,
)

__all__ = [
    "CatalogTrack",
    "clean_text",
    "join_key_from_fields",
    "join_key_from_path",
    "normalize_location",
    "parse_duration_to_seconds",
    "parse_sitting",
    "rows_to_tracks",
    "split_performers",
]
