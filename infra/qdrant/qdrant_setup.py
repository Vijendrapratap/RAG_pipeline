"""Create the Qdrant 'transcripts' collection. Idempotent — safe to re-run."""
import os
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, ScalarQuantization, ScalarQuantizationConfig,
    ScalarType, HnswConfigDiff, PayloadSchemaType,
)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "transcripts")

def main():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY or None)
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION in existing:
        print(f"Collection {COLLECTION!r} already exists — skipping creation.")
    else:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8, always_ram=True,
                )
            ),
            hnsw_config=HnswConfigDiff(m=32, ef_construct=256),
            on_disk_payload=True,
        )
        print(f"✅ Created collection {COLLECTION!r}")

    # Idempotent payload indexes.
    # Phase 6 fields: source_file, speakers, start_sec, date.
    # Phase 12 adds: session_date, track_type, location, event_id, season,
    # year, primary_speaker — to support the filter-aware retrieval queries
    # the LLM will issue ("monsoon day", "Noida camp", discourses-only, etc.).
    for field, schema in [
        # --- Phase 6 ---
        ("source_file",     PayloadSchemaType.KEYWORD),
        ("speakers",        PayloadSchemaType.KEYWORD),
        ("start_sec",       PayloadSchemaType.FLOAT),
        ("date",            PayloadSchemaType.DATETIME),
        # --- Phase 12 ---
        ("session_date",    PayloadSchemaType.DATETIME),
        ("track_type",      PayloadSchemaType.KEYWORD),
        ("location",        PayloadSchemaType.KEYWORD),
        ("event_id",        PayloadSchemaType.KEYWORD),
        ("season",          PayloadSchemaType.KEYWORD),
        ("year",            PayloadSchemaType.INTEGER),
        ("primary_speaker", PayloadSchemaType.KEYWORD),
        # --- Phase 13 (content tags propagated from file_meta) ---
        ("event_type",            PayloadSchemaType.KEYWORD),
        ("primary_language",      PayloadSchemaType.KEYWORD),
        ("topics",                PayloadSchemaType.KEYWORD),
        ("people_named",          PayloadSchemaType.KEYWORD),
        ("places_named",          PayloadSchemaType.KEYWORD),
        ("scriptures_referenced", PayloadSchemaType.KEYWORD),
        # --- Phase 14 (catalog enrichment) ---
        # performers: the new filter facet. session_seq / track_no: indexed so
        # the backfill's date+seq+track set_payload filter is cheap.
        ("performers",      PayloadSchemaType.KEYWORD),
        ("session_seq",     PayloadSchemaType.INTEGER),
        ("track_no",        PayloadSchemaType.INTEGER),
    ]:
        try:
            client.create_payload_index(COLLECTION, field, schema)
            print(f"✅ Payload index: {field}")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"⏭️  Payload index {field} already exists")
            else:
                raise

if __name__ == "__main__":
    main()
