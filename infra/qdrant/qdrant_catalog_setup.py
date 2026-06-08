"""Create the standalone Qdrant 'catalog' collection. Idempotent.

This is a SEPARATE data source from the main 'transcripts' collection — the
curated 'Camp Record Export' spreadsheet indexed as its own searchable
documents (per-sitting hand transcriptions + canonical track titles). It uses
the same bge-m3 vector geometry (1024-dim COSINE) so the same query embedding
searches it, and the same reranker scores it.

Keeping it separate means: zero risk to the running 'transcripts' index, a
clean Whisper-vs-catalog accuracy comparison, and trivial reversibility (drop
the collection and nothing else changes). The only cost is result-level
duplication, deduped later by join_key if desired.

Run:  python -m infra.qdrant.qdrant_catalog_setup
"""
import os

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, HnswConfigDiff, PayloadSchemaType, ScalarQuantization,
    ScalarQuantizationConfig, ScalarType, VectorParams,
)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION = os.environ.get("QDRANT_CATALOG_COLLECTION", "catalog")


def main() -> None:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY or None)
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION in existing:
        print(f"Collection {COLLECTION!r} already exists — skipping creation.")
    else:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=True)
            ),
            hnsw_config=HnswConfigDiff(m=32, ef_construct=256),
            on_disk_payload=True,
        )
        print(f"✅ Created collection {COLLECTION!r}")

    for field, schema in [
        ("source_type",  PayloadSchemaType.KEYWORD),   # always 'catalog'
        ("doc_type",     PayloadSchemaType.KEYWORD),   # sitting_detail | track_title
        ("join_key",     PayloadSchemaType.KEYWORD),
        ("sitting_key",  PayloadSchemaType.KEYWORD),
        ("location",     PayloadSchemaType.KEYWORD),
        ("session_date", PayloadSchemaType.DATETIME),
        ("season",       PayloadSchemaType.KEYWORD),
        ("track_type",   PayloadSchemaType.KEYWORD),
        ("performers",   PayloadSchemaType.KEYWORD),
        ("track_no",     PayloadSchemaType.INTEGER),
    ]:
        try:
            client.create_payload_index(COLLECTION, field, schema)
            print(f"✅ Payload index: {field}")
        except Exception as e:  # noqa: BLE001 — idempotency guard, re-raise real errors
            if "already exists" in str(e).lower():
                print(f"⏭️  Payload index {field} already exists")
            else:
                raise


if __name__ == "__main__":
    main()
