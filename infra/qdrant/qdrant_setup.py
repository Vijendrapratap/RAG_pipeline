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

    # Idempotent payload indexes
    for field, schema in [
        ("source_file", PayloadSchemaType.KEYWORD),
        ("speakers",    PayloadSchemaType.KEYWORD),
        ("start_sec",   PayloadSchemaType.FLOAT),
        ("date",        PayloadSchemaType.DATETIME),
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
