import json
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# ── Config ─────────────────────────────────────────────────────────────────
CHUNKS_PATH = Path("data/processed/chunks.json")
COLLECTION_NAME = "aviation_incidents"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # better than MiniLM for technical text
BATCH_SIZE = 64  # how many chunks to embed at once

# ── Load chunks ────────────────────────────────────────────────────────────
print("Loading chunks...")
with open(CHUNKS_PATH) as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks")

# ── Load embedding model ───────────────────────────────────────────────────
print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
print("(First run will download ~1.3GB — this is normal)")
model = SentenceTransformer(EMBEDDING_MODEL)
VECTOR_SIZE = model.get_sentence_embedding_dimension()
print(f"Embedding dimension: {VECTOR_SIZE}")

# ── Connect to Qdrant ──────────────────────────────────────────────────────
print("\nConnecting to Qdrant...")
client = QdrantClient(host="localhost", port=6333)

# Create collection (delete first if it already exists)
existing = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME in existing:
    client.delete_collection(COLLECTION_NAME)
    print(f"Deleted existing collection: {COLLECTION_NAME}")

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
)
print(f"Created collection: {COLLECTION_NAME}")

# ── Embed and upload in batches ────────────────────────────────────────────
print(f"\nEmbedding {len(chunks)} chunks in batches of {BATCH_SIZE}...")
start = time.time()

for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]
    texts = [c["text"] for c in batch]

    # Embed
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    # Upload to Qdrant
    points = [
        PointStruct(
            id=i + j,
            vector=embeddings[j].tolist(),
            payload={
                "chunk_id": batch[j]["chunk_id"],
                "acn": batch[j]["acn"],
                "date": batch[j]["date"],
                "aircraft": batch[j]["aircraft"],
                "phase_of_flight": batch[j]["phase_of_flight"],
                "primary_problem": batch[j]["primary_problem"],
                "contributing_factors": batch[j]["contributing_factors"],
                "human_factors": batch[j]["human_factors"],
                "anomaly": batch[j]["anomaly"],
                "synopsis": batch[j]["synopsis"],
                "text": batch[j]["text"]
            }
        )
        for j in range(len(batch))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    # Progress
    done = min(i + BATCH_SIZE, len(chunks))
    elapsed = time.time() - start
    print(f"  [{done}/{len(chunks)}] {elapsed:.1f}s elapsed")

print(f"\nDone! All {len(chunks)} chunks embedded and stored in Qdrant.")
print(f"Total time: {(time.time() - start) / 60:.1f} minutes")