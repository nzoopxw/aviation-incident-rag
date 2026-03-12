import json
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# ── Load everything ────────────────────────────────────────────────────────
COLLECTION_NAME = "aviation_incidents"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("Loading reranker...")
reranker = CrossEncoder(RERANKER_MODEL)

print("Connecting to Qdrant...")
client = QdrantClient(host="localhost", port=6333)

print("Loading chunks for BM25...")
with open(Path("data/processed/chunks.json")) as f:
    chunks = json.load(f)

# Build BM25 index over all chunk texts
tokenized = [c["text"].lower().split() for c in chunks]
bm25 = BM25Okapi(tokenized)
print(f"BM25 index built over {len(chunks)} chunks")

# ── Core retrieval function ────────────────────────────────────────────────
def retrieve(query: str, top_k: int = 20, filters: dict = None) -> list[dict]:
    """
    Hybrid retrieval: dense + BM25, reranked by cross-encoder.
    
    filters: optional dict with keys like:
        {"phase_of_flight": "Landing", "aircraft": "B737"}
    """

    # ── 1. Dense retrieval from Qdrant ─────────────────────────────────────
    query_vector = embedder.encode(query, normalize_embeddings=True).tolist()
    
    qdrant_filter = None
    if filters:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        qdrant_filter = Filter(must=conditions)

    dense_results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=top_k,
    query_filter=qdrant_filter
    )
    dense_hits = {r.payload["chunk_id"]: r.payload for r in dense_results.points}
    # ── 2. BM25 retrieval ──────────────────────────────────────────────────
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_hits = {chunks[i]["chunk_id"]: chunks[i] for i in top_bm25_indices}

    # ── 3. Combine (union of both result sets) ─────────────────────────────
    combined = {**bm25_hits, **dense_hits}  # dense overwrites if duplicate
    candidates = list(combined.values())

    # ── 4. Rerank with cross-encoder ───────────────────────────────────────
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top_results = [doc for _, doc in ranked[:10]]  # return top 10 after reranking

    return top_results


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Test Query ---")
    results = retrieve("hydraulic failure during landing")
    
    for i, r in enumerate(results):
        print(f"\n[{i+1}] ACN: {r['acn']} | {r['date']} | {r['aircraft']} | {r['phase_of_flight']}")
        print(f"     {r['text'][:200]}...")