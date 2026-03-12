import json
import numpy as np
from pathlib import Path
from collections import Counter
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import hdbscan
import networkx as nx

# ── Load models ────────────────────────────────────────────────────────────
print("Loading models...")
embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
kw_model = KeyBERT(model=embedder)

def detect_patterns(retrieved_chunks: list[dict]) -> dict:
    """
    Given a list of retrieved chunks, detect patterns across them.
    Returns a dict with clusters, topics, temporal trends, and co-occurrences.
    """

    if len(retrieved_chunks) < 3:
        return {"error": "Not enough chunks to detect patterns"}

    texts = [c["text"] for c in retrieved_chunks]

    # ── 1. Embed retrieved chunks ──────────────────────────────────────────
    embeddings = embedder.encode(texts, normalize_embeddings=True)

    # ── 2. HDBSCAN clustering ──────────────────────────────────────────────
    # min_cluster_size=2 because retrieved set is small (10-20 chunks)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean")
    labels = clusterer.fit_predict(embeddings)

    # Group chunks by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue  # -1 means noise/outlier in HDBSCAN
        clusters.setdefault(label, []).append(retrieved_chunks[i])

    print(f"Found {len(clusters)} clusters ({sum(labels == -1)} outliers)")

    # ── 3. KeyBERT topic labeling per cluster ──────────────────────────────
    cluster_topics = {}
    for label, chunk_group in clusters.items():
        combined_text = " ".join([c["text"] for c in chunk_group])
        keywords = kw_model.extract_keywords(
            combined_text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=5
        )
        cluster_topics[label] = {
            "keywords": [kw for kw, score in keywords],
            "size": len(chunk_group),
            "chunks": chunk_group
        }

    # ── 4. Temporal analysis ───────────────────────────────────────────────
    # Count incidents per year across retrieved chunks
    year_counts = Counter()
    for c in retrieved_chunks:
        date = str(c.get("date", ""))
        if len(date) >= 4:
            year = date[:4]
            year_counts[year] += 1

    temporal = dict(sorted(year_counts.items()))

    # ── 5. Co-occurrence analysis ──────────────────────────────────────────
    # Which contributing factors appear together across retrieved chunks?
    G = nx.Graph()
    for c in retrieved_chunks:
        factors = str(c.get("contributing_factors", ""))
        if factors and factors != "Unknown":
            # Split by semicolon or comma
            items = [f.strip() for f in factors.replace(";", ",").split(",") if f.strip()]
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    if G.has_edge(items[i], items[j]):
                        G[items[i]][items[j]]["weight"] += 1
                    else:
                        G.add_edge(items[i], items[j], weight=1)

    # Get top co-occurring factor pairs by weight
    edges = sorted(G.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)
    top_cooccurrences = [(u, v, d["weight"]) for u, v, d in edges[:10]]

    return {
        "clusters": cluster_topics,
        "temporal": temporal,
        "co_occurrences": top_cooccurrences,
        "total_retrieved": len(retrieved_chunks),
        "total_clustered": sum(len(v["chunks"]) for v in cluster_topics.values())
    }


# ── Test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load some chunks to test with
    with open(Path("data/processed/chunks.json")) as f:
        all_chunks = json.load(f)

    # Sample evenly across the dataset for more variety
    indices = list(range(0, len(all_chunks), len(all_chunks) // 15))[:15]
    test_chunks = [all_chunks[i] for i in indices]

    print("\nRunning pattern detection on 15 test chunks...")
    patterns = detect_patterns(test_chunks)

    print(f"\nTemporal distribution: {patterns['temporal']}")
    print(f"\nTop co-occurrences:")
    for u, v, w in patterns["co_occurrences"][:5]:
        print(f"  '{u}' + '{v}' → {w} times")

    print(f"\nClusters found: {len(patterns['clusters'])}")
    for label, info in patterns["clusters"].items():
        print(f"\n  Cluster {label} ({info['size']} chunks)")
        print(f"  Topics: {info['keywords']}")