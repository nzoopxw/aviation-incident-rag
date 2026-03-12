# Aviation Incident Report Pattern Finder
RAG system over NASA's ASRS dataset with semantic clustering and pattern detection.

## Stack
- Embeddings: BAAI/bge-large-en-v1.5
- Vector DB: Qdrant
- Retrieval: Hybrid (BM25 + dense + cross-encoder reranking)
- Pattern Detection: HDBSCAN + KeyBERT + co-occurrence graphs
- LLM: Llama 3.2 3B via Ollama
- Dataset: NASA ASRS (15,462 reports, 2022-2024)
