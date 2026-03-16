# Aviation Incident Report Pattern Finder

A research-grade Retrieval-Augmented Generation (RAG) system for analyzing NASA's Aviation Safety Reporting System (ASRS) dataset. Goes beyond basic retrieval by adding a **pattern detection layer** that clusters retrieved incidents, extracts topics, and surfaces systemic safety relationships — the way a real safety analyst would work.

Built as a portfolio project targeting aviation safety AI applications.

---

## The Problem

NASA's ASRS database contains 1.5M+ aviation incident reports. Current search is keyword-only. Safety analysts manually sift through hundreds of reports to find patterns — which failure modes cluster together, which contributing factors co-occur, whether a pattern is trending.

This system automates that pattern analysis on top of semantic retrieval.

---

## Architecture
```
query → hybrid retrieval (BM25 + dense + reranker) → pattern detection layer → LLM answer
                                                            ↓
                                                   HDBSCAN clustering
                                                   KeyBERT topic labeling
                                                   Co-occurrence graph
                                                   Temporal trend analysis
```

---

## Key Technical Decisions

**Why hybrid retrieval?**
Aviation reports use precise technical terminology (TCAS RA, FCMC, hydraulic actuator). Pure dense retrieval misses exact acronym matches. BM25 catches exact terms; dense search catches meaning. Cross-encoder reranking then selects the best 10 from the combined candidate set.

**Why HDBSCAN over K-Means?**
Incident patterns don't have uniform cluster sizes — some failure modes appear in hundreds of reports, others in two. HDBSCAN handles variable-density clusters and doesn't require specifying k in advance.

**Why paragraph-level chunking?**
Each paragraph in a pilot narrative describes a distinct moment in the incident timeline. Paragraph chunking preserves semantic coherence better than arbitrary character-count splitting.

**Why BAAI/bge-large-en-v1.5?**
Top-ranked on MTEB retrieval benchmarks at time of development. 1024-dim embeddings capture more nuance than MiniLM (384-dim) for technical domain text.

---

## Stack

| Component | Tool |
|---|---|
| Embeddings | BAAI/bge-large-en-v1.5 |
| Vector DB | Qdrant (Docker) |
| Retrieval | BM25 (rank-bm25) + dense + CrossEncoder reranking |
| Clustering | HDBSCAN |
| Topic Extraction | KeyBERT |
| Co-occurrence | NetworkX |
| LLM | Llama 3.2 3B via Ollama |
| Orchestration | LangChain |
| UI | Streamlit + Plotly |

---

## Dataset

- **Source:** NASA Aviation Safety Reporting System (ASRS)
- **Size:** 15,462 reports → 15,445 chunks
- **Date range:** January 2022 – December 2024
- **Fields used:** ACN, date, aircraft type, phase of flight, narrative, synopsis, anomaly, contributing factors, human factors

---

## Quantization Benchmark

Benchmarked Llama 3.2 3B at Q8 vs Q4 quantization across 10 domain-specific queries:

| Model | Avg Latency | Avg Words |
|---|---|---|
| Q8 (default) | 11.22s | 351 |
| Q4 (quantized) | 11.18s | 362 |

On Apple Silicon (M-series iMac), quantization produced negligible latency improvement (~0.4%), contrasting with reported GPU speedups in literature. This suggests quantization benefits are architecture-dependent — a relevant finding for edge deployment decisions.

---

## Setup
```bash
# 1. Clone and install
git clone https://github.com/nzoopxw/aviation-incident-rag
cd aviation-incident-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Start Qdrant
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

# 3. Download and embed data
# Download ASRS CSVs from https://asrs.arc.nasa.gov/search/database.html
# Place in data/raw/, then:
python3 src/preprocess.py
python3 src/embeddings.py

# 4. Start Ollama and pull model
ollama pull llama3.2:3b

# 5. Run the app
python3 -m streamlit run app.py
```

---

## Limitations & Future Work

- **Domain-specific embeddings:** Fine-tuning the embedding model on ASRS data would improve retrieval quality for aviation-specific terminology
- **Larger dataset:** Expanding to the full ASRS corpus (1.5M+ reports) would reveal longer-term trends
- **Evaluation:** Formal retrieval evaluation (MRR@5, NDCG) on a manually labeled test set
- **Real-time updates:** Pipeline to ingest new ASRS reports as they are published

---

## Author

Navya — B.Tech AI & Data Science, Shiv Nadar Chennai