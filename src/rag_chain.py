import json
import requests
from src.hybrid_retriever import retrieve
from src.pattern_detector import detect_patterns

# ── LLM via Ollama ─────────────────────────────────────────────────────────
def call_llm(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2:3b", "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an aviation safety analyst reviewing NASA ASRS incident reports.

Given retrieved incident reports and detected patterns, provide:
1. A direct answer to the query
2. The dominant pattern(s) you observe across incidents
3. One non-obvious insight the user may not have considered
4. Two recommended follow-up queries

Always cite specific ACN report numbers when making claims.
Be concise and analytical — you are writing for safety professionals."""

# ── Main RAG function ──────────────────────────────────────────────────────
def query_rag(user_query: str, filters: dict = None) -> dict:
    print(f"\nQuery: {user_query}")
    print("Retrieving relevant incidents...")

    # Step 1: Retrieve
    chunks = retrieve(user_query, top_k=20, filters=filters)
    print(f"Retrieved {len(chunks)} chunks")

    # Step 2: Detect patterns
    print("Detecting patterns...")
    patterns = detect_patterns(chunks)

    # Step 3: Build context for LLM
    context = "\n\n".join([
        f"ACN {c['acn']} ({c['date']}) | {c['aircraft']} | {c['phase_of_flight']}\n{c['text']}"
        for c in chunks[:8]  # top 8 for context window
    ])

    pattern_summary = f"""
Temporal distribution: {patterns.get('temporal', {})}
Co-occurring factors: {patterns.get('co_occurrences', [])[:3]}
Clusters found: {len(patterns.get('clusters', {}))}
Cluster topics: {[v['keywords'][:3] for v in patterns.get('clusters', {}).values()]}
"""

    # Step 4: Call LLM
    print("Generating answer...")
    prompt = f"""{SYSTEM_PROMPT}

USER QUERY: {user_query}

RETRIEVED INCIDENTS:
{context}

DETECTED PATTERNS:
{pattern_summary}

Provide your analysis:"""

    answer = call_llm(prompt)

    return {
        "query": user_query,
        "answer": answer,
        "patterns": patterns,
        "retrieved_chunks": chunks
    }


# ── Test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = query_rag("hydraulic failure during landing")

    print("\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(result["answer"])

    print("\n" + "="*60)
    print("PATTERNS:")
    print("="*60)
    print(f"Temporal: {result['patterns']['temporal']}")
    print(f"Top co-occurrences: {result['patterns']['co_occurrences'][:3]}")
    print(f"Clusters: {len(result['patterns']['clusters'])}")