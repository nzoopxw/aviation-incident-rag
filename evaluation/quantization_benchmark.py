import time
import requests
import json
from pathlib import Path

# ── Test queries ───────────────────────────────────────────────────────────
TEST_QUERIES = [
    "What are the most common causes of hydraulic failure during landing?",
    "How do pilots typically respond to TCAS resolution advisories?",
    "What role does crew fatigue play in approach and landing incidents?",
    "Describe common ATC communication breakdowns during busy traffic periods.",
    "What maintenance issues most frequently ground commercial aircraft?",
    "How do weather conditions contribute to runway incursion incidents?",
    "What are the most reported human factors in aviation incidents?",
    "Describe incidents involving unstabilized approaches.",
    "What equipment failures most commonly occur during takeoff?",
    "How do pilots handle engine failure shortly after takeoff?"
]

MODELS = {
    "Q8 (default)": "llama3.2:3b",
    "Q4 (quantized)": "llama3.2:3b-instruct-q4_K_M"
}

def call_ollama(model: str, prompt: str) -> tuple[str, float]:
    start = time.time()
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    latency = time.time() - start
    answer = response.json()["response"]
    return answer, latency

# ── Run benchmark ──────────────────────────────────────────────────────────
results = {}

for model_label, model_name in MODELS.items():
    print(f"\nBenchmarking {model_label} ({model_name})")
    print("-" * 50)
    
    model_results = []
    total_latency = 0
    
    for i, query in enumerate(TEST_QUERIES):
        print(f"  Query {i+1}/{len(TEST_QUERIES)}: {query[:50]}...")
        answer, latency = call_ollama(model_name, query)
        total_latency += latency
        
        model_results.append({
            "query": query,
            "answer": answer,
            "latency": round(latency, 2),
            "answer_length": len(answer.split())
        })
        print(f"  Latency: {latency:.2f}s | Words: {len(answer.split())}")
    
    avg_latency = total_latency / len(TEST_QUERIES)
    results[model_label] = {
        "model": model_name,
        "avg_latency": round(avg_latency, 2),
        "results": model_results
    }
    print(f"\n  Average latency: {avg_latency:.2f}s")

# ── Print comparison table ─────────────────────────────────────────────────
print("\n" + "="*60)
print("QUANTIZATION BENCHMARK RESULTS")
print("="*60)
print(f"{'Model':<20} {'Avg Latency':>12} {'Avg Words':>10}")
print("-"*45)

for label, data in results.items():
    avg_words = sum(r["answer_length"] for r in data["results"]) / len(data["results"])
    print(f"{label:<20} {data['avg_latency']:>10.2f}s {avg_words:>10.0f}")

# ── Save results ───────────────────────────────────────────────────────────
Path("evaluation").mkdir(exist_ok=True)
output_path = Path("evaluation/quantization_results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nFull results saved to {output_path}")
print("\nKey insight for README:")
q8_latency = results["Q8 (default)"]["avg_latency"]
q4_latency = results["Q4 (quantized)"]["avg_latency"]
speedup = ((q8_latency - q4_latency) / q8_latency) * 100
print(f"Q4 is {speedup:.1f}% faster than Q8 on average")