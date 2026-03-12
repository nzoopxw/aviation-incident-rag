import pandas as pd
import os
import json
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(exist_ok=True)

# ── Step 1: Load and merge all 6 CSVs ─────────────────────────────────────
dfs = []
for csv_file in sorted(RAW_DIR.glob("*.csv")):
    df = pd.read_csv(csv_file, header=1, low_memory=False)
    print(f"Loaded {csv_file.name}: {len(df)} rows")
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal reports after merge: {len(df)}")

# ── Step 2: Keep only the columns we care about ────────────────────────────
KEEP_COLS = {
    "ACN": "acn",
    "Date": "date",
    "Flight Phase": "phase_of_flight",
    "Make Model Name": "aircraft",
    "Primary Problem": "primary_problem",
    "Contributing Factors / Situations": "contributing_factors",
    "Human Factors": "human_factors",
    "Anomaly": "anomaly",
    "Narrative": "narrative",
    "Synopsis": "synopsis"
}

existing_cols = {k: v for k, v in KEEP_COLS.items() if k in df.columns}
df = df[list(existing_cols.keys())].rename(columns=existing_cols)
print(f"Columns kept: {list(df.columns)}")

# ── Step 3: Clean ──────────────────────────────────────────────────────────
df = df.dropna(subset=["narrative"])
df = df[df["narrative"].str.strip() != ""]
df = df.fillna("Unknown")
df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
print(f"Reports after cleaning: {len(df)}")

# ── Step 4: Chunk by paragraph ─────────────────────────────────────────────
chunks = []
for _, row in df.iterrows():
    paragraphs = [p.strip() for p in row["narrative"].split("\n") if len(p.strip()) > 50]
    
    for i, para in enumerate(paragraphs):
        chunk = {
            "chunk_id": f"{row['acn']}_{i}",
            "acn": row["acn"],
            "date": row["date"],
            "aircraft": row["aircraft"],
            "phase_of_flight": row["phase_of_flight"],
            "primary_problem": row["primary_problem"],
            "contributing_factors": row["contributing_factors"],
            "human_factors": row["human_factors"],
            "anomaly": row["anomaly"],
            "synopsis": row["synopsis"],
            "text": para
        }
        chunks.append(chunk)

print(f"Total chunks: {len(chunks)}")

# ── Step 5: Save ───────────────────────────────────────────────────────────
output_path = PROCESSED_DIR / "chunks.json"
with open(output_path, "w") as f:
    json.dump(chunks, f, indent=2)

print(f"\nSaved {len(chunks)} chunks to {output_path}")