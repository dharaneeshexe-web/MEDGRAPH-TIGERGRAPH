import json
import time
import numpy as np
import faiss
from groq import Groq
from pathlib import Path
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────
GROQ_KEY = os.environ["GROQ_KEY"]
ABSTRACTS_FILE = "pubmed_abstracts.json"
INDEX_FILE     = "faiss_index.bin"
META_FILE      = "faiss_meta.json"
GEN_MODEL      = "llama-3.3-70b-versatile"
BATCH_SIZE     = 256

client   = Groq(api_key=GROQ_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Step 1: Build or load FAISS index ────────────────────────────────────
def build_index():
    print("Loading abstracts...")
    with open(ABSTRACTS_FILE) as f:
        abstracts = json.load(f)

    texts = [f"{a['title']}. {a['abstract']}" for a in abstracts[:5000]]
    print(f"Embedding {len(texts)} abstracts locally...")

    matrix = embedder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")

    dim = matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(matrix)

    faiss.write_index(index, INDEX_FILE)
    meta = [{"pmid": a["pmid"], "title": a["title"], "text": t}
            for a, t in zip(abstracts[:5000], texts)]
    with open(META_FILE, "w") as f:
        json.dump(meta, f)

    print(f"\n✅ FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index, meta

def load_index():
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE) as f:
        meta = json.load(f)
    print(f"✅ Loaded FAISS index: {index.ntotal} vectors")
    return index, meta

# ── Step 2: Retrieve + Generate ──────────────────────────────────────────
def baseline_query(question: str, index, meta, top_k=5) -> dict:
    q_vec = embedder.encode([question], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_vec, top_k)
    retrieved = [meta[i] for i in indices[0]]

    context = "\n\n".join(
        f"[{r['title']}]\n{r['text'][:400]}" for r in retrieved
    )

    prompt = f"""You are a medical research assistant.
Answer the question using ONLY the provided abstracts.
Be concise and cite which abstracts support your answer.

Question: {question}

Abstracts:
{context}

Answer:"""

    response = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "pipeline": "baseline",
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": [r["pmid"] for r in retrieved],
        "num_sources": top_k
    }

# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if Path(INDEX_FILE).exists():
        index, meta = load_index()
    else:
        index, meta = build_index()

    test_questions = [
        "What drugs are used to treat type 2 diabetes?",
        "What are the symptoms of Alzheimer's disease?",
        "How does metformin work?",
    ]

    results = []
    for q in test_questions:
        print(f"\n🔍 Q: {q}")
        r = baseline_query(q, index, meta)
        print(f"💬 A: {r['answer'][:300]}...")
        print(f"📄 Sources: {r['sources']}")
        results.append(r)
        time.sleep(1)

    with open("results_baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Baseline results saved to results_baseline.json")