import json
import time
import requests
import numpy as np
import faiss
from groq import Groq
from pathlib import Path
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────
GROQ_KEY   = os.environ["GROQ_KEY"]
TG_HOST    = os.environ["TG_HOST"]
TG_USER    = os.environ["TG_USERNAME"]
TG_PASS    = os.environ["TG_PASSWORD"]
TG_GRAPH   = "MedGraph"
GEN_MODEL  = "llama-3.3-70b-versatile"
INDEX_FILE = "faiss_index.bin"
META_FILE  = "faiss_meta.json"

client   = Groq(api_key=GROQ_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── TigerGraph token ──────────────────────────────────────────────────────
def get_tg_token():
    resp = requests.post(
        f"{TG_HOST}/gsql/v1/tokens",
        auth=(TG_USER, TG_PASS)
    )
    resp.raise_for_status()
    return resp.json()["token"]

# ── FAISS index ───────────────────────────────────────────────────────────
def load_index():
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE) as f:
        meta = json.load(f)
    print(f"✅ Loaded FAISS index: {index.ntotal} vectors")
    return index, meta

# ── Pipeline 1: Vector retrieval ─────────────────────────────────────────
def vector_retrieve(question: str, index, meta, top_k=4) -> list:
    q_vec = embedder.encode([question], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_vec, top_k)
    return [meta[i] for i in indices[0]]

# ── Pipeline 2: Graph retrieval ───────────────────────────────────────────
def extract_keywords(question: str) -> list[str]:
    stopwords = {"what","how","does","is","are","the","a","an","of",
                 "in","to","for","and","or","with","used","treat","do"}
    words = question.lower().replace("?","").replace(",","").replace("'s","").split()
    return [w for w in words if w not in stopwords and len(w) > 3]

def lookup_vertex(vtype: str, vid: str, token: str) -> dict | None:
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(
        f"{TG_HOST}/restpp/graph/{TG_GRAPH}/vertices/{vtype}/{vid}",
        headers=headers
    )
    if resp.status_code == 200:
        results = resp.json().get("results", [])
        return results[0] if results else None
    return None

def get_neighbors(vtype: str, vid: str, token: str) -> list:
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(
        f"{TG_HOST}/restpp/graph/{TG_GRAPH}/edges/{vtype}/{vid}",
        headers=headers
    )
    if resp.status_code == 200:
        return resp.json().get("results", [])
    return []

def get_abstract(pmid: str, token: str) -> dict | None:
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(
        f"{TG_HOST}/restpp/graph/{TG_GRAPH}/vertices/Abstract/{pmid}",
        headers=headers
    )
    if resp.status_code == 200:
        results = resp.json().get("results", [])
        return results[0] if results else None
    return None

def graph_retrieve(question: str, token: str) -> list[str]:
    keywords = extract_keywords(question)
    context_parts = []

    for kw in keywords[:4]:
        for vtype in ["Disease", "Drug", "Symptom", "Treatment"]:
            v = lookup_vertex(vtype, kw, token)
            if v:
                vid = v.get("v_id", "")
                context_parts.append(f"[{vtype}] {vid}")
                neighbors = get_neighbors(vtype, vid, token)
                for n in neighbors[:5]:
                    n_type = n.get("to_type", "")
                    n_id   = n.get("to_id", "")
                    e_type = n.get("e_type", "")
                    if n_type == "Abstract":
                        abstract = get_abstract(n_id, token)
                        if abstract:
                            attrs = abstract.get("attributes", {})
                            context_parts.append(
                                f"  → [Abstract via {e_type}] {attrs.get('title','')}\n"
                                f"    {attrs.get('text','')[:300]}"
                            )
                    else:
                        context_parts.append(f"  → [{n_type} via {e_type}] {n_id}")

    return context_parts

# ── Context deduplication ─────────────────────────────────────────────────
def deduplicate_context(vector_context: str, graph_context: str, question: str, max_lines: int = 35) -> tuple[str, str]:
    """
    Deduplicate and rank lines from both contexts by keyword relevance.
    Returns (clean_vector_ctx, clean_graph_ctx) with duplicates removed.
    Avoids sending contradictory / redundant info that tanks LLM quality.
    """
    keywords = set(
        w.lower() for w in question.replace("?", "").split()
        if len(w) > 3
    )

    def score_line(line: str) -> int:
        low = line.lower()
        return sum(1 for k in keywords if k in low)

    seen_norm: set[str] = set()

    def dedup_block(text: str) -> list[str]:
        kept = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            norm = " ".join(stripped.lower().split())
            if norm not in seen_norm:
                seen_norm.add(norm)
                kept.append(stripped)
        return kept

    vec_lines   = dedup_block(vector_context)
    graph_lines = dedup_block(graph_context)   # seen_norm already populated from vec

    # Sort graph lines by keyword relevance so most useful lines come first
    graph_lines.sort(key=score_line, reverse=True)

    return "\n".join(vec_lines[:max_lines]), "\n".join(graph_lines[:max_lines])

# ── Pipeline 3: Hybrid query ──────────────────────────────────────────────
def hybrid_query(question: str, index, meta, token: str) -> dict:
    # Vector context
    vector_results = vector_retrieve(question, index, meta, top_k=3)
    vector_context_raw = "\n\n".join(
        f"[Vector Match] {r['title']}\n{r['text'][:300]}" for r in vector_results
    )

    # Graph context
    graph_parts = graph_retrieve(question, token)
    graph_context_raw = "\n".join(graph_parts) if graph_parts else "No graph entities found."

    # Deduplicate — prevents redundant/conflicting info hurting LLM output
    vector_context, graph_context = deduplicate_context(vector_context_raw, graph_context_raw, question)

    print(f"  📄 Vector lines: {len(vector_context.splitlines())} | 📊 Graph lines: {len(graph_context.splitlines())}")

    prompt = f"""You are a medical research assistant.
Use BOTH vector search results and knowledge graph traversal to answer.
The vector results provide relevant research abstracts.
The graph results provide structured co-mention relationships between medical entities.

Question: {question}

--- VECTOR SEARCH RESULTS ---
{vector_context}

--- KNOWLEDGE GRAPH CO-MENTIONS ---
{graph_context}

Answer (name specific entities from both sources, be precise):"""

    response = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "pipeline": "hybrid",
        "question": question,
        "answer": response.choices[0].message.content,
        "vector_sources": [r["pmid"] for r in vector_results],
        "graph_entities": len(graph_parts)
    }

# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔐 Getting TigerGraph token...")
    token = get_tg_token()
    print("✅ Token obtained")

    index, meta = load_index()

    test_questions = [
        "What drugs are used to treat type 2 diabetes?",
        "What are the symptoms of Alzheimer's disease?",
        "How does metformin work?",
    ]

    results = []
    for q in test_questions:
        print(f"\n🔍 Q: {q}")
        r = hybrid_query(q, index, meta, token)
        print(f"💬 A: {r['answer'][:300]}...")
        print(f"📄 Vector: {r['vector_sources']} | 📊 Graph parts: {r['graph_entities']}")
        results.append(r)
        time.sleep(1)

    with open("results_hybrid.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Hybrid results saved to results_hybrid.json")