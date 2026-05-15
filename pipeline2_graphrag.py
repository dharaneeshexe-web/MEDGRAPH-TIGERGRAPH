import json
import time
import requests
from groq import Groq
from pathlib import Path
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────
GROQ_KEY  = os.environ["GROQ_KEY"]
TG_HOST   = os.environ["TG_HOST"]
TG_USER   = os.environ["TG_USERNAME"]
TG_PASS   = os.environ["TG_PASSWORD"]
TG_GRAPH   = "MedGraph"
GEN_MODEL  = "llama-3.3-70b-versatile"

client   = Groq(api_key=GROQ_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── TigerGraph helpers ────────────────────────────────────────────────────
def get_tg_token():
    resp = requests.post(f"{TG_HOST}/gsql/v1/tokens", auth=(TG_USER, TG_PASS))
    resp.raise_for_status()
    return resp.json()["token"]

def lookup_vertex(vtype, vid, token):
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(
        f"{TG_HOST}/restpp/graph/{TG_GRAPH}/vertices/{vtype}/{vid}",
        headers=headers
    )
    if resp.status_code == 200:
        results = resp.json().get("results", [])
        return results[0] if results else None
    return None

def get_neighbors(vtype, vid, token):
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(
        f"{TG_HOST}/restpp/graph/{TG_GRAPH}/edges/{vtype}/{vid}",
        headers=headers
    )
    if resp.status_code == 200:
        return resp.json().get("results", [])
    return []

def get_abstract(pmid, token):
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(
        f"{TG_HOST}/restpp/graph/{TG_GRAPH}/vertices/Abstract/{pmid}",
        headers=headers
    )
    if resp.status_code == 200:
        results = resp.json().get("results", [])
        return results[0] if results else None
    return None

def get_abstract_edges(pmid, token):
    """Get all entities that also mention this abstract — reverse traversal."""
    headers = {"Authorization": f"Bearer {token}"}
    all_edges = []
    for vtype in ["Drug", "Disease", "Symptom", "Treatment"]:
        resp = requests.get(
            f"{TG_HOST}/restpp/graph/{TG_GRAPH}/edges/{vtype}",
            headers=headers,
            params={"limit": 100}
        )
        if resp.status_code == 200:
            edges = resp.json().get("results", [])
            # keep only edges pointing to this abstract
            matching = [e for e in edges if e.get("to_id") == pmid]
            all_edges.extend(matching)
    return all_edges

def extract_keywords(question: str) -> list[str]:
    stopwords = {"what","how","does","is","are","the","a","an","of",
                 "in","to","for","and","or","with","used","treat","do",
                 "drugs","symptoms","treatments","research","studies",
                 "literature","appear","alongside","co-mentioned","related"}
    words = question.lower().replace("?","").replace(",","").replace("'s","").split()
    return [w for w in words if w not in stopwords and len(w) > 3]

# ── 2-hop graph traversal ─────────────────────────────────────────────────
def two_hop_traverse(seed_entity: str, seed_type: str, token: str) -> dict:
    """
    Hop 1: seed entity → MENTIONED_IN → abstracts
    Hop 2: abstracts ← MENTIONED_IN ← other entities

    Uses reverse-edge lookup on each abstract vertex instead of scanning
    all vertices — O(abstracts) requests vs the previous O(abstracts × vtypes × all_vertices).
    """
    headers = {"Authorization": f"Bearer {token}"}

    # Hop 1: get abstracts mentioning this entity
    neighbors = get_neighbors(seed_type, seed_entity, token)
    abstract_ids = [n["to_id"] for n in neighbors if n.get("to_type") == "Abstract"]
    print(f"  Hop 1: '{seed_entity}' → {len(abstract_ids)} abstracts")

    if not abstract_ids:
        return {"abstracts": [], "co_entities": {}}

    # Fetch abstract content (top 8)
    abstracts = []
    for pmid in abstract_ids[:8]:
        ab = get_abstract(pmid, token)
        if ab:
            abstracts.append({
                "pmid": pmid,
                "title": ab["attributes"].get("title", ""),
                "text":  ab["attributes"].get("text",  "")[:600],
            })

    # Hop 2: for each abstract, get its incoming edges (reverse traversal)
    # TigerGraph returns edges regardless of direction when queried on the target vertex.
    co_entities = {"Drug": set(), "Disease": set(), "Symptom": set(), "Treatment": set()}
    for pmid in abstract_ids[:8]:
        ab_edges = get_neighbors("Abstract", pmid, token)
        for edge in ab_edges:
            co_type = edge.get("to_type") or edge.get("from_type")
            co_id   = edge.get("to_id")   or edge.get("from_id")
            if co_type in co_entities and co_id and co_id != seed_entity:
                co_entities[co_type].add(co_id)

    co_summary = {k: list(v)[:10] for k, v in co_entities.items() if v}
    print(f"  Hop 2: co-entities → {co_summary}")
    return {"abstracts": abstracts, "co_entities": co_summary}

# ── Build graph context ───────────────────────────────────────────────────
def build_graph_context(question: str, token: str) -> str:
    keywords = extract_keywords(question)
    print(f"  🔑 Keywords: {keywords}")

    context_parts = []
    found = False

    for kw in keywords[:4]:
        for vtype in ["Disease", "Drug", "Symptom", "Treatment"]:
            v = lookup_vertex(vtype, kw, token)
            if v:
                found = True
                vid = v["v_id"]
                print(f"  📍 Found [{vtype}] {vid} — running 2-hop traversal...")
                result = two_hop_traverse(vid, vtype, token)

                context_parts.append(f"[{vtype}] {vid}")

                if result["co_entities"]:
                    context_parts.append("Co-mentioned entities in shared research abstracts:")
                    for etype, entities in result["co_entities"].items():
                        context_parts.append(f"  [{etype}s]: {', '.join(entities)}")

                if result["abstracts"]:
                    context_parts.append("Supporting abstracts:")
                    for ab in result["abstracts"][:5]:
                        context_parts.append(f"  → {ab['title']}\n    {ab['text'][:400]}")

    if not found:
        return "No relevant graph entities found."

    return "\n".join(context_parts)

# ── GraphRAG query ────────────────────────────────────────────────────────
def graphrag_query(question: str, token: str) -> dict:
    graph_context = build_graph_context(question, token)

    prompt = f"""You are a medical research assistant with access to a knowledge graph.
The graph was traversed using 2-hop traversal:
  Entity → mentioned_in → Abstracts ← mentioned_in ← Related Entities

Use the graph context to answer with specific entity relationships.
Name the co-mentioned drugs, symptoms, and treatments explicitly.

Question: {question}

Knowledge Graph Context:
{graph_context}

Answer (be specific about which entities co-occur and what the research shows):"""

    response = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "pipeline": "graphrag",
        "question": question,
        "answer": response.choices[0].message.content,
        "graph_entities": graph_context.count("["),
        "context_preview": graph_context[:400]
    }

# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔐 Getting TigerGraph token...")
    token = get_tg_token()
    print("✅ Token obtained")

    test_questions = [
        "What drugs are co-mentioned with diabetes in research literature?",
        "What symptoms appear in the same studies as metformin?",
        "What treatments are researched alongside hypertension?",
    ]

    results = []
    for q in test_questions:
        print(f"\n🔍 Q: {q}")
        r = graphrag_query(q, token)
        print(f"💬 A: {r['answer'][:300]}...")
        results.append(r)
        time.sleep(1)

    with open("results_graphrag.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ GraphRAG results saved to results_graphrag.json")