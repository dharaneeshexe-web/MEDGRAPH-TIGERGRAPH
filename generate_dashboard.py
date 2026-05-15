import json

def load(path):
    with open(path) as f:
        return json.load(f)

baseline = load("results_baseline.json")
graphrag = load("results_graphrag.json")
hybrid   = load("results_hybrid.json")

# Pair by question
rows = []
for b, g, h in zip(baseline, graphrag, hybrid):
    rows.append({
        "question": b["question"],
        "baseline": b["answer"],
        "graphrag": g["answer"],
        "hybrid":   h["answer"],
        "baseline_sources": len(b.get("sources", [])),
        "graphrag_entities": g.get("graph_entities", 0),
        "hybrid_vector": len(h.get("vector_sources", [])),
        "hybrid_graph": h.get("graph_entities", 0),
    })

data_json = json.dumps(rows)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>MedGraph RAG — Eval Dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; padding: 24px; }}
  h1 {{ text-align: center; color: #7eb8f7; margin-bottom: 6px; font-size: 1.8rem; }}
  .subtitle {{ text-align: center; color: #888; margin-bottom: 32px; font-size: 0.9rem; }}

  .summary {{ display: flex; gap: 16px; justify-content: center; margin-bottom: 36px; flex-wrap: wrap; }}
  .card {{ background: #1a1d27; border-radius: 12px; padding: 20px 28px; text-align: center; min-width: 160px; }}
  .card .label {{ font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }}
  .card .value {{ font-size: 2rem; font-weight: bold; }}
  .baseline-color {{ color: #f87171; }}
  .graphrag-color {{ color: #34d399; }}
  .hybrid-color   {{ color: #7eb8f7; }}

  .question-block {{ background: #1a1d27; border-radius: 12px; margin-bottom: 28px; overflow: hidden; }}
  .question-header {{ background: #252836; padding: 14px 20px; font-size: 1rem; font-weight: 600; color: #f0c040; border-left: 4px solid #f0c040; }}
  .pipelines {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0; }}
  .pipeline {{ padding: 16px 20px; border-right: 1px solid #252836; }}
  .pipeline:last-child {{ border-right: none; }}
  .pipeline-label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 10px; padding: 3px 8px; border-radius: 4px; display: inline-block; }}
  .baseline-label {{ background: #3b1f1f; color: #f87171; }}
  .graphrag-label {{ background: #1a3b2f; color: #34d399; }}
  .hybrid-label   {{ background: #1a2a3b; color: #7eb8f7; }}
  .answer {{ font-size: 0.85rem; line-height: 1.6; color: #ccc; max-height: 200px; overflow-y: auto; }}
  .meta {{ margin-top: 10px; font-size: 0.72rem; color: #666; }}

  .winner-bar {{ display: flex; gap: 12px; padding: 12px 20px; background: #13151f; align-items: center; flex-wrap: wrap; }}
  .winner-label {{ font-size: 0.75rem; color: #888; }}
  .badge {{ padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }}
  .badge-graphrag {{ background: #1a3b2f; color: #34d399; }}
  .badge-hybrid   {{ background: #1a2a3b; color: #7eb8f7; }}

  @media (max-width: 768px) {{
    .pipelines {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<h1>🧬 MedGraph RAG — Evaluation Dashboard</h1>
<p class="subtitle">Comparing Baseline FAISS vs GraphRAG vs Hybrid on PubMed medical abstracts</p>

<div class="summary">
  <div class="card">
    <div class="label">Questions Tested</div>
    <div class="value" style="color:#f0c040" id="total-q">—</div>
  </div>
  <div class="card">
    <div class="label">Baseline</div>
    <div class="value baseline-color">Vector Only</div>
  </div>
  <div class="card">
    <div class="label">GraphRAG</div>
    <div class="value graphrag-color">Graph Only</div>
  </div>
  <div class="card">
    <div class="label">Hybrid</div>
    <div class="value hybrid-color">Vector + Graph</div>
  </div>
</div>

<div id="results"></div>

<script>
const data = {data_json};
document.getElementById("total-q").textContent = data.length;

const container = document.getElementById("results");

data.forEach((row, i) => {{
  const block = document.createElement("div");
  block.className = "question-block";

  block.innerHTML = `
    <div class="question-header">Q${{i+1}}: ${{row.question}}</div>
    <div class="pipelines">
      <div class="pipeline">
        <span class="pipeline-label baseline-label">Baseline</span>
        <div class="answer">${{row.baseline}}</div>
        <div class="meta">📄 ${{row.baseline_sources}} vector sources</div>
      </div>
      <div class="pipeline">
        <span class="pipeline-label graphrag-label">GraphRAG</span>
        <div class="answer">${{row.graphrag}}</div>
        <div class="meta">🕸️ ${{row.graphrag_entities}} graph entities</div>
      </div>
      <div class="pipeline">
        <span class="pipeline-label hybrid-label">Hybrid</span>
        <div class="answer">${{row.hybrid}}</div>
        <div class="meta">📄 ${{row.hybrid_vector}} vector + 🕸️ ${{row.hybrid_graph}} graph</div>
      </div>
    </div>
    <div class="winner-bar">
      <span class="winner-label">✨ Best coverage:</span>
      <span class="badge badge-hybrid">Hybrid uses most sources</span>
      <span class="badge badge-graphrag">GraphRAG adds entity relationships</span>
    </div>
  `;
  container.appendChild(block);
}});
</script>
</body>
</html>"""

with open("dashboard.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ Dashboard generated → open dashboard.html in your browser")