import json
import os
from dotenv import load_dotenv

load_dotenv()
import requests
import base64
import warnings
warnings.filterwarnings("ignore")

BASE     = os.environ["TG_HOST"]
USERNAME = os.environ["TG_USERNAME"]
PASSWORD = os.environ["TG_PASSWORD"]

def get_token():
    creds = base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()
    res = requests.post(
        f"{BASE}/gsql/v1/tokens",
        headers={"Authorization": f"Basic {creds}", "Content-Type": "application/json"},
        json={"graph": "MedGraph"},
        verify=False
    )
    return res.json()["token"]

def upsert(token, data):
    res = requests.post(
        f"{BASE}/restpp/graph/MedGraph",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=data,
        verify=False
    )
    return res.json()

with open("entities.json", "r", encoding="utf-8") as f:
    abstracts = json.load(f)

print(f"Loading {len(abstracts)} abstracts into TigerGraph...")
token = get_token()
print("✅ Token obtained")

total_vertices = 0
total_edges = 0

for i, item in enumerate(abstracts):
    if i < 3000:  # skip already loaded
        continue
    pmid = item["pmid"]
    entities = item["entities"]

    vertices = {
        "Abstract": {
            pmid: {
                "title": {"value": item["title"][:500]},
                "text": {"value": item["abstract"][:2000]},
                "pubmed_id": {"value": pmid}
            }
        },
        "Disease": {},
        "Drug": {},
        "Symptom": {},
        "Treatment": {}
    }

    edges = {}

    for disease in entities.get("diseases", []):
        did = disease.replace(" ", "_")
        vertices["Disease"][did] = {
            "name": {"value": disease},
            "description": {"value": ""}
        }
        edges.setdefault("Disease", {}).setdefault(did, {}).setdefault("MENTIONED_IN", {}).setdefault("Abstract", {})[pmid] = {}

    for drug in entities.get("drugs", []):
        drid = drug.replace(" ", "_")
        vertices["Drug"][drid] = {
            "name": {"value": drug},
            "description": {"value": ""}
        }
        edges.setdefault("Drug", {}).setdefault(drid, {}).setdefault("PRESCRIBED_IN", {}).setdefault("Abstract", {})[pmid] = {}

    for symptom in entities.get("symptoms", []):
        sid = symptom.replace(" ", "_")
        vertices["Symptom"][sid] = {
            "name": {"value": symptom}
        }

    for treatment in entities.get("treatments", []):
        tid = treatment.replace(" ", "_")
        vertices["Treatment"][tid] = {
            "name": {"value": treatment},
            "description": {"value": ""}
        }

    payload = {"vertices": vertices, "edges": edges}
    result = upsert(token, payload)

    if result.get("error"):
        print(f"  ⚠️ Error at {i}: {result.get('message', '')[:100]}")
        token = get_token()
    
    total_vertices += sum(len(v) for v in vertices.values())
    total_edges += sum(
        sum(len(et) for et in ev.values())
        for sv in edges.values()
        for ev in sv.values()
    )

    if (i+1) % 500 == 0:
        print(f"  ✅ Loaded {i+1}/5000 — vertices: {total_vertices}, edges: {total_edges}")
        token = get_token()

print(f"\n✅ Done! Total vertices: {total_vertices}, Total edges: {total_edges}")