import requests
import os
from dotenv import load_dotenv

load_dotenv()

TG_HOST  = os.environ["TG_HOST"]
TG_USER  = os.environ["TG_USERNAME"]
TG_PASS  = os.environ["TG_PASSWORD"]
TG_GRAPH = os.environ.get("TG_GRAPHNAME", "MedGraph")

token = requests.post(f"{TG_HOST}/gsql/v1/tokens", auth=(TG_USER, TG_PASS)).json()["token"]
headers = {"Authorization": f"Bearer {token}"}

for vtype in ["Disease", "Drug", "Symptom", "Treatment", "Abstract"]:
    resp = requests.get(
        f"{TG_HOST}/restpp/graph/{TG_GRAPH}/vertices/{vtype}",
        headers=headers,
        params={"limit": 3}
    )
    print(f"\n=== {vtype} (status {resp.status_code}) ===")
    print(resp.text[:500])

resp = requests.get(
    f"{TG_HOST}/restpp/graph/{TG_GRAPH}/edges/Disease/diabetes",
    headers={"Authorization": f"Bearer {token}"}
)
print("\n=== Edges from 'diabetes' ===")
print(resp.text[:1000])
