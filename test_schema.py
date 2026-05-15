import requests
import base64
import os
from dotenv import load_dotenv

load_dotenv()

base  = os.environ["TG_HOST"]
user  = os.environ["TG_USERNAME"]
passwd = os.environ["TG_PASSWORD"]

creds = base64.b64encode(f"{user}:{passwd}".encode()).decode("utf-8")
token_response = requests.post(
    f"{base}/gsql/v1/tokens",
    headers={"Authorization": f"Basic {creds}", "Content-Type": "application/json"},
    json={"graph": "MedGraph"},
    verify=False
)
token = token_response.json()["token"]
headers = {"Authorization": f"Bearer {token}"}

schema = requests.get(
    f"{base}/restpp/graph/MedGraph",
    headers=headers,
    verify=False
)
print("Status:", schema.status_code)
print("Body:", schema.text[:1000])
