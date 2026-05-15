# connection.py
import requests
import base64
import warnings
import os
from dotenv import load_dotenv

load_dotenv()

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

def get_headers():
    return {"Authorization": f"Bearer {get_token()}"}

if __name__ == "__main__":
    print("✅ Token:", get_token()[:30], "...")
