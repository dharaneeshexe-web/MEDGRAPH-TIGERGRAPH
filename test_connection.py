import pyTigerGraph as tg
import os
from dotenv import load_dotenv

load_dotenv()

conn = tg.TigerGraphConnection(
    host=os.environ["TG_HOST"],
    username=os.environ["TG_USERNAME"],
    password=os.environ["TG_PASSWORD"]
)

secret = conn.createSecret()
token = conn.getToken(secret)
print("✅ Connected!")
