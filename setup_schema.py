import pyTigerGraph as tg
import os
from dotenv import load_dotenv

load_dotenv()

conn = tg.TigerGraphConnection(
    host=os.environ["TG_HOST"],
    username=os.environ["TG_USERNAME"],
    password=os.environ["TG_PASSWORD"],
    gsqlVersion="3.9"
)

result = conn.gsql('''
USE GLOBAL

CREATE SCHEMA_CHANGE JOB medgraph_schema FOR GRAPH MedGraph {
  ADD VERTEX Disease;
  ADD VERTEX Drug;
  ADD VERTEX Symptom;
  ADD VERTEX Treatment;
  ADD VERTEX Abstract;
  ADD EDGE TREATS;
  ADD EDGE CAUSES;
  ADD EDGE INDICATES;
  ADD EDGE USED_IN;
  ADD EDGE MENTIONED_IN;
  ADD EDGE PRESCRIBED_IN;
}

RUN SCHEMA_CHANGE JOB medgraph_schema
''')

print(result)
