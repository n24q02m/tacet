from fastapi.testclient import TestClient
from tacet.serve.server import build_app
from tacet.core.graph import WorldGraph
from tacet.core.ontology import Ontology
import logging
import sys

app = build_app(WorldGraph(), Ontology())

@app.get("/crash")
def crash():
    raise Exception("test crash")

# Setup logging to write to stdout so we can see it
logger = logging.getLogger("tacet.serve.server")
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.ERROR)

client = TestClient(app, raise_server_exceptions=False)
response = client.get("/crash")
