from fastapi.testclient import TestClient
from tacet.serve.server import build_app
from tacet.core.graph import WorldGraph
from tacet.core.ontology import Ontology

app = build_app(WorldGraph(), Ontology())

@app.get("/crash")
def crash():
    raise Exception("test crash")

client = TestClient(app, raise_server_exceptions=False)

import logging
logging.basicConfig(level=logging.ERROR)

response = client.get("/crash")
