from fastapi.testclient import TestClient
from tacet.serve.server import build_app
from tacet.core.graph import WorldGraph
from tacet.core.ontology import Ontology

app = build_app(WorldGraph(), Ontology())
client = TestClient(app)

response = client.get("/nonexistent")
print(response.headers)
