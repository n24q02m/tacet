from tacet.serve.server import app, build_app
from tacet.core.graph import WorldGraph
from tacet.core.ontology import Ontology

graph = WorldGraph()
onto = Ontology()

app = build_app(graph, onto)

print(app.middleware_stack)
