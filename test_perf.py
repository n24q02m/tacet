import time

from tacet.core.graph import WorldGraph
from tacet.core.symbolic import Ontology, RuleEngine

graph = WorldGraph()
for i in range(5000):
    graph.add_edge(f"n{i}", "rel1", f"n{i + 1}")

ontology = Ontology()
engine = RuleEngine(ontology)
start = time.perf_counter()
engine.materialise(graph)
print(f"materialise 5000 facts took {time.perf_counter() - start:.4f}s")
