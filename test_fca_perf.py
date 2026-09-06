import time

from tacet.distill.fca import FormalContext


# Generate a smaller random formal context that finishes faster
def generate_context(n_objects=500, n_attributes=50, density=0.1):
    import random

    random.seed(42)
    objects = [f"obj_{i}" for i in range(n_objects)]
    attributes = [(f"rel_{i}", "tail") for i in range(n_attributes)]
    incidence = {}
    for obj in objects:
        attrs = [i for i in range(n_attributes) if random.random() < density]
        incidence[obj] = frozenset(attrs)

    return FormalContext(objects=objects, attributes=attributes, incidence=incidence)


ctx = generate_context()

start = time.time()
concepts = ctx.concepts()
end = time.time()

print(f"Time: {end - start:.3f}s")
print(f"Found {len(concepts)} concepts")
