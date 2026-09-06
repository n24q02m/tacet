import time

from tacet.data.privaci_graph import SLOT_RELATIONS
from tacet.distill.compliance_miner import LabeledCase, mine_compliance_rules


def _case(i, atoms, verdict, articles=()):
    return LabeledCase(
        case_id=f"C-{i:03d}",
        atoms=frozenset(atoms),
        verdict=verdict,
        articles=tuple(articles),
    )


def generate_large_dataset():
    cases = []

    import random

    random.seed(42)
    slots = list(SLOT_RELATIONS.keys())
    atoms = [(slot, f"cat_{j}") for slot in slots for j in range(25)]

    # 5000 cases, each with 10 random atoms
    for i in range(5000):
        case_atoms = random.sample(atoms, 10)
        verdict = "permit" if random.random() > 0.5 else "prohibit"
        articles = ["art1"] if verdict == "prohibit" else []
        cases.append(_case(i, case_atoms, verdict, articles))

    # Plant a frequent rule that gets pruned
    plant = [atoms[0], atoms[1], atoms[2]]
    for i in range(5000, 5200):
        cases.append(_case(i, plant, "prohibit", ["art99"]))

    return cases


data = generate_large_dataset()

start = time.time()
rules = mine_compliance_rules(data, min_support=20, min_confidence=0.9, max_atoms=3)
end = time.time()
print(f"Time: {end - start:.3f}s")
print(f"Mined {len(rules)} rules")
