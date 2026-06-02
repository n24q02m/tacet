"""A controlled synthetic KGQA benchmark.

The benchmark generates an "organisation & society" knowledge graph and a
stream of tail-prediction queries. Every query carries a ground-truth
*answerability class* — the property a controlled study of a routing cascade
needs:

* ``STATED``       — a base fact present in the graph (Tier-1 trivially).
* ``DED_GIVEN``    — entailed by a rule shipped with the system (Tier-1).
* ``DED_DISCOVER`` — entailed by a rule the teacher must *synthesise* first
                     (Tier-3, then Tier-1 once the rule is mined).
* ``INDUCTIVE``    — a withheld edge with no entailing rule but strong
                     structural regularity (Tier-2 KGE territory).
* ``NOVEL``        — an idiosyncratic fact with no graph signal (Tier-3 only;
                     exact repeats become Tier-1 via write-back).

Controlling the class mix, the workload size, the repeat rate and the graph
size lets the experiments isolate exactly when and why online distillation
lowers cost.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from tacet.core.graph import WorldGraph
from tacet.core.ontology import NodeType, Ontology, RelationType
from tacet.core.symbolic import Rule

Triple = tuple[str, str, str]
Query = tuple[str, str]

CLASSES = ("STATED", "DED_GIVEN", "DED_DISCOVER", "INDUCTIVE", "NOVEL")


@dataclass
class Benchmark:
    graph: WorldGraph
    ontology: Ontology
    given_rules: list[Rule]
    oracle: Callable[[str, str], list[str]]
    workload: list[Query]
    classes: list[str]  # parallel to workload
    discoverable_relations: tuple[str, ...]
    entity_pool: list[str]
    calibration: list[tuple[str, str, list[str], str]] = field(default_factory=list)
    truth: dict[tuple[str, str], list[str]] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    n_people: int = 120
    n_companies: int = 8
    n_departments: int = 6
    n_skills: int = 12
    n_languages: int = 4
    n_regions: int = 6
    cities_per_region: int = 3
    workload_size: int = 300
    repeat_rate: float = 0.30
    class_mix: tuple[float, ...] = (0.22, 0.20, 0.20, 0.23, 0.15)  # CLASSES order
    language_noise: float = 0.15
    lang_withhold: float = 0.30  # fraction of people whose language is a query
    lang_calibration: float = 0.10  # fraction reserved for KGE calibration
    seed: int = 0


def _closure(pairs: list[tuple[str, str]]) -> dict[str, set[str]]:
    """Transitive closure of a relation given as (parent, child) pairs."""
    direct: dict[str, set[str]] = {}
    for a, b in pairs:
        direct.setdefault(a, set()).add(b)
    closure: dict[str, set[str]] = {}
    for start in direct:
        seen: set[str] = set()
        stack = list(direct.get(start, ()))
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(direct.get(node, ()))
        closure[start] = seen
    return closure


def generate(config: BenchmarkConfig | None = None) -> Benchmark:
    """Generate a complete benchmark instance from `config`."""
    cfg = config or BenchmarkConfig()
    rng = random.Random(cfg.seed)

    # ---- entity universe -------------------------------------------------
    people = [f"person_{i:03d}" for i in range(cfg.n_people)]
    companies = [f"company_{i}" for i in range(cfg.n_companies)]
    departments = [f"dept_{i}" for i in range(cfg.n_departments)]
    skills = [f"skill_{i}" for i in range(cfg.n_skills)]
    languages = [f"lang_{i}" for i in range(cfg.n_languages)]
    regions = [f"region_{i}" for i in range(cfg.n_regions)]
    countries = [f"country_{i}" for i in range(max(2, cfg.n_regions // 3))]
    cities = [f"city_{r}_{j}" for r in range(cfg.n_regions) for j in range(cfg.cities_per_region)]
    foods = [f"food_{i}" for i in range(8)]

    g = WorldGraph(name="synthetic-org")
    for p in people:
        g.add_node(p, "Person")
    for c in companies:
        g.add_node(c, "Company")
    for d in departments:
        g.add_node(d, "Department")
    for s in skills:
        g.add_node(s, "Skill")
    for la in languages:
        g.add_node(la, "Language")
    for ci in cities:
        g.add_node(ci, "City")
    for re in regions:
        g.add_node(re, "Region")
    for co in countries:
        g.add_node(co, "Country")

    # ---- geography hierarchy (part_of, transitive) -----------------------
    region_country = {re: countries[i % len(countries)] for i, re in enumerate(regions)}
    city_region: dict[str, str] = {}
    for ci in cities:
        re = "region_" + ci.split("_")[1]
        city_region[ci] = re
        g.add_edge(ci, "part_of", re)
    for re, co in region_country.items():
        g.add_edge(re, "part_of", co)

    # language clusters by employer — a 1-hop homophily signal the KGE can learn
    # (most colleagues share a language; ~language_noise fraction differ).
    company_language = {co: languages[i % cfg.n_languages] for i, co in enumerate(companies)}
    dept_skill_profile = {d: rng.sample(skills, k=min(4, len(skills))) for d in departments}

    # ---- per-person base facts ------------------------------------------
    person_company: dict[str, str] = {}
    person_city: dict[str, str] = {}
    person_dept: dict[str, str] = {}
    person_language: dict[str, str] = {}
    company_city = {co: rng.choice(cities) for co in companies}
    for co, ci in company_city.items():
        g.add_edge(co, "located_in", ci)

    for p in people:
        co = rng.choice(companies)
        person_company[p] = co
        g.add_edge(p, "works_at", co)
        ci = company_city[co] if rng.random() < 0.75 else rng.choice(cities)
        person_city[p] = ci
        g.add_edge(p, "lives_in", ci)
        d = rng.choice(departments)
        person_dept[p] = d
        g.add_edge(p, "member_of", d)
        for s in rng.sample(dept_skill_profile[d], k=rng.randint(2, 4)):
            g.add_edge(p, "has_skill", s)
        if rng.random() < cfg.language_noise:
            person_language[p] = rng.choice(languages)
        else:
            person_language[p] = company_language[co]

    # ---- friendships (symmetric, homophilous) ---------------------------
    for p in people:
        peers = [q for q in people if q != p and person_company[q] == person_company[p]]
        for q in rng.sample(peers, k=min(len(peers), rng.randint(1, 3))):
            g.add_edge(p, "friend_of", q)
            g.add_edge(q, "friend_of", p)

    # ---- family forest (parent_of) --------------------------------------
    parent_pairs: list[tuple[str, str]] = []
    shuffled = people[:]
    rng.shuffle(shuffled)
    for i, child in enumerate(shuffled):
        if i >= cfg.n_people // 2 and rng.random() < 0.7:
            parent = shuffled[rng.randint(0, i - 1)]
            g.add_edge(parent, "parent_of", child)
            parent_pairs.append((parent, child))

    # ---- management tree per company (manages) --------------------------
    manage_pairs: list[tuple[str, str]] = []
    by_company: dict[str, list[str]] = {}
    for p in people:
        by_company.setdefault(person_company[p], []).append(p)
    for staff in by_company.values():
        rng.shuffle(staff)
        for i in range(1, len(staff)):
            boss = staff[rng.randint(0, i - 1)]
            g.add_edge(boss, "manages", staff[i])
            manage_pairs.append((boss, staff[i]))

    # ---- derived relations (oracle only) --------------------------------
    ancestor = _closure(parent_pairs)
    superior = _closure(manage_pairs)
    colleagues: dict[str, set[str]] = {}
    for staff in by_company.values():
        for p in staff:
            colleagues[p] = {q for q in staff if q != p}
    person_food = {p: rng.choice(foods) for p in people}

    # ---- truth table -----------------------------------------------------
    truth: dict[tuple[str, str], list[str]] = {}
    for p in people:
        truth[(p, "works_at")] = [person_company[p]]
        truth[(p, "lives_in")] = [person_city[p]]
        truth[(p, "member_of")] = [person_dept[p]]
        truth[(p, "ancestor_of")] = sorted(ancestor.get(p, set()))
        truth[(p, "superior_of")] = sorted(superior.get(p, set()))
        truth[(p, "colleague_of")] = sorted(colleagues.get(p, set()))
        truth[(p, "primary_language")] = [person_language[p]]
        truth[(p, "favourite_food")] = [person_food[p]]

    def oracle(head: str, relation: str) -> list[str]:
        return list(truth.get((head, relation), []))

    # ---- language: stated / withheld / calibration split ----------------
    lang_people = people[:]
    rng.shuffle(lang_people)
    n_q = int(cfg.lang_withhold * cfg.n_people)
    n_cal = int(cfg.lang_calibration * cfg.n_people)
    lang_query = set(lang_people[:n_q])
    lang_cal = set(lang_people[n_q : n_q + n_cal])
    for p in people:
        if p not in lang_query and p not in lang_cal:
            g.add_edge(p, "primary_language", person_language[p])
    calibration = [(p, "primary_language", languages, person_language[p]) for p in sorted(lang_cal)]

    # ---- ontology --------------------------------------------------------
    onto = Ontology()
    for t in ("Person", "Company", "Department", "Skill", "Language", "City", "Region", "Country"):
        onto.add_node_type(NodeType(t))
    P = frozenset({"Person"})
    rel_specs = [
        RelationType("works_at", P, frozenset({"Company"}), functional=True),
        RelationType("lives_in", P, frozenset({"City"}), functional=True),
        RelationType("member_of", P, frozenset({"Department"}), functional=True),
        RelationType("located_in", frozenset({"Company"}), frozenset({"City"}), functional=True),
        RelationType(
            "part_of",
            frozenset({"City", "Region"}),
            frozenset({"Region", "Country"}),
            transitive=True,
        ),
        RelationType("friend_of", P, P, symmetric=True),
        RelationType("has_skill", P, frozenset({"Skill"})),
        RelationType("parent_of", P, P),
        RelationType("manages", P, P),
        RelationType("primary_language", P, frozenset({"Language"}), functional=True),
        RelationType("ancestor_of", P, P),
        RelationType("superior_of", P, P),
        RelationType("colleague_of", P, P),
        RelationType("favourite_food", P, frozenset({"Food"})),
    ]
    for rt in rel_specs:
        onto.add_relation_type(rt)

    # ---- rules shipped with the system (ancestor_of) --------------------
    given_rules = [
        Rule("given:ancestor_base", (("?x", "parent_of", "?y"),), ("?x", "ancestor_of", "?y")),
        Rule(
            "given:ancestor_step",
            (("?x", "ancestor_of", "?z"), ("?z", "parent_of", "?y")),
            ("?x", "ancestor_of", "?y"),
        ),
    ]
    discoverable = ("superior_of", "colleague_of")

    # ---- build the query workload ---------------------------------------
    workload, classes = _build_workload(cfg, rng, people, lang_query, ancestor, superior)

    return Benchmark(
        graph=g,
        ontology=onto,
        given_rules=given_rules,
        oracle=oracle,
        workload=workload,
        classes=classes,
        discoverable_relations=discoverable,
        entity_pool=people,
        calibration=calibration,
        truth=truth,
    )


def _build_workload(
    cfg: BenchmarkConfig,
    rng: random.Random,
    people: list[str],
    lang_query: set[str],
    ancestor: dict[str, set[str]],
    superior: dict[str, set[str]],
) -> tuple[list[Query], list[str]]:
    """Sample the query stream: distinct queries per class, then add repeats."""
    have_ancestor = [p for p in people if ancestor.get(p)]
    have_superior = [p for p in people if superior.get(p)]
    lang_q = sorted(lang_query)

    pools: dict[str, Callable[[], Query]] = {
        "STATED": lambda: (rng.choice(people), rng.choice(["works_at", "lives_in", "member_of"])),
        "DED_GIVEN": lambda: (
            (rng.choice(have_ancestor) if have_ancestor else rng.choice(people)),
            "ancestor_of",
        ),
        "DED_DISCOVER": lambda: (
            (rng.choice(have_superior) if have_superior else rng.choice(people), "superior_of")
            if rng.random() < 0.5
            else (rng.choice(people), "colleague_of")
        ),
        "INDUCTIVE": lambda: (
            (rng.choice(lang_q) if lang_q else rng.choice(people)),
            "primary_language",
        ),
        "NOVEL": lambda: (rng.choice(people), "favourite_food"),
    }

    n_distinct = max(1, int(cfg.workload_size * (1 - cfg.repeat_rate)))
    distinct: list[Query] = []
    distinct_cls: list[str] = []
    seen: set[Query] = set()
    weights = list(cfg.class_mix)
    guard = 0
    while len(distinct) < n_distinct and guard < n_distinct * 50:
        guard += 1
        cls = rng.choices(CLASSES, weights=weights, k=1)[0]
        q = pools[cls]()
        if q in seen:
            continue
        seen.add(q)
        distinct.append(q)
        distinct_cls.append(cls)

    # interleave repeats (Zipfian-ish: re-draw an already-issued query)
    workload: list[Query] = []
    classes: list[str] = []
    issued: list[int] = []
    di = 0
    while len(workload) < cfg.workload_size:
        if issued and rng.random() < cfg.repeat_rate:
            j = rng.choice(issued)
            workload.append(distinct[j])
            classes.append(distinct_cls[j])
        elif di < len(distinct):
            workload.append(distinct[di])
            classes.append(distinct_cls[di])
            issued.append(di)
            di += 1
        elif issued:
            j = rng.choice(issued)
            workload.append(distinct[j])
            classes.append(distinct_cls[j])
        else:
            break
    return workload, classes
