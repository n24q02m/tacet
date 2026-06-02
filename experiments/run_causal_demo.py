"""Worked causal demonstration over a small confounded concept graph.

    python experiments/run_causal_demo.py [--seed 0]

What it does
------------
Builds a tiny structural causal model (SCM) over five named *concept* nodes
that mirrors how a subset of an TACET knowledge graph is promoted to a typed
causal DAG (see ``tacet.core.causal``). The graph contains a classic confounder so
that the *naive* observational association ``P(downstream | treatment)`` is
biased, while the interventional estimate ``P(downstream | do(treatment))``
recovers the de-confounded effect.

It then walks the three rungs of Pearl's ladder as first-class engine
operations and prints the **identification certificate**: the estimand plus the
backdoor adjustment set that licenses the interventional query.

The graph
---------

    confounder ---> treatment ---> downstream
        |                              ^
        |                              |
        +------------------------------+
    treatment ---> mediator ---> downstream
    downstream ---> report           (a pure descendant; must NOT be adjusted)

``confounder`` is a common cause of both ``treatment`` and ``downstream``: it is
the source of confounding bias on the ``treatment -> downstream`` edge. The
correct backdoor adjustment set for that edge is therefore ``{confounder}``.
``report`` is a downstream descendant included only to show the criterion does
not wrongly adjust for post-treatment variables.

Honesty note
------------
The engine performs causal *identification given a provided structure*, not
causal *discovery*: the DAG and its mechanisms are supplied, and the engine
plans the query (backdoor / front-door / IV) and emits the certificate. It does
not learn the causal structure from observational data.
"""

from __future__ import annotations

import argparse

from tacet.core.causal import (
    CausalModel,
    backdoor_set,
    counterfactual,
)


def build_demo_scm(seed: int = 0) -> CausalModel:
    """Construct the small confounded SCM over five concept nodes.

    Variables (all binary, domain ``(0, 1)``):

    * ``confounder``  exogenous common cause of treatment and downstream.
    * ``treatment``   the intervention target; biased upward by the confounder.
    * ``mediator``    a node on the front-door path treatment -> downstream.
    * ``downstream``  the outcome; a function of treatment, mediator and the
                      confounder (the latter is what makes the naive
                      ``P(downstream | treatment)`` association confounded).
    * ``report``      a pure descendant of the outcome (post-treatment).

    Returns a fully-specified :class:`~tacet.core.causal.CausalModel` ready for
    ``observe`` / ``intervene`` / ``counterfactual`` queries.
    """
    model = CausalModel(seed=seed)

    # confounder ~ Bernoulli(0.5): the unobserved-in-naive-analysis common cause.
    model.add_variable(
        "confounder",
        domain=(0, 1),
        mechanism=lambda _parents, u: 1 if u < 0.5 else 0,
    )

    # treatment is pushed up by the confounder: P(treat=1 | conf=1) high.
    # This is precisely the back-door path confounder -> treatment.
    model.add_variable(
        "treatment",
        domain=(0, 1),
        parents=("confounder",),
        mechanism=lambda p, u: 1 if u < (0.8 if p["confounder"] == 1 else 0.2) else 0,
    )

    # mediator lies on the front-door path treatment -> mediator -> downstream.
    model.add_variable(
        "mediator",
        domain=(0, 1),
        parents=("treatment",),
        mechanism=lambda p, u: 1 if u < (0.7 if p["treatment"] == 1 else 0.3) else 0,
    )

    # downstream depends on treatment, mediator AND the confounder. The
    # confounder term is what biases the naive observational estimate.
    def _downstream(p: dict[str, object], u: float) -> int:
        base = 0.1
        base += 0.3 if p["treatment"] == 1 else 0.0
        base += 0.2 if p["mediator"] == 1 else 0.0
        base += 0.3 if p["confounder"] == 1 else 0.0
        return 1 if u < base else 0

    model.add_variable(
        "downstream",
        domain=(0, 1),
        parents=("treatment", "mediator", "confounder"),
        mechanism=_downstream,
    )

    # report is a pure post-outcome descendant (must never be adjusted for).
    model.add_variable(
        "report",
        domain=(0, 1),
        parents=("downstream",),
        mechanism=lambda p, u: p["downstream"],
    )

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--samples", type=int, default=200000, help="Monte-Carlo sample size for the estimates."
    )
    args = parser.parse_args()

    model = build_demo_scm(seed=args.seed)
    n = args.samples

    # --- Rung 1: association -- P(downstream=1 | treatment=1) ----------------
    observational = model.probability("downstream", 1, evidence={"treatment": 1}, n=n)

    # --- Rung 2: intervention -- P(downstream=1 | do(treatment=1)) ----------
    intervened_on = model.intervene(do={"treatment": 1})
    interventional_on = intervened_on.probability("downstream", 1, n=n)
    intervened_off = model.intervene(do={"treatment": 0})
    interventional_off = intervened_off.probability("downstream", 1, n=n)
    ate = interventional_on - interventional_off

    # --- Rung 3: counterfactual ---------------------------------------------
    # "Given that we observed treatment=0 and downstream=0, what would the
    # downstream outcome have been had we instead set do(treatment=1)?"
    cf = counterfactual(
        model,
        evidence={"treatment": 0, "downstream": 0},
        do={"treatment": 1},
        target="downstream",
        n=n // 4,
    )

    # --- Identification certificate -----------------------------------------
    adjustment = backdoor_set(model, "treatment", "downstream")

    print("TACET causal demonstration -- confounded concept graph")
    print("=" * 60)
    print(f"seed={args.seed}  samples={n}")
    print()
    print(f"Rung 1 (association):  P(downstream=1 | treatment=1)        = {observational:.3f}")
    print(f"Rung 2 (intervention): P(downstream=1 | do(treatment=1))    = {interventional_on:.3f}")
    print(f"Rung 2 (intervention): P(downstream=1 | do(treatment=0))    = {interventional_off:.3f}")
    print(f"Rung 2 (ATE):          E[do(treat=1)] - E[do(treat=0)]      = {ate:.3f}")
    print("Rung 3 (counterfactual): P(downstream_{do(treat=1)}=1 |")
    print(
        "        treatment=0, downstream=0)                          "
        f"= {cf.get(1, float('nan')):.3f}"
    )
    print()
    print(
        f"Observational - interventional divergence (bias)            "
        f"= {observational - interventional_on:+.3f}"
    )
    print()
    print("Identification certificate")
    print("-" * 60)
    print("  estimand:        P(downstream | do(treatment))")
    print("  criterion:       backdoor")
    print(f"  adjustment set:  {sorted(adjustment)}")
    print("  derivation:      sum_z P(downstream | treatment, z) P(z),  z = adjustment set")


if __name__ == "__main__":
    main()
