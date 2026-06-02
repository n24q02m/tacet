"""WorldModel interface — the seam between TACET and a continuous predictor.

TACET is the *discrete, sound, editable* side of an agent's mind: it
records what is true (`WorldGraph`), what is entailed (`RuleEngine`), what
is statistically regular (`ComplEx`), and what happened (`EpisodicStore`).
The other side is the *continuous, learned, predictive* model that maps
perception to future state — Dreamer / JEPA / Sora-class systems
\\citep{ha2018worldmodels, lecun2022jepa}. TACET does not provide one;
the trade-off in §8.2 of the paper is precisely why it does not.

What this module *does* provide is the **seam**: a small Protocol every
external continuous world model can satisfy, plus a deterministic
``IdentityWorldModel`` for tests / demos / agents that do not yet need
imagined rollouts.

For a real integration: implement ``observe`` and ``rollout`` using your
JEPA-encoded latent and dynamics network, return a ``Trajectory`` per
candidate plan, and let ``tacet.experimental.agent.Planner`` choose the plan with the
best TACET-evaluated goal satisfaction on the predicted trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

S = TypeVar("S")  # state type (latent vector, image batch, …)
A = TypeVar("A")  # action type (vector, discrete enum, …)


@dataclass
class Trajectory(Generic[S, A]):
    """A predicted future under a sequence of actions."""

    states: list[S] = field(default_factory=list)
    actions: list[A] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)


class WorldModel(Protocol[S, A]):
    """The interface a continuous world model must satisfy to plug into TACET.

    All three operations may be stochastic; ``rollout`` returns a single
    sample. A real implementation can additionally cache, batch over
    multiple plans, or return distributions.
    """

    def observe(self, observation: object) -> S: ...
    def predict(self, state: S, action: A) -> S: ...
    def rollout(self, state: S, plan: list[A]) -> Trajectory[S, A]: ...


class IdentityWorldModel:
    """Trivial deterministic stand-in: state never changes. Use for unit tests
    and for agents whose value function does not depend on imagined dynamics."""

    def observe(self, observation: object) -> object:
        return observation

    def predict(self, state: object, action: object) -> object:
        return state

    def rollout(self, state: object, plan: list[object]) -> Trajectory:
        return Trajectory(
            states=[state] * (len(plan) + 1), actions=list(plan), rewards=[0.0] * len(plan)
        )


__all__ = ["IdentityWorldModel", "Trajectory", "WorldModel"]
