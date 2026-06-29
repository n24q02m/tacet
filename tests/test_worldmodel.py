from tacet.experimental.worldmodel import IdentityWorldModel, Trajectory


def test_trajectory_init():
    states = [1, 2, 3]
    actions = ["a", "b"]
    rewards = [0.1, 0.2]
    traj = Trajectory(states=states, actions=actions, rewards=rewards)
    assert traj.states == states
    assert traj.actions == actions
    assert traj.rewards == rewards


def test_identity_world_model_observe():
    wm = IdentityWorldModel()
    obs = {"key": "val"}
    assert wm.observe(obs) == obs


def test_identity_world_model_predict():
    wm = IdentityWorldModel()
    state = "state"
    action = "action"
    assert wm.predict(state, action) == state


def test_identity_world_model_rollout():
    wm = IdentityWorldModel()
    state = "initial_state"
    plan = ["a1", "a2", "a3"]
    traj = wm.rollout(state, plan)

    assert isinstance(traj, Trajectory)
    assert traj.states == [state, state, state, state]
    assert traj.actions == plan
    assert traj.rewards == [0.0, 0.0, 0.0]
