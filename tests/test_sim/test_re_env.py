"""
remarl/tests/test_sim/test_re_env.py
-------------------------------------
Integration test: full episode rollout through RESimEnv.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
import pytest


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Build a minimal RESimEnv with stub components."""
    from sim.scenario_gen import ScenarioGenerator
    from sim.oracle import Oracle
    from sim.re_env import RESimEnv, AGENT_ACTION_MAP
    from rl.state_encoder import StateEncoder
    from rl.reward import RewardEngine

    d = tmp_path_factory.mktemp("scenarios")

    class StubAgent:
        def perform_action(self, action_name, workspace):
            workspace.set(
                "req_draft",
                workspace.get("req_draft", "") +
                f"\nThe system shall support {action_name.replace('_', ' ')}."
            )
            if action_name in ("write_final_srs", "approve_and_document"):
                workspace.set("srs_document", workspace.get("req_draft", ""))
            return {"output": f"executed {action_name}"}

    return RESimEnv(
        scenario_gen=ScenarioGenerator(str(d)),
        oracle=Oracle(),
        state_encoder=StateEncoder(),
        reward_engine=RewardEngine(),
        agents={r: StubAgent() for r in AGENT_ACTION_MAP},
        agent_role="collector",
        max_steps=12,
        verbose=False,
    )


def test_reset_returns_correct_shape(env):
    obs, info = env.reset()
    assert obs.dtype == np.float32
    assert obs.shape == (env.observation_space.shape[0],)
    assert "domain" in info
    assert "scenario_id" in info


def test_step_returns_valid_tuple(env):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (env.observation_space.shape[0],)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "action_name" in info


def test_episode_terminates(env):
    """Episode must end within max_steps."""
    env.reset()
    done = False
    steps = 0
    while not done and steps < 50:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
    assert done, "Episode never terminated"
    assert steps <= env.max_steps + 1


def test_gymnasium_check(tmp_path):
    """Official SB3 environment checker must pass."""
    from stable_baselines3.common.env_checker import check_env
    from sim.scenario_gen import ScenarioGenerator
    from sim.oracle import Oracle
    from sim.re_env import RESimEnv, AGENT_ACTION_MAP
    from rl.state_encoder import StateEncoder
    from rl.reward import RewardEngine

    class StubAgent:
        def perform_action(self, action_name, workspace):
            workspace.set(
                "req_draft",
                workspace.get("req_draft", "") +
                f"\nThe system shall support {action_name.replace('_', ' ')}."
            )
            if action_name in ("write_final_srs", "approve_and_document"):
                workspace.set("srs_document", workspace.get("req_draft", ""))
            return {"output": f"executed {action_name}"}

    test_env = RESimEnv(
        scenario_gen=ScenarioGenerator(str(tmp_path)),
        oracle=Oracle(),
        state_encoder=StateEncoder(),
        reward_engine=RewardEngine(),
        agents={r: StubAgent() for r in AGENT_ACTION_MAP},
        agent_role="collector",
        max_steps=8,
    )
    check_env(test_env, warn=True)   # raises AssertionError if invalid
