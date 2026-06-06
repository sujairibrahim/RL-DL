"""
remarl/tests/test_mare/test_base_agent.py
------------------------------------------
Tests that the modified BaseAgent is backward-compatible with MARE
and correctly hooks RL policies when provided.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import pytest
from mare.agents.base_agent import BaseAgent
from sim.re_env import MockWorkspace


class MockLLM:
    def call(self, prompt):
        return "The system shall allow users to register."


class MockPromptBuilder:
    def build(self, action_name, workspace, role):
        return f"[PROMPT] {role} {action_name}"


def test_mare_mode_no_policy():
    """policy=None → original MARE behaviour, no RL override."""
    agent = BaseAgent(
        role="collector",
        llm=MockLLM(),
        prompt_builder=MockPromptBuilder(),
        policy=None,
        state_encoder=None,
    )
    ws = MockWorkspace()
    result = agent.perform_action("write_req_draft", ws)
    assert "output" in result
    assert result["rl_action"] is None
    assert "shall" in result["output"]


def test_workspace_updated_after_action():
    """BaseAgent must write output to the correct workspace field."""
    agent = BaseAgent(
        role="collector",
        llm=MockLLM(),
        prompt_builder=MockPromptBuilder(),
    )
    ws = MockWorkspace()
    agent.perform_action("write_req_draft", ws)
    assert ws.get("req_draft") != ""


def test_rl_policy_overrides_action():
    """When policy is active, RL action integer controls which action runs."""
    import numpy as np

    class FakePolicy:
        def predict(self, obs, deterministic=False):
            return np.array([2]), None   # action index 2

    class FakeEncoder:
        state_dim = 10
        def encode(self, ws, step=0):
            return np.zeros(10, dtype=np.float32)

    agent = BaseAgent(
        role="collector",
        llm=MockLLM(),
        prompt_builder=MockPromptBuilder(),
        policy=FakePolicy(),
        state_encoder=FakeEncoder(),
    )
    ws = MockWorkspace()
    result = agent.perform_action("write_req_draft", ws)
    assert result["rl_action"] == 2
