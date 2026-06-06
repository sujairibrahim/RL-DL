"""
remarl/tests/test_rl/test_reward.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import pytest
from rl.reward import RewardEngine
from sim.re_env import MockWorkspace


@pytest.fixture
def engine():
    return RewardEngine()


def _make_result(text: str) -> dict:
    return {"output": text, "action_used": "write_req_draft"}


def test_good_requirement_scores_positive(engine):
    ws = MockWorkspace()
    result = _make_result(
        "The system shall allow users to register and log in securely."
    )
    r = engine.score_immediate(result, ws)
    assert r > 0, f"Expected positive reward, got {r}"


def test_ambiguous_requirement_scores_lower(engine):
    ws = MockWorkspace()
    good   = _make_result("The system shall process payments within 2 seconds.")
    vague  = _make_result("The system should maybe consider some kind of fast payment processing etc.")
    r_good  = engine.score_immediate(good, ws)
    r_vague = engine.score_immediate(vague, ws)
    assert r_good > r_vague, f"Good req ({r_good:.3f}) should outscore vague ({r_vague:.3f})"


def test_empty_output_penalised(engine):
    ws = MockWorkspace()
    r = engine.score_immediate({"output": ""}, ws)
    assert r < 0


def test_contradiction_penalised(engine):
    ws = MockWorkspace()
    ws.set("req_draft", "The system shall allow guest checkout.")
    contradicting = _make_result(
        "The system shall not allow guest checkout; registration is mandatory."
    )
    r = engine.score_immediate(contradicting, ws)
    # Should be lower than adding consistent content
    consistent = _make_result("The system shall send a confirmation email.")
    r_consistent = engine.score_immediate(consistent, ws)
    assert r < r_consistent


def test_reward_clipped_to_range(engine):
    ws = MockWorkspace()
    result = _make_result("shall " * 50)   # extreme good signal
    r = engine.score_immediate(result, ws)
    assert -1.0 <= r <= 1.0
