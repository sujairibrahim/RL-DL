"""
remarl/tests/test_sim/test_oracle.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import pytest
from sim.oracle import Oracle
from sim.re_env import MockWorkspace


@pytest.fixture(scope="module")
def oracle():
    return Oracle(coverage_threshold=0.60)


class GoodWorkspace(MockWorkspace):
    def __init__(self):
        super().__init__()
        self.set("req_draft", """
            The system shall allow users to register and log in.
            The system shall display a product catalogue with search.
            The system shall provide a shopping cart that persists across sessions.
            The system shall process payments via credit card and PayPal.
            The system shall send order confirmation emails to buyers.
            The system shall allow buyers to leave ratings and reviews.
            The system shall provide a seller dashboard with sales analytics.
            The system shall enforce a return and refund workflow.
        """)
        self.set("error_report",
            "Conflict detected: refund timeline between buyer and seller.")


class EmptyWorkspace(MockWorkspace):
    pass


def test_good_workspace_scores_high(oracle, tmp_path):
    from sim.scenario_gen import ScenarioGenerator
    gen = ScenarioGenerator(str(tmp_path))
    scenario = gen.sample(domain="e_commerce_marketplace")
    result = oracle.score(GoodWorkspace(), scenario)
    assert result.total_reward > 0.3, f"Expected >0.3, got {result.total_reward}"
    assert result.coverage_score > 0.3


def test_empty_workspace_scores_zero(oracle, tmp_path):
    from sim.scenario_gen import ScenarioGenerator
    gen = ScenarioGenerator(str(tmp_path))
    scenario = gen.sample()
    result = oracle.score(EmptyWorkspace(), scenario)
    assert result.total_reward == 0.0


def test_covered_plus_missed_equals_ground_truth(oracle, tmp_path):
    from sim.scenario_gen import ScenarioGenerator
    gen = ScenarioGenerator(str(tmp_path))
    scenario = gen.sample()
    result = oracle.score(GoodWorkspace(), scenario)
    total = len(result.covered_reqs) + len(result.missed_reqs)
    assert total == len(scenario.ground_truth_reqs)
