"""
remarl/tests/test_sim/test_scenario_gen.py
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import pytest
from sim.scenario_gen import ScenarioGenerator, Scenario


@pytest.fixture(scope="module")
def gen(tmp_path_factory):
    d = tmp_path_factory.mktemp("scenarios")
    return ScenarioGenerator(str(d))


def test_sample_returns_scenario(gen):
    s = gen.sample()
    assert isinstance(s, Scenario)
    assert s.rough_idea
    assert len(s.ground_truth_reqs) >= 5
    assert len(s.hidden_reqs) >= 1
    assert len(s.visible_reqs) >= 1


def test_hidden_plus_visible_equals_ground_truth(gen):
    for _ in range(10):
        s = gen.sample()
        assert set(s.hidden_reqs) | set(s.visible_reqs) == set(s.ground_truth_reqs)
        assert set(s.hidden_reqs) & set(s.visible_reqs) == set()


def test_domain_filter(gen):
    s = gen.sample(domain="e_commerce_marketplace")
    assert s.domain == "e_commerce_marketplace"


def test_difficulty_filter(gen):
    s = gen.sample(difficulty="easy")
    assert s.difficulty == "easy"


def test_stats(gen):
    stats = gen.stats()
    assert stats["total"] > 0
    assert stats["avg_reqs"] > 0


def test_batch(gen):
    batch = gen.sample_batch(n=5)
    assert len(batch) == 5
    assert all(isinstance(s, Scenario) for s in batch)
