"""
remarl/tests/test_rl/test_state_encoder.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from rl.state_encoder import StateEncoder, STATE_DIM
from sim.re_env import MockWorkspace


@pytest.fixture(scope="module")
def encoder():
    return StateEncoder()


def test_shape(encoder):
    ws = MockWorkspace()
    ws.set("req_draft", "The system shall allow login.")
    v = encoder.encode(ws, step=3)
    assert v.shape == (STATE_DIM,)


def test_dtype(encoder):
    ws = MockWorkspace()
    v = encoder.encode(ws)
    assert v.dtype == np.float32


def test_no_nan(encoder):
    ws = MockWorkspace()
    ws.set("user_stories", "As a user I want to log in.")
    ws.set("req_draft",    "The system shall authenticate users.")
    v = encoder.encode(ws, step=5)
    assert not np.any(np.isnan(v))
    assert not np.any(np.isinf(v))


def test_different_workspaces_differ(encoder):
    ws1, ws2 = MockWorkspace(), MockWorkspace()
    ws1.set("req_draft", "The system shall allow login.")
    ws2.set("req_draft", "The system shall display a map of all locations.")
    v1 = encoder.encode(ws1)
    v2 = encoder.encode(ws2)
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    assert cosine < 0.999, "Different workspaces should produce different state vectors"


def test_phase_one_hot(encoder):
    ws = MockWorkspace()
    ws.current_phase = 2   # verification phase
    v = encoder.encode(ws)
    # Phase one-hot lives at indices 1536:1541
    phase_vec = v[1536:1541]
    assert phase_vec[2] == 1.0
    assert phase_vec.sum() == 1.0
