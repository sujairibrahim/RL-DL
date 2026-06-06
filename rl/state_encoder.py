"""
remarl/rl/state_encoder.py
--------------------------
Converts a MARE SharedWorkspace into a fixed-size float32 vector
that the RL policy network can consume.

State vector layout (total: 1544 dims):
  [0:384]    embedding of user_stories text
  [384:768]  embedding of req_draft text
  [768:1152] embedding of req_model text
  [1152:1536] embedding of error_report text
  [1536:1541] phase one-hot (5 phases)
  [1541:1544] progress scalars: step_ratio, req_count_norm, error_flag
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

PHASE_NAMES = ["elicitation", "modeling", "verification", "specification", "done"]
EMBED_DIM   = 384   # all-MiniLM-L6-v2 output dimension
N_TEXT_FIELDS = 4
SCALAR_DIMS   = 3
STATE_DIM = EMBED_DIM * N_TEXT_FIELDS + len(PHASE_NAMES) + SCALAR_DIMS  # 1544


class StateEncoder:
    """
    Encodes a SharedWorkspace into a numpy observation vector.

    Args:
        model_name: sentence-transformers model name.
        max_steps:  used to normalise step counter into [0,1].
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        max_steps: int = 24,
    ):
        self._model_name = model_name
        self._model = None          # lazy load
        self.max_steps = max_steps
        self.state_dim = STATE_DIM

    def encode(self, workspace, step: int = 0) -> np.ndarray:
        """
        Encode workspace → (state_dim,) float32 vector.

        Args:
            workspace: SharedWorkspace or any object with .get(key, default)
            step:      current episode step count (for progress scalar)
        """
        model = self._get_model()

        # ── Text fields → embeddings ──────────────────────────────────
        text_fields = [
            workspace.get("user_stories",  ""),
            workspace.get("req_draft",      ""),
            workspace.get("req_model",      ""),
            workspace.get("error_report",   ""),
        ]
        # Replace empty strings with a neutral placeholder
        text_fields = [t if t.strip() else "[empty]" for t in text_fields]
        embeddings = model.encode(
            text_fields,
            batch_size=4,
            show_progress_bar=False,
            normalize_embeddings=True,  # unit vectors → stable cosine space
        )  # shape: (4, 384)
        flat_embeddings = embeddings.flatten()  # (1536,)

        # ── Phase one-hot ─────────────────────────────────────────────
        phase_idx = getattr(workspace, "current_phase", 0)
        phase_idx = int(np.clip(phase_idx, 0, len(PHASE_NAMES) - 1))
        phase_vec = np.zeros(len(PHASE_NAMES), dtype=np.float32)
        phase_vec[phase_idx] = 1.0

        # ── Progress scalars ──────────────────────────────────────────
        step_ratio = float(step) / float(self.max_steps)

        req_draft = workspace.get("req_draft", "")
        # Rough requirement count: number of "shall" occurrences
        req_count = min(req_draft.lower().count("shall"), 20) / 20.0

        error_flag = 1.0 if workspace.get("error_report", "").strip() else 0.0

        scalars = np.array([step_ratio, req_count, error_flag], dtype=np.float32)

        # ── Concatenate ───────────────────────────────────────────────
        state = np.concatenate([
            flat_embeddings.astype(np.float32),
            phase_vec,
            scalars,
        ])  # (1544,)

        assert state.shape == (STATE_DIM,), f"Expected ({STATE_DIM},), got {state.shape}"
        return state

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info(f"Loaded encoder model: {self._model_name}")
        return self._model


if __name__ == "__main__":
    from sim.re_env import MockWorkspace
    ws = MockWorkspace()
    ws.set("user_stories", "As a user I want to log in.")
    ws.set("req_draft", "The system shall allow users to authenticate.")
    enc = StateEncoder()
    v = enc.encode(ws, step=3)
    print(f"State shape : {v.shape}")
    print(f"Min/Max/Mean: {v.min():.3f} / {v.max():.3f} / {v.mean():.3f}")
    assert v.shape == (STATE_DIM,)
    assert not np.any(np.isnan(v))
    print("StateEncoder OK")
