"""
remarl/rl/reward.py
-------------------
Immediate reward signals computed after each agent action.
Terminal (oracle) reward is computed separately in sim/oracle.py.

Immediate reward = clarity + consistency + coverage_delta
Each component in [-1, 1]; total clipped to [-1, 1].
"""

import re
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Words that signal ambiguity in requirements
AMBIGUOUS_WORDS = [
    "may", "might", "could", "should consider", "etc", "and/or",
    "appropriate", "as necessary", "user-friendly", "fast", "efficient",
    "some", "many", "usually", "normally", "ideally",
]

# Words that signal a well-formed requirement
GOOD_REQ_WORDS = [
    "shall", "must", "will", "the system", "the user", "the application",
]


class RewardEngine:
    """
    Computes immediate reward after each agent action.

    Args:
        clarity_weight:        weight for text clarity score
        consistency_weight:    weight for inter-requirement consistency
        coverage_delta_weight: weight for coverage increase
    """

    def __init__(
        self,
        clarity_weight: float = 0.30,
        consistency_weight: float = 0.40,
        coverage_delta_weight: float = 0.30,
        coverage_threshold: float = 0.65,
    ):
        self.w_clar  = clarity_weight
        self.w_cons  = consistency_weight
        self.w_cov   = coverage_delta_weight
        self.threshold = coverage_threshold
        self._encoder = None   # lazy: sentence-transformers

    def score_immediate(self, action_result, workspace) -> float:
        """
        Score one agent action.

        Args:
            action_result: dict with at least {"output": str}
            workspace:     current SharedWorkspace

        Returns:
            float in [-1, 1]
        """
        output = ""
        if isinstance(action_result, dict):
            output = action_result.get("output", "")
        elif isinstance(action_result, str):
            output = action_result

        if not output or not output.strip():
            return -0.05   # small penalty for empty output

        clarity     = self._clarity(output)
        consistency = self._consistency(output, workspace)
        cov_delta   = self._coverage_delta(output, workspace)

        total = (
            self.w_clar * clarity  +
            self.w_cons * consistency +
            self.w_cov  * cov_delta
        )
        return float(np.clip(total, -1.0, 1.0))

    # ── Component scorers ────────────────────────────────────────────

    def _clarity(self, text: str) -> float:
        """
        Penalise ambiguous language, reward shall-statements.
        Returns [-1, 1].
        """
        text_lower = text.lower()
        word_count = max(len(text.split()), 1)

        ambiguous_count = sum(
            text_lower.count(w) for w in AMBIGUOUS_WORDS
        )
        good_count = sum(
            text_lower.count(w) for w in GOOD_REQ_WORDS
        )

        # Normalise per 100 words
        ambiguity_rate = (ambiguous_count / word_count) * 100
        goodness_rate  = (good_count  / word_count) * 100

        score = min(1.0, goodness_rate * 0.2) - min(1.0, ambiguity_rate * 0.15)
        return float(np.clip(score, -1.0, 1.0))

    def _consistency(self, new_text: str, workspace) -> float:
        """
        Check new text does not contradict existing requirements.
        Uses NLI-style keyword heuristic (fast, no extra model needed).
        For a stronger version, swap in a cross-encoder NLI model.
        Returns [-1, 1].
        """
        existing = workspace.get("req_draft", "")
        if not existing.strip():
            return 0.0   # nothing to contradict yet

        # Contradiction signals: direct negation pairs
        CONTRADICTION_PAIRS = [
            ("shall allow",   "shall not allow"),
            ("shall require", "shall not require"),
            ("must",          "must not"),
            ("always",        "never"),
            ("mandatory",     "optional"),
        ]

        new_lower = new_text.lower()
        existing_lower = existing.lower()
        contradictions = 0

        for pos, neg in CONTRADICTION_PAIRS:
            new_has_pos = pos in new_lower
            new_has_neg = neg in new_lower
            old_has_pos = pos in existing_lower
            old_has_neg = neg in existing_lower

            if (new_has_pos and old_has_neg) or (new_has_neg and old_has_pos):
                contradictions += 1

        if contradictions == 0:
            return 0.4   # no contradiction = small positive
        elif contradictions == 1:
            return -0.3  # one contradiction = moderate penalty
        else:
            return -0.8  # multiple contradictions = severe penalty

    def _coverage_delta(self, new_text: str, workspace) -> float:
        """
        Reward adding new meaningful content to the requirement draft.
        Measures how much new semantic content was introduced.
        Returns [0, 1].
        """
        existing = workspace.get("req_draft", "")

        # Fast proxy: count new "shall" statements introduced
        new_shalls    = new_text.lower().count("shall")
        total_shalls  = existing.lower().count("shall")

        if total_shalls == 0:
            # First contribution — reward it
            return min(1.0, new_shalls * 0.2)

        # Reward proportional increase in requirement count
        delta_ratio = new_shalls / max(total_shalls, 1)
        return float(np.clip(delta_ratio * 0.5, 0.0, 1.0))
