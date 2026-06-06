"""
remarl/mare/agents/base_agent.py
---------------------------------
MARE BaseAgent — modified for REMARL.

Changes from original MARE:
  1. __init__ accepts optional `policy` and `state_encoder` arguments.
  2. perform_action() checks for policy before calling LLM.
  3. Returns (result, state_vec, rl_action) tuple instead of just result.
  4. Backward compatible: policy=None → original MARE behaviour.
"""

import logging
from typing import Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    REMARL-extended MARE BaseAgent.

    When policy=None  → behaves exactly like original MARE.
    When policy!=None → RL policy chooses the action strategy.
    """

    def __init__(
        self,
        role: str,
        llm,
        prompt_builder,
        policy=None,           # ← NEW: SB3 PPO model or None
        state_encoder=None,    # ← NEW: StateEncoder or None
    ):
        self.role          = role
        self.llm           = llm
        self.prompt_builder = prompt_builder
        self.policy        = policy
        self.state_encoder = state_encoder

    def perform_action(
        self,
        action_name: str,
        workspace,
    ) -> dict:
        """
        Execute an action. If RL policy is active, it may override
        the action_name with a strategically chosen alternative.

        Returns:
            dict with at least {"output": str, "action_used": str}
        """
        state_vec  = None
        rl_action  = None

        # ── RL policy override ────────────────────────────────────────
        if self.policy is not None and self.state_encoder is not None:
            state_vec = self.state_encoder.encode(workspace)
            rl_action, _ = self.policy.predict(
                state_vec.reshape(1, -1), deterministic=False
            )
            rl_action = int(rl_action)
            action_name = self._map_rl_action(rl_action, action_name)
            logger.debug(f"[{self.role}] RL chose action={rl_action} → {action_name}")

        # ── Build prompt and call LLM ─────────────────────────────────
        prompt = self.prompt_builder.build(action_name, workspace, self.role)
        raw_output = self.llm.call(prompt)

        # ── Update workspace ──────────────────────────────────────────
        self._update_workspace(action_name, raw_output, workspace)

        result = {
            "output":      raw_output,
            "action_used": action_name,
            "rl_action":   rl_action,
            "state_vec":   state_vec,
            "role":        self.role,
        }
        return result

    def _map_rl_action(self, rl_action_int: int, default: str) -> str:
        """
        Map RL action integer to a MARE action name.
        Override in subclasses for role-specific mappings.
        Default: return the passed action_name unchanged.
        """
        # Import here to avoid circular dependency
        try:
            from sim.re_env import AGENT_ACTION_MAP
            actions = AGENT_ACTION_MAP.get(self.role, [])
            if actions and 0 <= rl_action_int < len(actions):
                return actions[rl_action_int]
        except ImportError:
            pass
        return default

    def _update_workspace(self, action_name: str, output: str, workspace):
        """
        Write LLM output to the correct workspace field.
        Subclasses should override for precise field targeting.
        """
        field_map = {
            "speak_user_stories":      "user_stories",
            "propose_question":        "questions",
            "write_req_draft":         "req_draft",
            "refine_req_draft":        "req_draft",
            "extract_entity":          "req_model",
            "extract_relation":        "req_model",
            "build_use_case":          "req_model",
            "check_completeness":      "error_report",
            "check_consistency":       "error_report",
            "write_final_srs":         "srs_document",
            "approve_and_document":    "srs_document",
        }
        field = field_map.get(action_name)
        if field:
            existing = workspace.get(field, "")
            workspace.set(field, existing + "\n" + output if existing else output)
