"""
remarl/mare/pipeline.py
-----------------------
MARE pipeline — modified for REMARL.

Changes from original MARE:
  1. Accepts optional `policies` and `state_encoder` dicts.
  2. Wraps every action step to collect (state, action, reward, next_state).
  3. Calls reward_engine after each step.
  4. Returns (experiences, final_workspace) for the training loop.
  5. Backward compatible: if no policies passed → original MARE behaviour.
"""

import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# MARE's canonical action sequence.
# (agent_role, action_name) pairs executed in order.
# REMARL policies may deviate from the action_name via RL override.
DEFAULT_ACTION_SEQUENCE = [
    ("stakeholder", "speak_user_stories"),
    ("collector",   "propose_question"),
    ("collector",   "propose_question"),      # second round of questions
    ("collector",   "write_req_draft"),
    ("modeler",     "extract_entity"),
    ("modeler",     "extract_relation"),
    ("modeler",     "build_use_case"),
    ("checker",     "check_completeness"),
    ("checker",     "check_consistency"),
    ("negotiator",  "accept_requirement"),    # ← ADD: negotiate conflicts
    ("negotiator",  "modify_priority"),       # ← ADD: prioritize requirements
    ("documenter",  "write_final_srs"),
]


class REMARLPipeline:
    """
    Runs the full MARE action sequence with optional RL policy overrides.

    Args:
        scenario:      Scenario object with rough_idea and ground_truth_reqs
        agents:        dict mapping role → BaseAgent instance
        workspace:     SharedWorkspace (or MockWorkspace)
        reward_engine: RewardEngine for immediate rewards
        state_encoder: StateEncoder — required if policies provided
        policies:      dict mapping role → trained PPO model (optional)
        action_sequence: list of (role, action) pairs to execute
    """

    def __init__(
        self,
        scenario,
        agents: dict,
        workspace,
        reward_engine,
        state_encoder=None,
        policies: Optional[Dict] = None,
        action_sequence: Optional[List] = None,
    ):
        self.scenario       = scenario
        self.agents         = agents
        self.workspace      = workspace
        self.reward_engine  = reward_engine
        self.state_encoder  = state_encoder
        self.policies       = policies or {}
        self.action_sequence = action_sequence or DEFAULT_ACTION_SEQUENCE

        # Load scenario into workspace
        self.workspace.set("rough_idea",        scenario.rough_idea)
        self.workspace.set("domain",            scenario.domain)
        if scenario.visible_reqs:
            self.workspace.set(
                "initial_context",
                "Known requirements:\n" +
                "\n".join(f"- {r}" for r in scenario.visible_reqs)
            )

    def run(self) -> tuple:
        """
        Execute the full pipeline.

        Returns:
            (experiences: List[dict], final_workspace)
            experiences contains one entry per action step.
        """
        experiences = []

        for step_idx, (role, default_action) in enumerate(self.action_sequence):
            agent = self.agents.get(role)
            if agent is None:
                logger.warning(f"No agent registered for role '{role}', skipping.")
                continue

            # ── Encode state before action ────────────────────────────
            state_vec = None
            if self.state_encoder:
                state_vec = self.state_encoder.encode(
                    self.workspace, step=step_idx
                )

            # ── Inject RL policy if available for this role ───────────
            if role in self.policies and self.policies[role] is not None:
                agent.policy        = self.policies[role]
                agent.state_encoder = self.state_encoder

            # ── Execute action ────────────────────────────────────────
            result = agent.perform_action(default_action, self.workspace)

            # ── Score immediate reward ────────────────────────────────
            immediate_reward = self.reward_engine.score_immediate(
                result, self.workspace
            )

            # ── Encode next state ─────────────────────────────────────
            next_state_vec = None
            if self.state_encoder:
                next_state_vec = self.state_encoder.encode(
                    self.workspace, step=step_idx + 1
                )

            experiences.append({
                "step":             step_idx,
                "role":             role,
                "action":           result.get("action_used", default_action),
                "rl_action":        result.get("rl_action"),
                "immediate_reward": immediate_reward,
                "state":            state_vec,
                "next_state":       next_state_vec,
                "output_length":    len(result.get("output", "")),
            })

            logger.debug(
                f"Step {step_idx:2d} | {role:<12} | "
                f"{result.get('action_used', default_action):<30} | "
                f"r={immediate_reward:+.4f}"
            )

        return experiences, self.workspace
