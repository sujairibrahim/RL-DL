"""
remarl/sim/re_env.py
--------------------
Gymnasium-compatible simulation environment for REMARL training.

Each episode = one complete Requirements Engineering project:
  reset()  → generate a new synthetic scenario, initialise workspace
  step()   → execute one agent action, return (obs, reward, done, info)
  render() → print workspace state (for debugging)

The environment is designed to work with Stable-Baselines3's PPO.
Run `check_env(RESimEnv(config))` to validate compatibility.

State space (observation):
  - Sentence embedding of current workspace contents (1536 dims)
  - Phase one-hot vector (5 dims)
  - Progress counters: steps taken, errors found, reqs count (3 dims)
  Total: 1544-dimensional float32 vector

Action space:
  - Discrete(4): four strategic choices per agent role
  - The mapping from action int → MARE action is agent-role-specific

Reward:
  - Immediate: clarity + consistency + coverage delta after each action
  - Terminal: oracle score (coverage, precision, conflict, nfr)
  - Step penalty: -0.01 per step to encourage efficiency
"""

import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Action space definitions
# ─────────────────────────────────────────────

# Each agent role has exactly 4 discrete actions.
# The RL policy outputs an integer 0-3.
# The environment maps it to a MARE action name.

AGENT_ACTION_MAP: Dict[str, list] = {
    "stakeholder": [
        "speak_user_stories",          # 0: emit user stories from scenario idea
        "speak_detailed_user_stories",  # 1: add more detail to user stories
        "clarify_ambiguity",            # 2: add clarification notes
        "add_nfr_context",              # 3: provide non-functional context
    ],
    "collector": [
        "propose_question",            # 0: ask stakeholder a clarifying question
        "write_req_draft",             # 1: commit current understanding to draft
        "refine_req_draft",            # 2: improve existing draft entries
        "flag_missing_coverage",       # 3: signal gaps to stakeholder
    ],
    "modeler": [
        "extract_entity",              # 0: extract domain entities
        "extract_relation",            # 1: extract relationships
        "build_use_case",              # 2: build use case model
        "flag_modeling_inconsistency", # 3: flag model issues
    ],
    "checker": [
        "check_completeness",          # 0: run completeness check
        "check_consistency",           # 1: run consistency check
        "approve_and_document",        # 2: approve — moves to Documenter
        "request_revision",            # 3: send back with error report
    ],
    "documenter": [
        "write_srs_section",           # 0: write a section of the SRS
        "refine_srs_section",          # 1: improve existing section
        "write_final_srs",             # 2: produce final SRS document
        "add_traceability_matrix",     # 3: add req-to-test mapping
    ],
    "negotiator": [
        "accept_requirement",    # 0: resolve conflict by accepting both
        "reject_requirement",    # 1: resolve by removing one requirement
        "modify_priority",       # 2: re-prioritize to resolve tension
        "defer_to_next_sprint",  # 3: defer lower-priority conflicting req
    ],
}

# MARE's natural action sequence (REMARL can deviate from this)
DEFAULT_PHASE_SEQUENCE = [
    ("stakeholder", "speak_user_stories"),
    ("collector",   "propose_question"),
    ("collector",   "write_req_draft"),
    ("modeler",     "extract_entity"),
    ("modeler",     "extract_relation"),
    ("checker",     "check_completeness"),
    ("checker",     "check_consistency"),
    ("negotiator",  "accept_requirement"),    # ← ADD
    ("negotiator",  "modify_priority"),       # ← ADD
    ("documenter",  "write_final_srs"),
]


# ─────────────────────────────────────────────
#  RESimEnv
# ─────────────────────────────────────────────

class RESimEnv(gym.Env):
    """
    Full REMARL simulation environment.

    Args:
        scenario_gen:    ScenarioGenerator instance
        oracle:          Oracle instance for terminal reward
        state_encoder:   StateEncoder instance
        reward_engine:   RewardEngine instance for immediate rewards
        agents:          dict mapping role → MAREAgent instance
        agent_role:      which agent this env is training ("collector" etc.)
                         Set to "multi" to train all agents jointly.
        max_steps:       maximum actions per episode before forced termination
        step_penalty:    negative reward per step to encourage efficiency
        verbose:         print episode summary if True
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        scenario_gen,
        oracle,
        state_encoder,
        reward_engine,
        agents: dict,
        agent_role: str = "collector",
        max_steps: int = 24,
        step_penalty: float = 0.01,
        verbose: bool = False,
    ):
        super().__init__()

        self.scenario_gen = scenario_gen
        self.oracle = oracle
        self.state_encoder = state_encoder
        self.reward_engine = reward_engine
        self.agents = agents
        self.agent_role = agent_role
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.verbose = verbose

        # ── Gymnasium spaces ─────────────────────────────────────────
        obs_dim = self.state_encoder.state_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        # ── Episode state ─────────────────────────────────────────────
        self._scenario = None
        self._workspace = None
        self._step_count = 0
        self._phase_idx = 0
        self._episode_experiences = []
        self._last_obs = None

        logger.info(
            f"RESimEnv initialised | agent_role={agent_role} "
            f"obs_dim={obs_dim} max_steps={max_steps}"
        )

    # ── Gymnasium API ─────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        scenario=None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Start a new episode.
        Samples a fresh scenario, initialises a blank workspace,
        loads the scenario's rough idea into the workspace.

        Pass scenario= to replay an exact Scenario object (e.g. for paired
        evaluation where REMARL and baseline must use identical hidden_reqs).
        """
        super().reset(seed=seed)
        # ── NEW: Reset all agent conversation histories ───────────────
        # This prevents conversation history from accumulating across
        # episodes, which causes exponentially growing context and
        # 70+ second Ollama calls by episode 13.
        for agent in self.agents.values():
            if hasattr(agent, "reset"):
                agent.reset()
        # Sample new scenario — or replay an externally-provided one
        if scenario is not None:
            self._scenario = scenario
        else:
            domain = options.get("domain") if options else None
            difficulty = options.get("difficulty") if options else None
            self._scenario = self.scenario_gen.sample(domain=domain, difficulty=difficulty)

        # Fresh workspace
        self._workspace = self._create_workspace()
        self._workspace.set("rough_idea", self._scenario.rough_idea)
        self._workspace.set("domain", self._scenario.domain)
        # Visible requirements are pre-loaded (hidden ones must be elicited)
        if self._scenario.visible_reqs:
            self._workspace.set(
                "initial_context",
                "Initial requirements mentioned by client:\n" +
                "\n".join(f"- {r}" for r in self._scenario.visible_reqs)
            )

        # Reset counters
        self._step_count = 0
        self._phase_idx = 0
        self._episode_experiences = []

        obs = self._get_obs()
        self._last_obs = obs

        info = {
            "scenario_id": self._scenario.scenario_id,
            "domain": self._scenario.domain,
            "difficulty": self._scenario.difficulty,
            "n_hidden_reqs": len(self._scenario.hidden_reqs),
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one agent action.

        Args:
            action: integer 0-3, mapped to agent-specific action name

        Returns:
            obs, reward, terminated, truncated, info
        """
        if self._scenario is None:
            raise RuntimeError("Call reset() before step().")

        # ── Map action int → MARE action name ────────────────────────
        action_name = self._map_action(action)
        agent_role = self._get_current_agent_role()
        agent = self.agents.get(agent_role)

        # ── Execute via MARE agent ────────────────────────────────────
        try:
            result = agent.perform_action(action_name, self._workspace)
        except Exception as e:
            logger.warning(f"Agent action failed: {e}")
            result = {"error": str(e), "output": ""}

        # ── Immediate reward ──────────────────────────────────────────
        immediate_reward = self.reward_engine.score_immediate(
            result, self._workspace
        )
        step_reward = immediate_reward - self.step_penalty

        # ── Advance phase ─────────────────────────────────────────────
        self._advance_phase_if_ready(action_name)
        self._step_count += 1

        # ── Check termination ─────────────────────────────────────────
        terminated = self._is_done(action_name)
        truncated = self._step_count >= self.max_steps

        # ── Terminal reward from oracle ───────────────────────────────
        terminal_reward = 0.0
        oracle_result = None
        if terminated or truncated:
            oracle_result = self.oracle.score(self._workspace, self._scenario)
            terminal_reward = oracle_result.total_reward
            if self.verbose:
                self._print_episode_summary(oracle_result)

        total_reward = step_reward + terminal_reward

        # ── Store experience ──────────────────────────────────────────
        obs = self._get_obs()
        self._episode_experiences.append({
            "step": self._step_count,
            "agent_role": agent_role,
            "action_name": action_name,
            "action_int": action,
            "immediate_reward": immediate_reward,
            "terminal_reward": terminal_reward,
            "total_reward": total_reward,
        })
        self._last_obs = obs

        info = {
            "step": self._step_count,
            "agent_role": agent_role,
            "action_name": action_name,
            "immediate_reward": round(immediate_reward, 4),
            "terminal_reward": round(terminal_reward, 4),
            "phase": self._get_current_phase_name(),
            "oracle_result": oracle_result,
            "experiences": self._episode_experiences,
        }

        return obs, float(total_reward), terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[str]:
        """Print current workspace state for debugging."""
        if self._workspace is None:
            return "No active episode."

        lines = [
            f"── RESimEnv Episode State ──",
            f"Step    : {self._step_count}/{self.max_steps}",
            f"Phase   : {self._get_current_phase_name()}",
            f"Domain  : {self._scenario.domain if self._scenario else 'N/A'}",
            f"Req draft length : {len(self._workspace.get('req_draft', ''))} chars",
            f"SRS length       : {len(self._workspace.get('srs_document', ''))} chars",
            f"Error report     : {'YES' if self._workspace.get('error_report') else 'NO'}",
        ]
        output = "\n".join(lines)
        if mode == "human":
            print(output)
        return output

    def close(self):
        """Clean up resources."""
        pass

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Encode current workspace + phase into observation vector."""
        obs = self.state_encoder.encode(self._workspace)
        return obs.astype(np.float32)

    def _map_action(self, action: int) -> str:
        """Map action integer → MARE action name for current agent."""
        role = self._get_current_agent_role()
        actions = AGENT_ACTION_MAP.get(role, AGENT_ACTION_MAP["collector"])
        idx = int(np.clip(action, 0, len(actions) - 1))
        return actions[idx]

    def _get_current_agent_role(self) -> str:
        """Which agent should act at the current phase index?"""
        if self.agent_role != "multi":
            return self.agent_role
        # Multi-agent mode: cycle through phase sequence
        phase_roles = [p[0] for p in DEFAULT_PHASE_SEQUENCE]
        idx = self._phase_idx % len(phase_roles)
        return phase_roles[idx]

    def _get_current_phase_name(self) -> str:
        idx = self._phase_idx % len(DEFAULT_PHASE_SEQUENCE)
        return f"{DEFAULT_PHASE_SEQUENCE[idx][0]}.{DEFAULT_PHASE_SEQUENCE[idx][1]}"

    def _advance_phase_if_ready(self, last_action: str):
        """
        Move to the next phase if the current action suggests completion.
        This is a simplified phase transition — in production, the Checker
        agent's approve/request_revision action drives transitions.
        """
        # Documenter writing the final SRS always marks phase complete
        if last_action in ("write_final_srs", "approve_and_document"):
            self._phase_idx = len(DEFAULT_PHASE_SEQUENCE) - 1
            return

        # Normal progression: advance after 2 steps per phase
        steps_this_phase = self._step_count % 3
        if steps_this_phase == 0:
            self._phase_idx = min(
                self._phase_idx + 1,
                len(DEFAULT_PHASE_SEQUENCE) - 1
            )

    def _is_done(self, last_action: str) -> bool:
        """Episode ends when the SRS is written or checker approves."""
        if last_action in ("write_final_srs", "approve_and_document"):
            srs = self._workspace.get("srs_document", "")
            if len(srs.strip()) > 100:
                return True
        return False

    def _create_workspace(self):
        """
        Create a fresh SharedWorkspace.
        Tries to import MARE's SharedWorkspace;
        falls back to a dict-based mock if MARE is not on the path.
        """
        try:
            from mare.workspace.shared_workspace import SharedWorkspace
            return SharedWorkspace()
        except ImportError:
            logger.warning(
                "MARE SharedWorkspace not found — using MockWorkspace. "
                "Add the MARE repo to your PYTHONPATH."
            )
            return MockWorkspace()

    def _print_episode_summary(self, oracle_result):
        """Print end-of-episode summary to stdout."""
        print(
            f"\n{'─'*50}\n"
            f"Episode done | domain={self._scenario.domain} "
            f"steps={self._step_count}\n"
            f"  Coverage : {oracle_result.coverage_score:.3f}\n"
            f"  Precision: {oracle_result.precision_score:.3f}\n"
            f"  Conflict : {oracle_result.conflict_score:.3f}\n"
            f"  NFR      : {oracle_result.nfr_score:.3f}\n"
            f"  TOTAL    : {oracle_result.total_reward:.3f}\n"
            f"  Covered  : {len(oracle_result.covered_reqs)}/"
            f"{len(self._scenario.ground_truth_reqs)} reqs\n"
            f"  Missed   : {oracle_result.missed_reqs[:2]}{'...' if len(oracle_result.missed_reqs)>2 else ''}\n"
            f"{'─'*50}"
        )


# ─────────────────────────────────────────────
#  Mock workspace for testing without MARE
# ─────────────────────────────────────────────

class MockWorkspace:
    """
    Minimal dict-backed workspace for testing and unit tests.
    Replace with MARE's real SharedWorkspace in production.
    """

    def __init__(self):
        self._data: Dict[str, str] = {}
        self.current_phase: int = 0

    def set(self, key: str, value: str):
        self._data[key] = value

    def get(self, key: str, default: str = "") -> str:
        return self._data.get(key, default)

    def update(self, updates: dict):
        self._data.update(updates)

    def get_srs(self) -> str:
        parts = [
            self._data.get("srs_document", ""),
            self._data.get("req_draft", ""),
        ]
        return "\n".join(p for p in parts if p)

    def __repr__(self):
        keys = list(self._data.keys())
        return f"MockWorkspace(keys={keys}, phase={self.current_phase})"


# ─────────────────────────────────────────────
#  Smoke test — validates Gymnasium compatibility
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.INFO)

    from sim.scenario_gen import ScenarioGenerator
    from sim.oracle import Oracle

    # ── Minimal stubs for testing without full MARE/RL stack ─────────
    class StubStateEncoder:
        state_dim = 1544
        def encode(self, workspace):
            return np.random.randn(self.state_dim).astype(np.float32)

    class StubRewardEngine:
        def score_immediate(self, result, workspace):
            return float(np.random.uniform(0, 0.3))

    class StubAgent:
        def perform_action(self, action_name, workspace):
            workspace.set(
                "req_draft",
                workspace.get("req_draft", "") + f"\nThe system shall support {action_name}."
            )
            if action_name == "write_final_srs":
                workspace.set("srs_document", workspace.get("req_draft", ""))
            return {"output": f"executed {action_name}", "success": True}

    # ── Build the environment ─────────────────────────────────────────
    gen = ScenarioGenerator("data/scenarios/")
    oracle = Oracle()
    enc = StubStateEncoder()
    rew = StubRewardEngine()
    agents = {role: StubAgent() for role in AGENT_ACTION_MAP}

    env = RESimEnv(
        scenario_gen=gen,
        oracle=oracle,
        state_encoder=enc,
        reward_engine=rew,
        agents=agents,
        agent_role="collector",
        max_steps=12,
        verbose=True,
    )

    # ── Gymnasium compatibility check ─────────────────────────────────
    print("\nRunning gymnasium check_env...")
    from stable_baselines3.common.env_checker import check_env
    check_env(env, warn=True)
    print("check_env PASSED\n")

    # ── Manual episode rollout ─────────────────────────────────────────
    print("Running manual episode...")
    obs, info = env.reset()
    print(f"Reset | obs shape={obs.shape} | domain={info['domain']}")

    total_reward = 0.0
    for step in range(12):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"Step {step+1:2d} | action={action} ({info['action_name']:<30}) "
            f"| reward={reward:+.4f} | phase={info['phase']}"
        )
        if terminated or truncated:
            print(f"\nEpisode ended | total_reward={total_reward:.4f}")
            break

    env.render()
