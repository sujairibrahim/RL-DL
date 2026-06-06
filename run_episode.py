"""
remarl/run_episode.py
---------------------
Run a single REMARL episode end-to-end and print a detailed trace.
Useful for debugging before training, and for demos.

Usage:
    # With random policy (baseline)
    python run_episode.py

    # With a trained checkpoint
    python run_episode.py --checkpoint data/checkpoints/collector_final

    # Force a specific domain
    python run_episode.py --domain patient_portal --difficulty hard
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.WARNING,   # suppress INFO noise during demo
    format="%(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("remarl.run_episode")


def run(config_path: str, checkpoint: str = None,
        domain: str = None, difficulty: str = None):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    from sim.scenario_gen import ScenarioGenerator
    from sim.oracle import Oracle
    from sim.re_env import RESimEnv, MockWorkspace, AGENT_ACTION_MAP
    from rl.state_encoder import StateEncoder
    from rl.reward import RewardEngine

    # ── Build components ──────────────────────────────────────────────
    gen     = ScenarioGenerator(config["env"]["scenario_dir"])
    oracle  = Oracle()
    encoder = StateEncoder(model_name=config["state_encoder"]["model"])
    reward  = RewardEngine()

    class StubAgent:
        """
        Placeholder — replace with real MARE agents.
        Appends one requirement per action call.
        write_final_srs copies draft → srs_document.
        """
        def perform_action(self, action_name, workspace):
            text = f"The system shall support {action_name.replace('_', ' ')}."
            workspace.set(
                "req_draft",
                workspace.get("req_draft", "") + "\n" + text
            )
            if action_name in ("write_final_srs", "approve_and_document"):
                workspace.set("srs_document", workspace.get("req_draft", ""))
            return {"output": text, "action_used": action_name}

    agents = {role: StubAgent() for role in AGENT_ACTION_MAP}

    env = RESimEnv(
        scenario_gen=gen,
        oracle=oracle,
        state_encoder=encoder,
        reward_engine=reward,
        agents=agents,
        agent_role="collector",
        max_steps=config["env"]["max_steps_per_episode"],
        step_penalty=config["reward"]["step_penalty"],
        verbose=True,
    )

    # ── Load model or use random ───────────────────────────────────────
    model = None
    if checkpoint:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        def make_env():
            return RESimEnv(
                scenario_gen=gen, oracle=oracle,
                state_encoder=encoder, reward_engine=reward,
                agents=agents, agent_role="collector",
                max_steps=config["env"]["max_steps_per_episode"],
            )
        model = PPO.load(checkpoint, env=DummyVecEnv([make_env]))
        print(f"Loaded checkpoint: {checkpoint}")

    policy_label = "REMARL (trained)" if model else "Random baseline"
    print(f"\n{'═'*60}")
    print(f"  REMARL Episode Runner — {policy_label}")
    print(f"{'═'*60}")

    # ── Reset ─────────────────────────────────────────────────────────
    options = {}
    if domain:     options["domain"]     = domain
    if difficulty: options["difficulty"] = difficulty
    obs, info = env.reset(options=options or None)

    scenario = env._scenario
    print(f"\nScenario")
    print(f"  Domain     : {scenario.domain}")
    print(f"  Difficulty : {scenario.difficulty}")
    print(f"  Idea       : {scenario.rough_idea[:80]}...")
    print(f"  Total reqs : {len(scenario.ground_truth_reqs)}")
    print(f"  Hidden reqs: {len(scenario.hidden_reqs)} (agents must discover these)")

    print(f"\n  Hidden requirements:")
    for r in scenario.hidden_reqs:
        print(f"    - {r[:70]}...")

    print(f"\n{'─'*60}")
    print(f"  Episode trace")
    print(f"{'─'*60}")

    # ── Step loop ─────────────────────────────────────────────────────
    total_reward = 0.0
    for step in range(env.max_steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward

        print(
            f"  Step {step+1:2d} | {step_info['agent_role']:<12} "
            f"| {step_info['action_name']:<32} "
            f"| r={reward:+.4f}"
        )

        if terminated or truncated:
            break

    # ── Final summary ─────────────────────────────────────────────────
    oracle_result = step_info.get("oracle_result")
    print(f"\n{'─'*60}")
    print(f"  Final results")
    print(f"{'─'*60}")
    print(f"  Steps taken  : {step+1}")
    print(f"  Total reward : {total_reward:.4f}")

    if oracle_result:
        print(f"  Coverage     : {oracle_result.coverage_score:.4f}")
        print(f"  Precision    : {oracle_result.precision_score:.4f}")
        print(f"  Conflict     : {oracle_result.conflict_score:.4f}")
        print(f"  NFR          : {oracle_result.nfr_score:.4f}")
        print(f"  ORACLE TOTAL : {oracle_result.total_reward:.4f}")
        print(f"\n  Covered {len(oracle_result.covered_reqs)}/{len(scenario.ground_truth_reqs)} ground-truth requirements")
        if oracle_result.missed_reqs:
            print(f"\n  Missed requirements:")
            for r in oracle_result.missed_reqs:
                print(f"    - {r[:70]}...")
    else:
        print("  (Episode ended without oracle score — increase max_steps)")

    print(f"\n{'═'*60}\n")
    return total_reward, oracle_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one REMARL episode")
    parser.add_argument("--config",     default="configs/remarl_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--domain",     default=None)
    parser.add_argument("--difficulty", default=None,
                        choices=["easy", "medium", "hard"])
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.domain, args.difficulty)
