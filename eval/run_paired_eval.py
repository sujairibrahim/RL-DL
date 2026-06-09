"""
Paired evaluation: each scenario is replayed for (a) MAREFaithfulPolicy
baseline, (b) the trained REMARL policy, and (c) MARERandomPolicy,
all on identical hidden subsets.

Run after training:
    python eval/run_paired_eval.py \\
        --checkpoint data/checkpoints/collector_final.zip \\
        --n 22

Then analyse results:
    python eval/analyze_paired.py \\
        --input data/benchmarks/paired_eval_<timestamp>.json \\
        --comparison remarl_vs_baseline

FIXES APPLIED
─────────────
FIX [1] — Removed the wasted env.reset() call before the scenario collection
          loop. Each reset fires a full LLM episode; the old code discarded the
          first scenario immediately, costing ~14 minutes of compute.

FIX [2] — Removed dead-code variables (coverage, precision, conflict, nfr) that
          were assigned in run_episode but never used. The return dict already
          captured these directly from the oracle result object.

FIX [3] — PPO model now loaded with DummyVecEnv([make_env_instance]) to match
          how it was saved by stable-baselines3. run_episode detects SB3 models
          via the .policy attribute and expands obs to (1, obs_dim) before
          calling predict().

FIX [4] — Added srs_text and ground_truth to each system's result dict.
          mare_style_eval.py's main() requires these fields; without them it
          silently returns all-zero F1 scores.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.benchmark import MAREFaithfulPolicy, MARERandomPolicy


def run_episode(env, policy, scenario) -> dict:
    """
    Run one episode on a specific scenario with any policy.

    Returns a dict containing oracle scores, step counts, the generated SRS
    text, and the ground-truth requirement list.  The srs_text / ground_truth
    fields (FIX [4]) are needed by mare_style_eval.py's standalone runner.
    """
    obs, _info = env.reset(options={"scenario": scenario})
    if hasattr(policy, "reset"):
        policy.reset()

    total_reward = 0.0
    per_step     = []

    for _ in range(env.max_steps):
        # FIX [3]: SB3 PPO models expect batched obs (1, obs_dim).
        # Custom policies (MAREFaithful, MARERandom) ignore obs entirely.
        if hasattr(policy, "policy"):
            action_arr, _ = policy.predict(obs[np.newaxis], deterministic=True)
            action = int(np.array(action_arr).flatten()[0])
        else:
            action_raw, _ = policy.predict(obs, deterministic=True)
            action = int(action_raw)

        obs, r, terminated, truncated, info = env.step(action)
        total_reward += r
        per_step.append(r)
        if terminated or truncated:
            break

    final = info.get("oracle_result")

    # FIX [4]: capture SRS text and ground truth for mare_style_eval compatibility.
    # Previously the JSON only had numeric scores; mare_style_eval.py's main()
    # looked for srs_text / ground_truth and silently got empty strings, producing
    # all-zero F1 tables without any error message.
    srs_text = (
        env._workspace.get("srs_document", "")
        or env._workspace.get("req_draft", "")
    )
    ground_truth = list(getattr(env._scenario, "ground_truth_reqs", []))

    # FIX [2]: removed dead-code assignments for coverage/precision/conflict/nfr.
    # Those variables were computed but never placed in the returned dict.
    return {
        "total_reward":       final.total_reward    if final else 0.0,
        "coverage":           final.coverage_score  if final else 0.0,
        "precision":          final.precision_score if final else 0.0,
        "conflict":           final.conflict_score  if final else 0.0,
        "nfr":                final.nfr_score       if final else 0.0,
        "episode_steps":      len(per_step),
        "episode_cumulative": total_reward,
        "srs_text":           srs_text,       # FIX [4]
        "ground_truth":       ground_truth,   # FIX [4]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to trained policy zip")
    ap.add_argument("--config", default="configs/remarl_train_fast.yaml")
    ap.add_argument(
        "--n", type=int, default=22,
        help="Number of scenarios (22 = ~80%% power at d≈0.643)",
    )
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--out",    default="data/benchmarks")
    ap.add_argument("--domain", default=None, help="Restrict to a single domain")
    ap.add_argument(
        "--role", default="collector",
        choices=["collector", "modeler", "checker"],
        help="Agent role being evaluated",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from sim.scenario_gen import ScenarioGenerator
    from sim.oracle import Oracle
    from sim.re_env import RESimEnv
    from rl.state_encoder import StateEncoder
    from rl.reward import RewardEngine
    from mare.agents.factory import AgentFactory
    from mare.rl_adapter import MARERLAgent

    gen = ScenarioGenerator(config["env"]["scenario_dir"], seed=args.seed)
    oracle = Oracle(
        coverage_threshold=config["reward"]["coverage_threshold"],
    )
    encoder = StateEncoder(
        model_name=config["state_encoder"]["model"],
        max_steps=config["env"]["max_steps_per_episode"],
    )
    reward_engine = RewardEngine(
        clarity_weight=config["reward"]["clarity_weight"],
        consistency_weight=config["reward"]["consistency_weight"],
        coverage_delta_weight=config["reward"]["coverage_delta_weight"],
    )

    raw_agents = AgentFactory.create_all_agents_from_config(config)
    rl_agents  = {r: MARERLAgent(raw_agents[r]) for r in raw_agents}

    def make_env_instance():
        for a in rl_agents.values():
            a.reset()
        return RESimEnv(
            scenario_gen=gen, oracle=oracle,
            state_encoder=encoder, reward_engine=reward_engine,
            agents=rl_agents,
            agent_role=args.role,
            max_steps=config["env"]["max_steps_per_episode"],
        )

    # FIX [3]: load with DummyVecEnv to match how SB3 saves obs/action shapes.
    trained = PPO.load(args.checkpoint, env=DummyVecEnv([make_env_instance]))

    env = make_env_instance()

    baseline = MAREFaithfulPolicy()
    rand     = MARERandomPolicy(seed=args.seed)

    domain_filter = {"domain": args.domain} if args.domain else None

    # FIX [1]: collect scenarios in a single loop — the old code had an extra
    # env.reset() before the loop that wasted ~14 minutes of LLM compute.
    scenarios = []
    for _ in range(args.n):
        obs, _ = env.reset(options=domain_filter)
        scenarios.append(env._scenario)

    rows = []
    print(f"\n{'─'*72}")
    print(f"  Paired evaluation — {args.n} scenarios  (role: {args.role})")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'─'*72}")
    print(
        f"  {'#':>3}  {'Domain':<20}  {'Baseline':>9}  "
        f"{'REMARL':>9}  {'Random':>9}"
    )
    print(f"{'─'*72}")

    for i, sc in enumerate(scenarios):
        r_base = run_episode(env, baseline, sc)
        r_rl   = run_episode(env, trained,  sc)
        r_rand = run_episode(env, rand,     sc)

        rows.append({
            "scenario_idx": i,
            "domain":       getattr(sc, "domain", "unknown"),
            "baseline":     r_base,
            "remarl":       r_rl,
            "random":       r_rand,
        })
        print(
            f"  {i+1:>3}  {getattr(sc,'domain','?'):<20}  "
            f"{r_base['total_reward']:>9.3f}  "
            f"{r_rl['total_reward']:>9.3f}  "
            f"{r_rand['total_reward']:>9.3f}"
        )

    print(f"{'─'*72}\n")

    out_file = out_dir / f"paired_eval_{stamp}.json"
    out_file.write_text(json.dumps(rows, indent=2, default=str))
    print(f"Wrote {out_file}")
    print(
        f"Next step — run statistics:\n"
        f"  python eval/analyze_paired.py --input {out_file} "
        f"--comparison remarl_vs_baseline\n"
        f"  python eval/analyze_paired.py --input {out_file} "
        f"--comparison remarl_vs_random\n"
        f"  python -m eval.mare_style_eval --episodes {out_file}"
    )


if __name__ == "__main__":
    main()