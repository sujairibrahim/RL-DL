"""
Paired evaluation: each scenario is replayed for (a) MAREFaithfulPolicy
baseline and (b) the trained REMARL policy, on identical hidden subsets.

Run after P0 patches + at least the collector is trained:
    python eval/run_paired_eval.py \
        --checkpoint data/checkpoints/collector_final.zip \
        --n 22
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml
from stable_baselines3 import PPO

from eval.benchmark import MAREFaithfulPolicy, MARERandomPolicy


def run_episode(env, policy, scenario) -> dict:
    obs, _info = env.reset(options={"scenario": scenario})
    if hasattr(policy, "reset"):
        policy.reset()
    total_reward = 0.0
    per_step = []
    for _ in range(env.max_steps):
        action, _ = policy.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(int(action))
        total_reward += r
        per_step.append(r)
        if terminated or truncated:
            break
    final = info.get("oracle_result")
    coverage  = float(final.coverage_score)  if final else None
    precision = float(getattr(final, "precision_score", None) or 0.0) if final else None
    conflict  = float(getattr(final, "conflict_score",  None) or 0.0) if final else None
    nfr       = float(getattr(final, "nfr_score",       None) or 0.0) if final else None
    return {
    "total_reward":   final.total_reward    if final else 0.0,   # ← oracle's 0-1 scalar
    "coverage":       final.coverage_score  if final else 0.0,
    "precision":      final.precision_score if final else 0.0,
    "conflict":       final.conflict_score  if final else 0.0,
    "nfr":            final.nfr_score       if final else 0.0,
    "episode_steps":  len(per_step),
    "episode_cumulative": total_reward,     # keep the old value here under a clear name
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to trained policy zip")
    ap.add_argument("--config", default="configs/remarl_train_fast.yaml")
    ap.add_argument("--n", type=int, default=22,
                    help="Number of scenarios (22 = 80%% power at d≈0.643)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/benchmarks")
    ap.add_argument("--domain", default=None, help="Restrict to a single domain")
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

    gen     = ScenarioGenerator(config["env"]["scenario_dir"], seed=args.seed)
    oracle  = Oracle(coverage_threshold=config["reward"]["coverage_threshold"])
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
            agents=rl_agents, agent_role="collector",
            max_steps=config["env"]["max_steps_per_episode"],
        )

    env = make_env_instance()
    trained = PPO.load(args.checkpoint, env=env)

    baseline = MAREFaithfulPolicy()
    rand     = MARERandomPolicy()

    domain_filter = {"domain": args.domain} if args.domain else None
    scenarios = []
    _tmp_obs, _ = env.reset(options=domain_filter)
    for _ in range(args.n):
        obs, _ = env.reset(options=domain_filter)
        scenarios.append(env._scenario)

    rows = []
    print(f"\n{'─'*70}")
    print(f"  Paired evaluation — {args.n} scenarios")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'─'*70}")
    print(f"  {'#':>3}  {'Domain':<16}  {'Base':>8}  {'REMARL':>8}  {'Rand':>8}")
    print(f"{'─'*70}")

    for i, sc in enumerate(scenarios):
        r_base = run_episode(env, baseline, sc)
        r_rl   = run_episode(env, trained,  sc)
        r_rand = run_episode(env, rand,     sc)
        rows.append({
            "scenario_idx": i,
            "domain": getattr(sc, "domain", "unknown"),
            "baseline": r_base,
            "remarl":   r_rl,
            "random":   r_rand,
        })
        print(f"  {i+1:>3}  {getattr(sc,'domain','?'):<16}  "
              f"{r_base['total_reward']:>8.3f}  "
              f"{r_rl['total_reward']:>8.3f}  "
              f"{r_rand['total_reward']:>8.3f}")

    out = out_dir / f"paired_eval_{stamp}.json"
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
