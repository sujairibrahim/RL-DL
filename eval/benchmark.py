"""
remarl/eval/benchmark.py
------------------------
Runs REMARL-trained policies against vanilla MARE baseline
on held-out evaluation scenarios and prints a comparison table.

Usage:
    python eval/benchmark.py --checkpoint data/checkpoints/collector_final
    python eval/benchmark.py --checkpoint data/checkpoints/collector_final --domain patient_portal
"""

import argparse, csv, json, logging, shutil, sys, yaml, datetime
import numpy as np
from pathlib import Path



sys.path.insert(0, str(Path(__file__).parent.parent))
logger = logging.getLogger("remarl.eval")


def run_eval_episode(env, model=None, already_reset: bool = False) -> dict:
    if not already_reset:
        obs, info = env.reset()
    else:
        obs = env._last_obs          # set by reset() at re_env.py:215
    total_reward = 0.0
    steps = 0
    oracle_result = None

    for _ in range(env.max_steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, step_info = env.step(int(action))
        total_reward += reward
        steps += 1

        if step_info.get("oracle_result"):
            oracle_result = step_info["oracle_result"]

        if terminated or truncated:
            break

    return {"total_reward": total_reward, "steps": steps, "oracle": oracle_result}


class MAREFixedPolicy:
    """Degenerate fixed-action policy — kept for reference only."""
    FIXED_ACTION = {
        "stakeholder": 0,
        "collector":   0,
        "modeler":     0,
        "checker":     0,
        "documenter":  2,
    }

    def __init__(self):
        self._step = 0

    def predict(self, obs, deterministic=True):
        from sim.re_env import DEFAULT_PHASE_SEQUENCE
        role = DEFAULT_PHASE_SEQUENCE[self._step % len(DEFAULT_PHASE_SEQUENCE)][0]
        action = self.FIXED_ACTION.get(role, 0)
        self._step += 1
        return action, None

    def reset(self):
        self._step = 0


class MARERandomPolicy:
    """
    Uniform random baseline over Discrete(4).
    Primary comparison baseline for REMARL evaluation.
    Each action is drawn uniformly from {0,1,2,3}, independent of observation.
    seed=i (per episode) gives deterministic-but-varied randomness across episodes.
    """

    def __init__(self, seed: int = 0):
        import random
        self._rng = random.Random(seed)

    def predict(self, obs, deterministic=False):
        return self._rng.randint(0, 3), None

    def reset(self):
        pass


def benchmark(config_path, checkpoint_path, n_eval, domain=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from sim.scenario_gen import ScenarioGenerator
    from sim.oracle import Oracle
    from sim.re_env import RESimEnv, AGENT_ACTION_MAP
    from rl.state_encoder import StateEncoder
    from rl.reward import RewardEngine
    from eval.metrics import aggregate_oracle_results, print_comparison, compare_with_significance
    from mare.agents.factory import AgentFactory
    from mare.rl_adapter import MARERLAgent
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from eval.mare_eval import MAREEvaluator, EvaluationSuite
    
    evaluator = MAREEvaluator()
    suite     = EvaluationSuite()
    remarl_math_results   = []
    baseline_math_results = []
    gen     = ScenarioGenerator(config["env"]["scenario_dir"])
    oracle  = Oracle()
    encoder = StateEncoder(model_name=config["state_encoder"]["model"])
    reward  = RewardEngine()

    raw_agents = AgentFactory.create_all_agents_from_config(config)
    rl_agents  = {r: MARERLAgent(raw_agents[r]) for r in raw_agents}

    def make_env():
        for a in rl_agents.values():
            a.reset()
        return RESimEnv(
            scenario_gen=gen, oracle=oracle,
            state_encoder=encoder, reward_engine=reward,
            agents=rl_agents, agent_role="collector",
            max_steps=config["env"]["max_steps_per_episode"],
        )

    env = make_env()
    trained_model = PPO.load(checkpoint_path, env=DummyVecEnv([make_env]))

    remarl_results   = []
    baseline_results = []
    episode_rows     = []
    opt = {"domain": domain} if domain else None
    start_time = datetime.datetime.now()

    print(f"\n{'─'*65}")
    print(f"  REMARL Evaluation — {n_eval} episodes")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Started:    {start_time.strftime('%H:%M:%S')}")
    print(f"  Est. time:  ~{n_eval * 14} minutes")
    print(f"{'─'*65}")
    print(f"  {'Ep':>3}  {'REMARL Cov':>11}  {'Base Cov':>9}  "
          f"{'REMARL Tot':>11}  {'Base Tot':>9}  {'Time':>8}")
    print(f"{'─'*65}")

    for i in range(n_eval):
        ep_start = datetime.datetime.now()

        # REMARL run — capture exact scenario for pairing
        env.reset(options=opt)
        r_ep = run_eval_episode(env, model=trained_model, already_reset=True)
        paired_scenario = env._scenario   # same hidden_reqs will be replayed for baseline
        srs_text = env._workspace.get('srs_document', '') or env._workspace.get('req_draft', '')
        math_result = evaluator.evaluate(srs_text, env._scenario)
        remarl_math_results.append(math_result)

        # Baseline run — replay the IDENTICAL scenario
        baseline_policy = MAREFixedPolicy()
        baseline_policy.reset()
        env.reset(scenario=paired_scenario)
        b_ep = run_eval_episode(env, model=baseline_policy, already_reset=True)
        srs_text_b = env._workspace.get('srs_document', '') or env._workspace.get('req_draft', '')
        math_result_b = evaluator.evaluate(srs_text_b, env._scenario)
        baseline_math_results.append(math_result_b)


        r_cov  = r_ep["oracle"].coverage_score  if r_ep["oracle"]  else 0.0
        b_cov  = b_ep["oracle"].coverage_score  if b_ep["oracle"]  else 0.0
        r_tot  = r_ep["oracle"].total_reward     if r_ep["oracle"]  else 0.0
        b_tot  = b_ep["oracle"].total_reward     if b_ep["oracle"]  else 0.0
        ep_min = (datetime.datetime.now() - ep_start).seconds // 60

        print(f"  {i+1:>3}  {r_cov:>11.3f}  {b_cov:>9.3f}  "
              f"{r_tot:>11.3f}  {b_tot:>9.3f}  {ep_min:>6}min")

        if r_ep["oracle"]:
            remarl_results.append(r_ep["oracle"])
        if b_ep["oracle"]:
            baseline_results.append(b_ep["oracle"])

        episode_rows.append({
            "episode": i + 1,
            "domain": getattr(env._scenario, "domain", "unknown"),
            "remarl_coverage": round(r_cov, 4),
            "baseline_coverage": round(b_cov, 4),
            "remarl_total_reward": round(r_tot, 4),
            "baseline_total_reward": round(b_tot, 4),
            "remarl_req_f1_semantic": round(math_result.req_prf_semantic.f1, 4) if math_result else None,
            "baseline_req_f1_semantic": round(math_result_b.req_prf_semantic.f1, 4) if math_result_b else None,
            "remarl_steps": r_ep["steps"],
            "baseline_steps": b_ep["steps"],
        })

    print(f"{'─'*65}\n")

    if not remarl_results or not baseline_results:
        print("No oracle results — check max_steps setting.")
        return

    remarl_agg   = aggregate_oracle_results(remarl_results)
    baseline_agg = aggregate_oracle_results(baseline_results)
    print_comparison(remarl_agg, baseline_agg)
    significance = compare_with_significance(remarl_results, baseline_results)

    latex_str = None
    if remarl_math_results and baseline_math_results:
        suite.print_comparison_table(remarl_math_results, baseline_math_results)
        suite.compare_with_significance(remarl_math_results, baseline_math_results)
        suite.print_per_domain_table(remarl_math_results, label="REMARL")
        latex_str = suite.print_latex_table(remarl_math_results, baseline_math_results)

    total_min = (datetime.datetime.now() - start_time).seconds // 60
    print(f"Total time: {total_min} minutes")

    # ── Save benchmark outputs ────────────────────────────────────────────────
    role_tag = Path(checkpoint_path).stem  # e.g. "collector_final"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"data/benchmarks/eval_{role_tag}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results_payload = {
        "checkpoint": str(checkpoint_path),
        "n_eval": n_eval,
        "domain_filter": domain,
        "total_minutes": total_min,
        "timestamp": ts,
        "remarl": remarl_agg.to_dict(),
        "baseline": baseline_agg.to_dict(),
        "significance": significance,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results_payload, f, indent=2)

    if episode_rows:
        with open(out_dir / "episodes.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=episode_rows[0].keys())
            writer.writeheader()
            writer.writerows(episode_rows)

    if latex_str:
        (out_dir / "table.tex").write_text(latex_str)

    shutil.copy(config_path, out_dir / "config.yaml")

    print(f"\n[Benchmark outputs saved to {out_dir}]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/remarl_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_eval",     type=int, default=50)
    parser.add_argument("--domain",     default=None)
    args = parser.parse_args()
    benchmark(args.config, args.checkpoint, args.n_eval, args.domain)