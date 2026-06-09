"""
remarl/eval/benchmark.py
------------------------
Runs REMARL-trained policies against vanilla MARE baseline
on held-out evaluation scenarios and prints a comparison table.

Usage:
    python eval/benchmark.py --checkpoint data/checkpoints/collector_final
    python eval/benchmark.py --checkpoint data/checkpoints/modeler_final --role modeler
    python eval/benchmark.py --checkpoint data/checkpoints/collector_final --domain patient_portal

FIXES APPLIED
─────────────
FIX [1] CRITICAL — Baseline scenario replay used wrong Gymnasium reset API.
        Was:    env.reset(scenario=paired_scenario)
        Fixed:  env.reset(options={"scenario": paired_scenario})
        Impact: Without this fix the baseline always runs on a DIFFERENT randomly
                sampled scenario, breaking the entire paired comparison design.

FIX [2] CRITICAL — run_eval_episode() used already_reset=True / env._last_obs
        which only exists on DummyVecEnv, not on a raw RESimEnv. Would raise
        AttributeError at runtime. Removed the pattern entirely. The caller now
        passes the initial obs directly.

FIX [3] CRITICAL — Oracle, RewardEngine and StateEncoder were constructed
        without config parameters (Oracle(), RewardEngine(), StateEncoder(model)).
        This means the eval environment uses different coverage thresholds,
        reward weights, and potentially a different state vector than during
        training — invalidating every comparison number.

FIX [4] CRITICAL — remarl_math_results and baseline_math_results grew on every
        episode, while remarl_results / baseline_results only grew when the
        oracle fired. The lists could end up different lengths, causing the
        paired t-test to compare the wrong scenarios. Now all four lists are
        appended (or skipped) together atomically.

FIX [5] — PPO model loaded with DummyVecEnv for correct SB3 obs shape, but
        eval loop uses a raw env for direct scenario control. run_eval_episode
        now handles the obs-dimension expansion needed by SB3 predict().

FIX [6] — agent_role was hardcoded as "collector". Now accepted as a parameter
        so modeler / checker checkpoints can be evaluated.

FIX [7] — aggregate_oracle_results() now receives per-episode step counts so
        mean_steps is no longer always 0.0 in the output.

FIX [8] — compare_with_significance() length assertion now enforced — mismatched
        list lengths would silently corrupt the paired t-test.
"""

import argparse
import csv
import datetime
import json
import logging
import shutil
import sys
import yaml
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
logger = logging.getLogger("remarl.eval")


# ── Episode runner ────────────────────────────────────────────────────────────

def run_eval_episode(env, initial_obs, policy=None) -> dict:
    """
    Run one evaluation episode starting from an already-obtained observation.

    Args:
        env:         Raw RESimEnv (already reset by the caller).
        initial_obs: The obs array returned by env.reset() — passed in so the
                     caller controls when reset happens (important for pairing).
        policy:      Trained SB3 PPO model, a custom policy object, or None
                     for pure random sampling. Custom policies must implement
                     predict(obs, deterministic) → (action, state).

    Returns:
        dict with keys: total_reward, steps, oracle (oracle result object or None)

    NOTE on SB3 observation shape (FIX [5]):
        SB3 PPO models trained with DummyVecEnv expect batched observations
        (1, obs_dim).  We detect SB3 models via the presence of the `.policy`
        attribute and expand dims accordingly.  Custom policies (MAREFaithful,
        MARERandomPolicy) ignore the obs argument so the shape doesn't matter.
    """
    obs = initial_obs
    total_reward = 0.0
    steps = 0
    oracle_result = None

    for _ in range(env.max_steps):
        if policy is None:
            action = env.action_space.sample()
        elif hasattr(policy, "policy"):
            # SB3 PPO model — needs batched obs (1, obs_dim)
            action_arr, _ = policy.predict(obs[np.newaxis], deterministic=True)
            action = int(np.array(action_arr).flatten()[0])
        else:
            # Custom policy (MAREFaithfulPolicy, MARERandomPolicy) — obs ignored
            action_raw, _ = policy.predict(obs, deterministic=True)
            action = int(action_raw)

        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        steps += 1

        if step_info.get("oracle_result"):
            oracle_result = step_info["oracle_result"]

        if terminated or truncated:
            break

    return {"total_reward": total_reward, "steps": steps, "oracle": oracle_result}


# ── Baseline policies ─────────────────────────────────────────────────────────

class MAREFaithfulPolicy:
    """
    Faithful re-implementation of MARE's per-phase action schedule
    for the collector role.

    Real MARE elicitation pattern (from mare_pipeline._elicitation_phase):
       stakeholder.speak_user_stories   (handled by env at reset)
       loop:
         collector.propose_question     (action 0)
         stakeholder.answer_question    (handled by env internally)
       collector.write_req_draft        (action 1)
       collector.refine_req_draft       (action 2)
       collector.flag_missing_coverage  (action 3)

    Sequence: 3 questions → 1 draft → 1 refinement → 1 coverage flag, then repeat.
    """

    COLLECTOR_SEQUENCE = [0, 0, 0, 1, 2, 3]

    def __init__(self):
        self._step = 0

    def reset(self):
        self._step = 0

    def predict(self, obs, deterministic=True):
        action = self.COLLECTOR_SEQUENCE[self._step % len(self.COLLECTOR_SEQUENCE)]
        self._step += 1
        return action, None


class MARERandomPolicy:
    """
    Uniform random baseline over Discrete(4).
    Primary comparison baseline for REMARL evaluation.
    seed=i (per episode) gives deterministic-but-varied randomness.
    """

    def __init__(self, seed: int = 0):
        import random
        self._rng = random.Random(seed)

    def predict(self, obs, deterministic=False):
        return self._rng.randint(0, 3), None

    def reset(self):
        pass


# ── Main benchmark function ───────────────────────────────────────────────────

def benchmark(config_path, checkpoint_path, n_eval, domain=None, agent_role="collector"):
    """
    Evaluate a trained policy against MAREFaithfulPolicy on paired scenarios.

    Args:
        config_path:     Path to the YAML config used during training.
        checkpoint_path: Path to the SB3 PPO checkpoint (.zip).
        n_eval:          Number of evaluation episodes.
        domain:          Optional domain filter string.
        agent_role:      Which agent role to evaluate (collector/modeler/checker).
                         FIX [6]: was hardcoded as "collector".
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from sim.scenario_gen import ScenarioGenerator
    from sim.oracle import Oracle
    from sim.re_env import RESimEnv
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

    # FIX [3]: pass all config parameters — previously Oracle/RewardEngine/
    # StateEncoder were constructed with no arguments, using internal defaults
    # that may differ from what was used during training.
    gen = ScenarioGenerator(config["env"]["scenario_dir"])

    oracle = Oracle(
        coverage_threshold=config["reward"]["coverage_threshold"],
    )
    encoder = StateEncoder(
        model_name=config["state_encoder"]["model"],
        max_steps=config["env"]["max_steps_per_episode"],   # FIX [3]: was missing
    )
    reward_engine = RewardEngine(
        clarity_weight=config["reward"]["clarity_weight"],
        consistency_weight=config["reward"]["consistency_weight"],
        coverage_delta_weight=config["reward"]["coverage_delta_weight"],
    )

    raw_agents = AgentFactory.create_all_agents_from_config(config)
    rl_agents  = {r: MARERLAgent(raw_agents[r]) for r in raw_agents}

    def make_env():
        for a in rl_agents.values():
            a.reset()
        return RESimEnv(
            scenario_gen=gen,
            oracle=oracle,
            state_encoder=encoder,
            reward_engine=reward_engine,
            agents=rl_agents,
            agent_role=agent_role,                          # FIX [6]
            max_steps=config["env"]["max_steps_per_episode"],
        )

    # FIX [5]: load model with DummyVecEnv so SB3's internal obs shape is correct.
    # The eval loop uses a raw env for direct scenario control — obs is expanded
    # to (1, obs_dim) inside run_eval_episode before calling model.predict().
    trained_model = PPO.load(checkpoint_path, env=DummyVecEnv([make_env]))

    # Single raw env instance drives both REMARL and baseline runs each episode.
    env = make_env()

    # FIX [4]: all four lists are appended/skipped atomically — they must stay
    # the same length so the paired t-test compares matching scenarios.
    remarl_results        = []
    baseline_results      = []
    remarl_math_results   = []
    baseline_math_results = []
    episode_rows          = []

    # Separate step-count tracking for FIX [7] (mean_steps aggregation)
    remarl_steps_list   = []
    baseline_steps_list = []

    domain_opt = {"domain": domain} if domain else None
    start_time = datetime.datetime.now()

    print(f"\n{'─'*65}")
    print(f"  REMARL Evaluation — {n_eval} episodes  (role: {agent_role})")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Config:     {config_path}")
    print(f"  Started:    {start_time.strftime('%H:%M:%S')}")
    print(f"  Est. time:  ~{n_eval * 14} minutes")
    print(f"{'─'*65}")
    print(
        f"  {'Ep':>3}  {'REMARL Cov':>11}  {'Base Cov':>9}  "
        f"{'REMARL Tot':>11}  {'Base Tot':>9}  {'Time':>8}"
    )
    print(f"{'─'*65}")

    for i in range(n_eval):
        ep_start = datetime.datetime.now()

        # ── REMARL run ────────────────────────────────────────────────────────
        r_obs, _ = env.reset(options=domain_opt)
        paired_scenario = env._scenario   # capture for baseline replay

        r_ep = run_eval_episode(env, r_obs, policy=trained_model)

        r_srs = (
            env._workspace.get("srs_document", "")
            or env._workspace.get("req_draft", "")
        )
        r_math = evaluator.evaluate(r_srs, paired_scenario)

        # ── Baseline run — replay the IDENTICAL scenario ───────────────────
        # FIX [1]: was env.reset(scenario=paired_scenario) — Gymnasium's reset()
        # does not accept a top-level `scenario` kwarg; it must go via `options`.
        # The old call silently discarded the scenario and sampled a fresh one,
        # breaking the entire paired comparison design.
        baseline_policy = MAREFaithfulPolicy()
        baseline_policy.reset()
        b_obs, _ = env.reset(options={"scenario": paired_scenario})   # FIX [1]

        b_ep = run_eval_episode(env, b_obs, policy=baseline_policy)

        b_srs = (
            env._workspace.get("srs_document", "")
            or env._workspace.get("req_draft", "")
        )
        b_math = evaluator.evaluate(b_srs, paired_scenario)

        # ── Extract oracle scores ─────────────────────────────────────────────
        r_oracle = r_ep["oracle"]
        b_oracle = b_ep["oracle"]

        r_cov = r_oracle.coverage_score if r_oracle else 0.0
        b_cov = b_oracle.coverage_score if b_oracle else 0.0
        r_tot = r_oracle.total_reward   if r_oracle else 0.0
        b_tot = b_oracle.total_reward   if b_oracle else 0.0
        ep_min = (datetime.datetime.now() - ep_start).seconds // 60

        print(
            f"  {i+1:>3}  {r_cov:>11.3f}  {b_cov:>9.3f}  "
            f"{r_tot:>11.3f}  {b_tot:>9.3f}  {ep_min:>6}min"
        )

        # FIX [4]: only record results when BOTH oracle results are present.
        # Previously remarl_math_results was always appended but remarl_results
        # was only appended when oracle fired — lists diverged in length, making
        # the paired t-test compare different scenarios.
        if r_oracle and b_oracle:
            remarl_results.append(r_oracle)
            baseline_results.append(b_oracle)
            remarl_math_results.append(r_math)
            baseline_math_results.append(b_math)
            remarl_steps_list.append(r_ep["steps"])
            baseline_steps_list.append(b_ep["steps"])

            episode_rows.append({
                "episode":                    i + 1,
                "domain":                     getattr(paired_scenario, "domain", "unknown"),
                "remarl_coverage":            round(r_cov, 4),
                "baseline_coverage":          round(b_cov, 4),
                "remarl_total_reward":        round(r_tot, 4),
                "baseline_total_reward":      round(b_tot, 4),
                "remarl_req_f1_semantic":     round(r_math.req_prf_semantic.f1, 4) if r_math else None,
                "baseline_req_f1_semantic":   round(b_math.req_prf_semantic.f1, 4) if b_math else None,
                "remarl_steps":               r_ep["steps"],
                "baseline_steps":             b_ep["steps"],
            })
        else:
            logger.warning(
                f"Episode {i+1}: oracle result missing for "
                f"{'REMARL' if not r_oracle else 'baseline'} — "
                f"skipped from aggregation to preserve pairing."
            )

    print(f"{'─'*65}\n")

    n_valid = len(remarl_results)
    if n_valid == 0:
        print("No valid paired oracle results — check max_steps setting.")
        return

    print(f"Valid episodes for statistics: {n_valid} / {n_eval}")

    # FIX [7]: pass step lists so mean_steps is computed, not 0.0
    remarl_agg   = aggregate_oracle_results(remarl_results,   steps=remarl_steps_list)
    baseline_agg = aggregate_oracle_results(baseline_results, steps=baseline_steps_list)

    print_comparison(remarl_agg, baseline_agg)

    # FIX [8]: compare_with_significance now asserts equal lengths internally
    significance = compare_with_significance(remarl_results, baseline_results)

    latex_str = None
    if remarl_math_results and baseline_math_results:
        suite.print_comparison_table(remarl_math_results, baseline_math_results)
        suite.compare_with_significance(remarl_math_results, baseline_math_results)
        suite.print_per_domain_table(remarl_math_results, label="REMARL")
        latex_str = suite.print_latex_table(remarl_math_results, baseline_math_results)

    total_min = (datetime.datetime.now() - start_time).seconds // 60
    print(f"Total time: {total_min} minutes")

    # ── Save benchmark outputs ─────────────────────────────────────────────────
    role_tag = Path(checkpoint_path).stem
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"data/benchmarks/eval_{role_tag}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results_payload = {
        "checkpoint":    str(checkpoint_path),
        "agent_role":    agent_role,
        "n_eval":        n_eval,
        "n_valid":       n_valid,           # episodes that produced valid oracle results
        "domain_filter": domain,
        "total_minutes": total_min,
        "timestamp":     ts,
        "remarl":        remarl_agg.to_dict(),
        "baseline":      baseline_agg.to_dict(),
        "significance":  significance,
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
    parser.add_argument(
        "--role", default="collector",
        choices=["collector", "modeler", "checker"],
    )
    args = parser.parse_args()
    benchmark(args.config, args.checkpoint, args.n_eval, args.domain, args.role)