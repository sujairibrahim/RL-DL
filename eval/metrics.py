"""
remarl/eval/metrics.py
----------------------
Evaluation metrics comparing REMARL vs baseline MARE.
"""

import numpy as np
from dataclasses import dataclass
from typing import List
from scipy import stats


@dataclass
class EvalResult:
    mean_coverage:     float
    mean_precision:    float
    mean_total_reward: float
    mean_steps:        float
    std_reward:        float
    n_episodes:        int
    ci95_reward_low:   float = 0.0
    ci95_reward_high:  float = 0.0
    ci95_cov_low:      float = 0.0
    ci95_cov_high:     float = 0.0

    def to_dict(self) -> dict:
        return {
            "mean_coverage": self.mean_coverage,
            "mean_precision": self.mean_precision,
            "mean_total_reward": self.mean_total_reward,
            "mean_steps": self.mean_steps,
            "std_reward": self.std_reward,
            "n_episodes": self.n_episodes,
            "ci95_reward": [self.ci95_reward_low, self.ci95_reward_high],
            "ci95_coverage": [self.ci95_cov_low, self.ci95_cov_high],
        }


def aggregate_oracle_results(results: list) -> EvalResult:
    rewards    = [r.total_reward    for r in results]
    coverages  = [r.coverage_score  for r in results]
    precisions = [r.precision_score for r in results]
    r_ci   = stats.t.interval(0.95, len(rewards) - 1,
                               loc=np.mean(rewards), scale=stats.sem(rewards))
    cov_ci = stats.t.interval(0.95, len(coverages) - 1,
                               loc=np.mean(coverages), scale=stats.sem(coverages))
    return EvalResult(
        mean_coverage=float(np.mean(coverages)),
        mean_precision=float(np.mean(precisions)),
        mean_total_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_steps=0.0,
        n_episodes=len(results),
        ci95_reward_low=float(r_ci[0]),
        ci95_reward_high=float(r_ci[1]),
        ci95_cov_low=float(cov_ci[0]),
        ci95_cov_high=float(cov_ci[1]),
    )


def print_comparison(remarl: EvalResult, baseline: EvalResult):
    delta = remarl.mean_total_reward - baseline.mean_total_reward
    print(f"\n{'─'*74}")
    print(f"{'Metric':<25} {'REMARL':>10} {'95% CI':>22} {'Baseline':>10} {'Delta':>8}")
    print(f"{'─'*74}")
    r_ci  = f"[{remarl.ci95_reward_low:.4f}, {remarl.ci95_reward_high:.4f}]"
    print(f"{'Mean total reward':<25} {remarl.mean_total_reward:>10.4f} {r_ci:>22}  "
          f"{baseline.mean_total_reward:>10.4f} {delta:>+8.4f}")
    rc_ci = f"[{remarl.ci95_cov_low:.4f}, {remarl.ci95_cov_high:.4f}]"
    print(f"{'Mean coverage':<25} {remarl.mean_coverage:>10.4f} {rc_ci:>22}  "
          f"{baseline.mean_coverage:>10.4f} {remarl.mean_coverage - baseline.mean_coverage:>+8.4f}")
    print(f"{'Mean precision':<25} {remarl.mean_precision:>10.4f} {'N/A':>22}  "
          f"{baseline.mean_precision:>10.4f} {remarl.mean_precision - baseline.mean_precision:>+8.4f}")
    print(f"{'─'*74}\n")




def compare_with_significance(
    remarl_results: list,
    baseline_results: list,
) -> dict:
    """
    Run paired t-test comparing REMARL vs MARE random baseline.
    Both lists must be equal length — each pair is the SAME scenario run twice.

    Uses Bonferroni correction (alpha=0.05/2=0.025) because two hypotheses
    are tested simultaneously (reward AND coverage).

    Returns dict with mean scores, delta, p-values, Cohen's d, and 95% CIs.
    """
    ALPHA = 0.05
    N_TESTS = 2                            # reward + coverage
    ALPHA_BONFERRONI = ALPHA / N_TESTS     # 0.025

    r_rewards  = np.array([r.total_reward   for r in remarl_results])
    b_rewards  = np.array([r.total_reward   for r in baseline_results])
    r_coverage = np.array([r.coverage_score for r in remarl_results])
    b_coverage = np.array([r.coverage_score for r in baseline_results])

    # Paired t-tests
    _t_rew, p_value = stats.ttest_rel(r_rewards, b_rewards)
    _t_cov, p_cov   = stats.ttest_rel(r_coverage, b_coverage)

    # Cohen's d on reward differences
    diff_rew = r_rewards - b_rewards
    cohen_d  = diff_rew.mean() / (diff_rew.std(ddof=1) + 1e-8)

    # 95% CI for mean paired difference (reward)
    n = len(diff_rew)
    ci_rew = stats.t.interval(0.95, n - 1,
                               loc=diff_rew.mean(), scale=stats.sem(diff_rew))
    # 95% CI for mean paired difference (coverage)
    diff_cov = r_coverage - b_coverage
    ci_cov   = stats.t.interval(0.95, n - 1,
                                 loc=diff_cov.mean(), scale=stats.sem(diff_cov))

    result = {
        "n":                          n,
        "alpha_bonferroni":           round(ALPHA_BONFERRONI, 4),
        "remarl_mean_reward":         round(float(r_rewards.mean()), 4),
        "baseline_mean_reward":       round(float(b_rewards.mean()), 4),
        "delta_reward":               round(float(diff_rew.mean()), 4),
        "ci95_delta_reward":          (round(float(ci_rew[0]), 4), round(float(ci_rew[1]), 4)),
        "p_value":                    round(float(p_value), 4),
        "significant_reward":         p_value < ALPHA_BONFERRONI,
        "cohen_d":                    round(float(cohen_d), 3),
        "effect_size":                ("large"  if abs(cohen_d) > 0.8 else
                                       "medium" if abs(cohen_d) > 0.5 else "small"),
        "remarl_mean_coverage":       round(float(r_coverage.mean()), 4),
        "baseline_mean_coverage":     round(float(b_coverage.mean()), 4),
        "delta_coverage":             round(float(diff_cov.mean()), 4),
        "ci95_delta_coverage":        (round(float(ci_cov[0]), 4), round(float(ci_cov[1]), 4)),
        "coverage_p_value":           round(float(p_cov), 4),
        "significant_coverage":       p_cov < ALPHA_BONFERRONI,
    }

    sig_r   = "✓ SIGNIFICANT" if result["significant_reward"]   else "✗ not significant"
    sig_cov = "✓ SIGNIFICANT" if result["significant_coverage"] else "✗ not significant"

    print(f"\n{'─'*66}")
    print(f"  REMARL vs MARE Random Baseline — Statistical Comparison")
    print(f"  (Bonferroni-corrected alpha = {ALPHA_BONFERRONI}  [{N_TESTS} tests])")
    print(f"{'─'*66}")
    print(f"  Episodes evaluated   : {result['n']}")
    print(f"  REMARL mean reward   : {result['remarl_mean_reward']}")
    print(f"  MARE mean reward     : {result['baseline_mean_reward']}")
    print(f"  Delta reward         : {result['delta_reward']:+.4f}  "
          f"95% CI {result['ci95_delta_reward']}")
    print(f"  p-value (reward)     : {result['p_value']}  {sig_r}")
    print(f"  Cohen's d            : {result['cohen_d']}  ({result['effect_size']} effect)")
    print(f"  REMARL mean coverage : {result['remarl_mean_coverage']}")
    print(f"  MARE mean coverage   : {result['baseline_mean_coverage']}")
    print(f"  Delta coverage       : {result['delta_coverage']:+.4f}  "
          f"95% CI {result['ci95_delta_coverage']}")
    print(f"  p-value (coverage)   : {result['coverage_p_value']}  {sig_cov}")
    print(f"{'─'*66}\n")

    return result
