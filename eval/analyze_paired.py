"""
Paired analysis: paired t-test, Wilcoxon signed-rank, 95% CI, Cohen's d,
Bonferroni and Holm corrections. Handles constant columns gracefully.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats

try:
    from statsmodels.stats.multitest import multipletests
    HAVE_SM = True
except ImportError:
    HAVE_SM = False


def paired_stats(base, rl, name):
    base = np.asarray(base, dtype=float)
    rl   = np.asarray(rl,   dtype=float)
    diff = rl - base
    n    = len(diff)

    mean_diff = float(diff.mean())
    sd_diff   = float(diff.std(ddof=1)) if n > 1 else 0.0
    wins      = int((diff > 0).sum())
    ties      = int((diff == 0).sum())
    losses    = int((diff < 0).sum())

    if sd_diff == 0:
        return {
            "metric": name, "n": n,
            "mean_base": float(base.mean()), "mean_rl": float(rl.mean()),
            "diff": mean_diff,
            "ci95_lo": mean_diff, "ci95_hi": mean_diff,
            "t": float("nan"), "p_t": 1.0,
            "wilcoxon_W": float("nan"), "p_w": 1.0,
            "cohen_d": 0.0,
            "wins": wins, "ties": ties, "losses": losses,
            "note": "CONSTANT — no variance"
        }

    t_stat, p_t = stats.ttest_rel(rl, base)
    try:
        w_stat, p_w = stats.wilcoxon(rl, base, zero_method="wilcox")
    except ValueError:
        w_stat, p_w = float("nan"), 1.0

    se     = sd_diff / np.sqrt(n)
    ci_lo, ci_hi = stats.t.interval(0.95, df=n-1, loc=mean_diff, scale=se)
    d      = mean_diff / sd_diff

    return {
        "metric": name, "n": n,
        "mean_base": float(base.mean()), "mean_rl": float(rl.mean()),
        "diff": mean_diff,
        "ci95_lo": float(ci_lo), "ci95_hi": float(ci_hi),
        "t": float(t_stat), "p_t": float(p_t),
        "wilcoxon_W": float(w_stat), "p_w": float(p_w),
        "cohen_d": float(d),
        "wins": wins, "ties": ties, "losses": losses,
        "note": ""
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--comparison", default="remarl_vs_baseline",
                    choices=["remarl_vs_baseline", "remarl_vs_random",
                             "random_vs_baseline"])
    args = ap.parse_args()

    rows = json.loads(Path(args.input).read_text())
    metrics = ["total_reward", "coverage", "precision", "conflict", "nfr"]

    if args.comparison == "remarl_vs_baseline":
        a, b = "baseline", "remarl"
    elif args.comparison == "remarl_vs_random":
        a, b = "random", "remarl"
    else:
        a, b = "baseline", "random"

    results = []
    for m in metrics:
        base_vals = [r[a][m] for r in rows if r[a].get(m) is not None]
        rl_vals   = [r[b][m] for r in rows if r[b].get(m) is not None]
        results.append(paired_stats(base_vals, rl_vals, m))

    # Family-wise correction across the 5 metrics
    pvals_t = [r["p_t"] for r in results]
    for r, p in zip(results, pvals_t):
        r["p_bonf"] = min(1.0, p * len(pvals_t))
    if HAVE_SM:
        _, p_holm, _, _ = multipletests(pvals_t, alpha=0.05, method='holm')
        for r, ph in zip(results, p_holm):
            r["p_holm"] = float(ph)

    print(f"\n=== Paired analysis: {args.comparison}, n={results[0]['n']} ===\n")
    hdr = f"{'metric':14} {'base':>7} {'rl':>7} {'Δ':>8} {'95% CI':>20} "\
          f"{'t':>6} {'p_t':>8} {'p_bonf':>8} {'p_w':>8} {'d':>6} {'W/T/L':>10}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        ci = f"[{r['ci95_lo']:+.3f},{r['ci95_hi']:+.3f}]"
        wtl = f"{r['wins']}/{r['ties']}/{r['losses']}"
        note = f"  {r['note']}" if r['note'] else ""
        print(f"{r['metric']:14} {r['mean_base']:>7.3f} {r['mean_rl']:>7.3f} "
              f"{r['diff']:>+8.3f} {ci:>20} {r['t']:>6.2f} "
              f"{r['p_t']:>8.4f} {r['p_bonf']:>8.4f} {r['p_w']:>8.4f} "
              f"{r['cohen_d']:>+6.2f} {wtl:>10}{note}")

    out = Path(args.input).with_suffix(".analysis.json")
    out.write_text(json.dumps({"comparison": args.comparison, "results": results},
                              indent=2, default=str))
    print(f"\nSaved structured results to {out}\n")


if __name__ == "__main__":
    main()