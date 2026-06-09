"""
Paired analysis: paired t-test, Wilcoxon signed-rank, 95% CI, Cohen's d,
Bonferroni and Holm corrections for both t-test and Wilcoxon p-values.
Handles constant columns gracefully.

Usage:
    python eval/analyze_paired.py \\
        --input data/benchmarks/paired_eval_<timestamp>.json \\
        --comparison remarl_vs_baseline

FIXES APPLIED
─────────────
FIX [1] CRITICAL — None-value filtering was independent per system, so if one
        system had a missing value for a metric that the other had, the lists
        ended up different lengths. A paired t-test on unequal, non-corresponding
        lists is statistically invalid — it compares the wrong scenarios against
        each other. Fixed: now filters rows where EITHER system has a missing
        value, so both lists always correspond to the same set of scenarios.

FIX [2] — Bonferroni correction was applied to t-test p-values only.
        Wilcoxon p-values were not corrected, so using p_w for significance
        decisions would inflate the family-wise false-positive rate.
        Fixed: p_bonf_w is now computed alongside p_bonf (t-test version).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

try:
    from statsmodels.stats.multitest import multipletests
    HAVE_SM = True
except ImportError:
    HAVE_SM = False


def paired_stats(base: list, rl: list, name: str) -> dict:
    """
    Compute paired statistics for a single metric.

    Both `base` and `rl` must be the same length and correspond to the same
    scenarios (i.e. already filtered jointly — see FIX [1] in main()).
    """
    base = np.asarray(base, dtype=float)
    rl   = np.asarray(rl,   dtype=float)
    diff = rl - base
    n    = len(diff)

    mean_diff = float(diff.mean())
    sd_diff   = float(diff.std(ddof=1)) if n > 1 else 0.0
    wins      = int((diff > 0).sum())
    ties      = int((diff == 0).sum())
    losses    = int((diff < 0).sum())

    # Constant-difference case: t and Wilcoxon both undefined.
    if sd_diff == 0:
        return {
            "metric":       name,
            "n":            n,
            "mean_base":    float(base.mean()),
            "mean_rl":      float(rl.mean()),
            "diff":         mean_diff,
            "ci95_lo":      mean_diff,
            "ci95_hi":      mean_diff,
            "t":            float("nan"),
            "p_t":          1.0,
            "p_bonf":       1.0,        # t-test Bonferroni
            "p_bonf_w":     1.0,        # Wilcoxon Bonferroni (FIX [2])
            "wilcoxon_W":   float("nan"),
            "p_w":          1.0,
            "cohen_d":      0.0,
            "wins":         wins,
            "ties":         ties,
            "losses":       losses,
            "note":         "CONSTANT — no variance",
        }

    t_stat, p_t = stats.ttest_rel(rl, base)

    try:
        w_stat, p_w = stats.wilcoxon(rl, base, zero_method="wilcox")
    except ValueError:
        # All differences are zero — already handled above, but guard here too.
        w_stat, p_w = float("nan"), 1.0

    se = sd_diff / np.sqrt(n)
    ci_lo, ci_hi = stats.t.interval(0.95, df=n - 1, loc=mean_diff, scale=se)
    d = mean_diff / sd_diff

    return {
        "metric":       name,
        "n":            n,
        "mean_base":    float(base.mean()),
        "mean_rl":      float(rl.mean()),
        "diff":         mean_diff,
        "ci95_lo":      float(ci_lo),
        "ci95_hi":      float(ci_hi),
        "t":            float(t_stat),
        "p_t":          float(p_t),
        "p_bonf":       None,       # filled in after all metrics are computed
        "p_bonf_w":     None,       # FIX [2]: filled in after all metrics
        "wilcoxon_W":   float(w_stat),
        "p_w":          float(p_w),
        "cohen_d":      float(d),
        "wins":         wins,
        "ties":         ties,
        "losses":       losses,
        "note":         "",
    }


def main():
    ap = argparse.ArgumentParser(
        description="Paired statistical analysis of run_paired_eval.py output."
    )
    ap.add_argument("--input", required=True, help="JSON file from run_paired_eval.py")
    ap.add_argument(
        "--comparison", default="remarl_vs_baseline",
        choices=["remarl_vs_baseline", "remarl_vs_random", "random_vs_baseline"],
    )
    args = ap.parse_args()

    rows = json.loads(Path(args.input).read_text())
    metrics = ["total_reward", "coverage", "precision", "conflict", "nfr"]

    if args.comparison == "remarl_vs_baseline":
        a_key, b_key = "baseline", "remarl"
    elif args.comparison == "remarl_vs_random":
        a_key, b_key = "random", "remarl"
    else:
        a_key, b_key = "baseline", "random"

    results = []
    for m in metrics:
        # FIX [1]: filter rows where EITHER system has a missing value.
        # Old code filtered each system independently, so the lists could be
        # different lengths — a paired t-test on mismatched lists compares the
        # wrong scenarios and produces invalid p-values.
        valid_rows = [
            r for r in rows
            if r[a_key].get(m) is not None and r[b_key].get(m) is not None
        ]
        base_vals = [r[a_key][m] for r in valid_rows]
        rl_vals   = [r[b_key][m] for r in valid_rows]

        if not base_vals:
            print(f"  [skip] {m}: no valid paired rows found.")
            continue

        results.append(paired_stats(base_vals, rl_vals, m))

    if not results:
        print("No results to analyse.")
        return

    n_tests = len(results)

    # ── Bonferroni correction for both t-test and Wilcoxon (FIX [2]) ──────────
    pvals_t = [r["p_t"] for r in results]
    pvals_w = [r["p_w"] for r in results]

    for r, pt, pw in zip(results, pvals_t, pvals_w):
        r["p_bonf"]   = min(1.0, pt * n_tests)   # t-test Bonferroni
        r["p_bonf_w"] = min(1.0, pw * n_tests)   # FIX [2]: Wilcoxon Bonferroni

    # ── Holm correction (statsmodels) ─────────────────────────────────────────
    if HAVE_SM:
        _, p_holm_t, _, _ = multipletests(pvals_t, alpha=0.05, method="holm")
        _, p_holm_w, _, _ = multipletests(pvals_w, alpha=0.05, method="holm")
        for r, ph_t, ph_w in zip(results, p_holm_t, p_holm_w):
            r["p_holm_t"] = float(ph_t)
            r["p_holm_w"] = float(ph_w)  # FIX [2]: Wilcoxon Holm too

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n=== Paired analysis: {args.comparison}, n={results[0]['n']} ===\n")

    col_note = "  p_holm_t" if HAVE_SM else ""
    hdr = (
        f"{'metric':14} {'base':>7} {'rl':>7} {'Δ':>8} {'95% CI':>20} "
        f"{'t':>6} {'p_t':>8} {'p_bonf':>8} {'p_bonf_w':>9} "
        f"{'p_w':>8} {'d':>6} {'W/T/L':>10}{col_note}"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        ci   = f"[{r['ci95_lo']:+.3f},{r['ci95_hi']:+.3f}]"
        wtl  = f"{r['wins']}/{r['ties']}/{r['losses']}"
        note = f"  {r['note']}" if r["note"] else ""
        holm = f"  {r.get('p_holm_t', '—'):>8}" if HAVE_SM else ""

        print(
            f"{r['metric']:14} {r['mean_base']:>7.3f} {r['mean_rl']:>7.3f} "
            f"{r['diff']:>+8.3f} {ci:>20} {r['t']:>6.2f} "
            f"{r['p_t']:>8.4f} {r['p_bonf']:>8.4f} {r['p_bonf_w']:>9.4f} "
            f"{r['p_w']:>8.4f} {r['cohen_d']:>+6.2f} {wtl:>10}"
            f"{note}{holm}"
        )

    # ── Save structured results ───────────────────────────────────────────────
    out = Path(args.input).with_suffix(
        f".{args.comparison.replace('_vs_', 'vs')}.analysis.json"
    )
    out.write_text(
        json.dumps(
            {"comparison": args.comparison, "n_tests": n_tests, "results": results},
            indent=2,
            default=str,
        )
    )
    print(f"\nSaved structured results to {out}\n")


if __name__ == "__main__":
    main()