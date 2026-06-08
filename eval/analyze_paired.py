"""
Paired statistical analysis with Bonferroni correction across the four oracle
dimensions and total reward.

Reads the JSON written by run_paired_eval.py:
    python eval/analyze_paired.py \
        --input data/benchmarks/paired_eval_<stamp>.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def paired(metric_baseline, metric_rl, name) -> dict:
    base = np.array(metric_baseline, dtype=float)
    rl   = np.array(metric_rl,       dtype=float)
    diff = rl - base
    t, p = stats.ttest_rel(rl, base)
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0
    return {
        "metric":         name,
        "n":              int(len(diff)),
        "mean_baseline":  float(base.mean()),
        "mean_rl":        float(rl.mean()),
        "mean_diff":      float(diff.mean()),
        "t":              float(t),
        "p_raw":          float(p),
        "cohen_d":        float(d),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="paired_eval JSON from run_paired_eval.py")
    args = ap.parse_args()

    rows = json.loads(Path(args.input).read_text())

    keys = ["total_reward", "coverage", "precision", "conflict", "nfr"]
    results = []
    for k in keys:
        b = [r["baseline"][k] for r in rows if r["baseline"].get(k) is not None]
        x = [r["remarl"][k]   for r in rows if r["remarl"].get(k)   is not None]
        if len(b) != len(x) or len(b) < 3:
            continue
        results.append(paired(b, x, k))

    m = len(results)
    for r in results:
        r["p_bonferroni"] = min(1.0, r["p_raw"] * m)
        r["sig_raw"]      = r["p_raw"] < 0.05
        r["sig_bonf"]     = r["p_bonferroni"] < 0.05

    n_str = str(results[0]["n"]) if results else "?"
    print(f"\nPaired evaluation, n={n_str} scenarios, m={m} comparisons\n")
    cols = ["metric", "mean_baseline", "mean_rl", "mean_diff",
            "t", "p_raw", "p_bonferroni", "cohen_d", "sig_bonf"]
    print("\t".join(cols))
    for r in results:
        print("\t".join(
            f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
            for c in cols
        ))


if __name__ == "__main__":
    main()
