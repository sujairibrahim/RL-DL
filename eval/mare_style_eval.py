"""
MARE-style evaluation: Precision, Recall, F1 at three matching levels
(exact, token, semantic) against ground-truth requirements.

Standalone — only requires numpy and sentence_transformers (which you
already use in sim/oracle.py).

Drop in eval/. Use as:
    from eval.mare_style_eval import evaluate_episode, aggregate, format_table
"""
from __future__ import annotations
import re
import json
from typing import List, Dict, Iterable
from pathlib import Path
from dataclasses import dataclass

import numpy as np


# ─── 1. Requirement extraction ────────────────────────────────────────────
_SHALL = re.compile(r'^\s*(?:[-*\d.)\s]*)?(?P<text>(?:The\s+\w+\s+)?(?:shall|must|should)\s.+?\.?)\s*$',
                    re.IGNORECASE | re.MULTILINE)

def extract_requirements(text: str) -> List[str]:
    """Pull 'shall/must/should' statements from any text blob (req_draft or SRS)."""
    if not text:
        return []
    reqs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _SHALL.match(line)
        if m:
            reqs.append(_normalize(m.group("text")))
    # de-duplicate while preserving order
    seen, out = set(), []
    for r in reqs:
        if r not in seen:
            seen.add(r); out.append(r)
    return out

def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)   # strip punctuation
    return s


# ─── 2. Matching primitives ───────────────────────────────────────────────
def exact_matches(gen: List[str], gt: List[str]) -> List[int]:
    """Return list[i] = index in gt that matches gen[i], or -1."""
    matched = [-1] * len(gen)
    used_gt = set()
    for i, g in enumerate(gen):
        for j, t in enumerate(gt):
            if j in used_gt: continue
            if g == t:
                matched[i] = j; used_gt.add(j); break
    return matched

def token_matches(gen: List[str], gt: List[str], threshold: float = 0.5) -> List[int]:
    """Jaccard similarity over tokens >= threshold."""
    matched = [-1] * len(gen)
    used_gt = set()
    gen_tok = [set(g.split()) for g in gen]
    gt_tok  = [set(t.split()) for t in gt]
    for i, gs in enumerate(gen_tok):
        if not gs: continue
        best_j, best_sim = -1, 0.0
        for j, ts in enumerate(gt_tok):
            if j in used_gt or not ts: continue
            jacc = len(gs & ts) / len(gs | ts)
            if jacc >= threshold and jacc > best_sim:
                best_j, best_sim = j, jacc
        if best_j != -1:
            matched[i] = best_j; used_gt.add(best_j)
    return matched

def semantic_matches(gen_emb: np.ndarray, gt_emb: np.ndarray,
                     threshold: float = 0.65) -> List[int]:
    """Cosine similarity >= threshold, greedy 1-to-1 assignment."""
    if gen_emb.size == 0 or gt_emb.size == 0:
        return [-1] * gen_emb.shape[0]
    # normalize
    gn = gen_emb / (np.linalg.norm(gen_emb, axis=1, keepdims=True) + 1e-9)
    tn = gt_emb  / (np.linalg.norm(gt_emb,  axis=1, keepdims=True) + 1e-9)
    sim = gn @ tn.T                     # (G, T)
    matched = [-1] * sim.shape[0]
    used_gt = set()
    # greedy: for each generated, pick best unused gt above threshold
    order = np.argsort(-sim.max(axis=1))  # process highest-confidence first
    for i in order:
        row = sim[i].copy()
        for j in used_gt: row[j] = -1
        j = int(np.argmax(row))
        if row[j] >= threshold:
            matched[i] = j; used_gt.add(j)
    return matched


# ─── 3. P / R / F1 ────────────────────────────────────────────────────────
def prf(matches: List[int], n_gen: int, n_gt: int) -> Dict[str, float]:
    tp = sum(1 for m in matches if m >= 0)
    fp = n_gen - tp
    fn = n_gt - tp
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"P": p, "R": r, "F1": f, "TP": tp, "FP": fp, "FN": fn}


# ─── 4. Episode-level evaluation ──────────────────────────────────────────
def evaluate_episode(generated_text: str,
                     ground_truth_reqs: List[str],
                     embedder=None,
                     token_threshold: float = 0.5,
                     semantic_threshold: float = 0.65) -> Dict[str, Dict[str, float]]:
    """Returns: {'exact': {P, R, F1, ...}, 'token': {...}, 'semantic': {...}}."""
    gen = extract_requirements(generated_text)
    gt  = [_normalize(t) for t in ground_truth_reqs]

    res = {}
    res["exact"]    = prf(exact_matches(gen, gt),              len(gen), len(gt))
    res["token"]    = prf(token_matches(gen, gt, token_threshold), len(gen), len(gt))

    if embedder is not None and gen and gt:
        ge = embedder.encode(gen, show_progress_bar=False)
        te = embedder.encode(gt,  show_progress_bar=False)
        res["semantic"] = prf(semantic_matches(ge, te, semantic_threshold),
                              len(gen), len(gt))
    else:
        res["semantic"] = {"P": 0.0, "R": 0.0, "F1": 0.0, "TP": 0, "FP": 0, "FN": 0}

    res["_meta"] = {"n_generated": len(gen), "n_ground_truth": len(gt)}
    return res


# ─── 5. Aggregation across scenarios + table formatting ───────────────────
def aggregate(episode_results: List[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Returns mean and std for each (level, metric) across episodes."""
    out = {}
    for level in ["exact", "token", "semantic"]:
        out[level] = {}
        for k in ["P", "R", "F1"]:
            vals = np.array([ep[level][k] for ep in episode_results])
            out[level][k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
    return out

def format_table(aggregates: Dict[str, Dict[str, Dict[str, float]]],
                 system_name: str = "System") -> str:
    """Print a MARE-paper-style 3x3 table (rows = match level, cols = P/R/F1)."""
    lines = []
    lines.append(f"\n  {system_name}")
    lines.append(f"  {'Level':<10} {'Precision':>14} {'Recall':>14} {'F1':>14}")
    lines.append("  " + "-" * 56)
    for level in ["exact", "token", "semantic"]:
        row = aggregates[level]
        lines.append(f"  {level.capitalize():<10} "
                     f"{row['P']['mean']:>7.3f} ± {row['P']['std']:>4.3f}   "
                     f"{row['R']['mean']:>7.3f} ± {row['R']['std']:>4.3f}   "
                     f"{row['F1']['mean']:>7.3f} ± {row['F1']['std']:>4.3f}")
    return "\n".join(lines)


# ─── 6. Standalone runner ────────────────────────────────────────────────
def main():
    """Usage:
        python -m eval.mare_style_eval --episodes data/benchmarks/paired_eval_*.json

    Where each episode JSON has fields:
        {'baseline': {'srs_text': ..., 'ground_truth': [...]},
         'remarl':   {'srs_text': ..., 'ground_truth': [...]},
         'random':   {'srs_text': ..., 'ground_truth': [...]}}
    """
    import argparse
    from sentence_transformers import SentenceTransformer

    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, help="JSON file with paired episode outputs")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    embedder = SentenceTransformer(args.model)
    data = json.loads(Path(args.episodes).read_text())

    by_system = {"baseline": [], "remarl": [], "random": []}
    for ep in data:
        for sys_name in by_system:
            if sys_name not in ep:
                continue
            text = ep[sys_name].get("srs_text", "") or ep[sys_name].get("req_draft", "")
            gt   = ep[sys_name].get("ground_truth", []) or ep.get("ground_truth", [])
            if text and gt:
                by_system[sys_name].append(evaluate_episode(text, gt, embedder=embedder))

    print(f"\n=== MARE-style evaluation, n_scenarios = {len(data)} ===")
    for sys_name, eps in by_system.items():
        if eps:
            agg = aggregate(eps)
            print(format_table(agg, system_name=sys_name.upper()))


if __name__ == "__main__":
    main()