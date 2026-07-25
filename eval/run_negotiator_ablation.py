"""
eval/run_negotiator_ablation.py
================================
Negotiator Effect Ablation for REMARL.

Runs the FULL six-agent pipeline on identical scenarios in two arms:

  Arm A ("with"):    DEFAULT_PHASE_SEQUENCE as-is (includes the 2 negotiator steps)
  Arm B ("without"): the same schedule with the negotiator steps removed
                     (this replicates original 5-agent MARE behaviour)

Then oracle-scores both arms and reports paired statistics per metric.
No RL, no PPO, no checkpoints - this is a scripted fixed-schedule ablation.

PREREQUISITES (must be applied first, see Doc 3 + Deep Audit doc):
  1. rl_adapter.py: the three Checker mapping edits
  2. rl_adapter.py: input_data must include "error_report" (bug N1)
  3. sim/oracle.py: remove "shall" from resolution_words (bug N5)

USAGE:
  # Wiring check, no LLM calls, no API key needed:
  python eval/run_negotiator_ablation.py --dry-run

  # Full run (22 scenarios x 2 arms, ~1-2 h, needs NVIDIA_API_KEY):
  python eval/run_negotiator_ablation.py --n 22 --seed 42

  # Small pilot first (recommended before the full run):
  python eval/run_negotiator_ablation.py --n 3 --seed 42

Place this file at: eval/run_negotiator_ablation.py
Run from the repo root.
"""

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime

# Make repo root importable when run as `python eval/run_negotiator_ablation.py`
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import yaml

# Single source of truth for the schedule - import, never copy:
from sim.re_env import DEFAULT_PHASE_SEQUENCE


def build_schedules():
    """Arm A = full schedule; Arm B = same minus negotiator steps."""
    with_neg = list(DEFAULT_PHASE_SEQUENCE)
    without_neg = [(role, act) for (role, act) in with_neg if role != "negotiator"]
    return with_neg, without_neg


def make_workspace(scenario):
    """Replicates exactly what RESimEnv.reset() puts into a fresh workspace."""
    from mare.workspace.shared_workspace import SharedWorkspace
    ws = SharedWorkspace()
    ws.set("rough_idea", scenario.rough_idea)
    ws.set("domain", scenario.domain)
    if scenario.visible_reqs:
        ws.set(
            "initial_context",
            "Initial requirements mentioned by client:\n"
            + "\n".join(f"- {r}" for r in scenario.visible_reqs),
        )
    return ws


def run_arm(schedule, scenario, agents, oracle, verbose=True):
    """Execute one fixed-schedule pipeline pass and oracle-score the result."""
    ws = make_workspace(scenario)

    # Fresh conversation history per arm (same fix as RESimEnv.reset)
    for wrapper in agents.values():
        inner = getattr(wrapper, "agent", wrapper)
        if hasattr(inner, "reset"):
            inner.reset()

    for step_i, (role, action_name) in enumerate(schedule, 1):
        wrapper = agents.get(role)
        if wrapper is None:
            raise RuntimeError(f"No agent for role '{role}'")
        t0 = time.time()
        try:
            result = wrapper.perform_action(action_name, ws)
            err = result.get("error")
        except Exception as e:  # keep going; empty output is scored as-is
            result, err = {"output": ""}, str(e)
        if verbose:
            status = f"ERROR: {err}" if err else "ok"
            print(f"      step {step_i:2d}  {role:11s} {action_name:22s} "
                  f"({time.time()-t0:4.1f}s)  {status}")

    r = oracle.score(ws, scenario)
    return {
        "coverage": r.coverage_score,
        "precision": r.precision_score,
        "conflict": r.conflict_score,
        "nfr": r.nfr_score,
        "total": r.total_reward,
        "hallucinated": r.hallucinated_count,
        "srs_len": len(ws.get("srs_document", "") or ""),
        "error_report_len": len(ws.get("error_report", "") or ""),
    }


def dry_run_check():
    """Validate wiring without any LLM calls or API key.

    Checks that every (role, action) in both schedules has a complete
    mapping chain: ACTION_TYPE_MAP entry -> FIELD_MAP entry, and that the
    action's ActionType is one the target agent class permits.
    """
    from mare.rl_adapter import ACTION_TYPE_MAP, FIELD_MAP
    from mare.agents.base import ActionType  # noqa: F401  (import sanity)
    from mare.agents.stakeholder import StakeholderAgent
    from mare.agents.collector import CollectorAgent
    from mare.agents.modeler import ModelerAgent
    from mare.agents.checker import CheckerAgent
    from mare.agents.negotiator import NegotiatorAgent
    from mare.agents.documenter import DocumenterAgent

    role_cls = {
        "stakeholder": StakeholderAgent, "collector": CollectorAgent,
        "modeler": ModelerAgent, "checker": CheckerAgent,
        "negotiator": NegotiatorAgent, "documenter": DocumenterAgent,
    }

    with_neg, without_neg = build_schedules()
    problems = []
    for label, sched in [("WITH", with_neg), ("WITHOUT", without_neg)]:
        print(f"\n  Schedule {label} ({len(sched)} steps):")
        for role, action in sched:
            atype = ACTION_TYPE_MAP.get(action)
            field = FIELD_MAP.get(action)
            cls = role_cls[role]
            # can_perform_action is an instance method; call unbound with cls
            try:
                allowed = cls.can_perform_action(cls, atype) if atype else False
            except Exception:
                allowed = "?"
            mark = "OK " if (atype and field and allowed is True) else "!! "
            print(f"    {mark} {role:11s} {action:22s} -> "
                  f"type={getattr(atype, 'name', 'MISSING'):20s} "
                  f"field={field or 'MISSING':15s} permitted={allowed}")
            if mark == "!! ":
                problems.append((role, action))

    print()
    if problems:
        print("  DRY-RUN FAILED - fix these mappings before a real run:")
        for role, action in problems:
            print(f"    - {role}.{action}")
        sys.exit(1)
    print("  DRY-RUN PASSED - all mappings and permissions are consistent.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/remarl_train_fast.yaml")
    ap.add_argument("--n", type=int, default=22)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        dry_run_check()
        return

    config = yaml.safe_load(open(args.config))

    from sim.scenario_gen import ScenarioGenerator
    from sim.oracle import Oracle
    from mare.agents.factory import AgentFactory
    from mare.rl_adapter import MARERLAgent

    gen = ScenarioGenerator(config["env"]["scenario_dir"], seed=args.seed)
    oracle = Oracle(coverage_threshold=config["reward"]["coverage_threshold"])
    raw = AgentFactory.create_all_agents_from_config(config)
    agents = {role: MARERLAgent(a) for role, a in raw.items()}

    with_neg, without_neg = build_schedules()
    scenarios = [gen.sample() for _ in range(args.n)]

    rows = []
    t_start = time.time()
    for i, sc in enumerate(scenarios, 1):
        print(f"\n[{i}/{args.n}] {sc.domain}")
        print("   Arm A (WITH negotiator):")
        a = run_arm(with_neg, sc, agents, oracle)
        print("   Arm B (WITHOUT negotiator):")
        b = run_arm(without_neg, sc, agents, oracle)
        rows.append({"scenario": i, "domain": sc.domain, "with": a, "without": b})
        print(f"   conflict: with={a['conflict']:.3f}  without={b['conflict']:.3f}   "
              f"total: with={a['total']:.3f}  without={b['total']:.3f}")

    # ---- paired statistics -------------------------------------------------
    from scipy import stats as st
    import numpy as np

    print(f"\n{'='*72}")
    print(f"  NEGOTIATOR ABLATION - paired results over n={len(rows)} scenarios")
    print(f"{'='*72}")
    summary = {}
    for metric in ["conflict", "total", "coverage", "precision", "nfr"]:
        wa = np.array([r["with"][metric] for r in rows], dtype=float)
        wo = np.array([r["without"][metric] for r in rows], dtype=float)
        diff = wa - wo
        entry = {
            "with_mean": float(wa.mean()), "without_mean": float(wo.mean()),
            "mean_diff": float(diff.mean()),
        }
        if np.allclose(diff, 0):
            entry.update({"t_p": None, "wilcoxon_p": None, "cohens_d": 0.0,
                          "note": "no variance in differences"})
        else:
            t, tp = st.ttest_rel(wa, wo)
            try:
                _, wp = st.wilcoxon(wa, wo)
            except ValueError:
                wp = None
            entry.update({
                "t_p": float(tp),
                "wilcoxon_p": (float(wp) if wp is not None else None),
                "cohens_d": float(diff.mean() / diff.std(ddof=1))
                if diff.std(ddof=1) > 0 else 0.0,
            })
        summary[metric] = entry
        pd = entry.get("t_p")
        print(f"  {metric:10s}  with={entry['with_mean']:.3f}  "
              f"without={entry['without_mean']:.3f}  "
              f"diff={entry['mean_diff']:+.3f}  "
              f"p={pd:.4f}" if pd is not None else
              f"  {metric:10s}  with={entry['with_mean']:.3f}  "
              f"without={entry['without_mean']:.3f}  diff=0 (constant)")

    out = args.out or (
        f"data/benchmarks/negotiator_ablation_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "n": len(rows), "seed": args.seed, "config": args.config,
            "minutes": round((time.time() - t_start) / 60, 1),
            "episodes": rows, "summary": summary,
        }, f, indent=2)
    print(f"\n  Saved: {out}")
    print("  Bonferroni note: if you report all 5 metrics, the corrected alpha")
    print("  is 0.05/5 = 0.01. If you pre-register conflict as the primary")
    print("  endpoint (recommended - it IS the hypothesis), report conflict at")
    print("  alpha=0.05 and the other four as secondary/descriptive.")


if __name__ == "__main__":
    main()
