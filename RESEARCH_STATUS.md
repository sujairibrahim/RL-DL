# REMARL Research Status Report

_Generated 2026-07-22 from a full-codebase audit of `remarl2_nvdia_copy_V1`._

## 1. Executive Summary

**You're on the right path.** The core method — a PPO policy trained inside a
Gymnasium environment wrapping real LLM agents, scored against a semantic-embedding
oracle, evaluated with a proper paired statistical design — is sound and produces a
genuine, statistically significant result for the **collector** role. That said,
there's one real bug and one scope gap that need to be closed before the "6-agent,
RL beats MARE" framing is fully defensible:

- **Core result holds up**: collector's RL policy beats the faithful-MARE action
  sequence with `total_reward` +0.056 (p<.0001, Cohen's d=1.11) and `precision`
  +0.269 (p<.0001, d=2.12) on a paired n=22 design (t-test + Wilcoxon agree,
  Bonferroni/Holm corrected). It also beats a random policy (+0.036, p=.006,
  d=0.65). This is a legitimate, publishable effect for one role.
- **Negotiator — your 6th agent and the paper's headline differentiator — is not
  RL-trained.** It's a fully implemented LLM agent (prompts, factory registration,
  action-type mappings) but has zero checkpoints, is excluded from
  `training.agent_roles`, is excluded from every training/eval CLI, and is only
  ever invoked inside the interactive human-in-the-loop script — never inside an
  RL episode. Right now the RL contribution covers 3 of 6 agents, not 6.
- **Checker's benchmark results are all zero** — a genuine bug, not a limitation.
  Checker's `approve_and_document` action maps to `ActionType.WRITE_SRS`, but the
  permission check only allows `WRITE_SRS` for the documenter role. Every checker
  episode's terminal action silently fails, so no SRS text is ever produced to
  score.
- **Modeler's requirement-F1 table is also all zero**, but for a structural reason:
  modeler's actions extract entities/relations, they never write "shall" prose, so
  the shall-statement extractor used for scoring has nothing to match against.
  This needs a modeler-appropriate metric, not a bug fix.
- Test coverage is real but narrow (~26 tests covering reward/state-encoder/env/
  oracle/scenario-gen only); the LLM agents, the RL↔LLM bridge, the entire `eval/`
  package, and all root scripts have zero tests. No CI. Dependency floors in
  `requirements.txt` are far below what's actually installed and validated.
- There's a meaningful dead-code surface (an old parallel agent-base
  implementation, an unused pipeline orchestrator, an unused action-space
  definition, and a never-invoked "multi-agent episode" mode) that doesn't break
  anything today but will confuse a reviewer or collaborator reading the repo.

**Bottom line**: the RL mechanism itself isn't the risk. The risk to your central
claim is scope accuracy — fix the checker bug, decide what to do about negotiator,
and get all three trained roles onto the same statistical footing as collector.
Do that and you have a defensible full comparison instead of a promising partial one.

---

## 2. What's Been Done (verified, with evidence)

| Component | State |
|---|---|
| **6 LLM agents** (`mare/agents/`) | All real, fully-prompted implementations — stakeholder, collector, modeler, checker, documenter, **negotiator** — wired through `AgentFactory` and `AbstractAgent` (`mare/agents/base.py`). |
| **RL environment** (`sim/re_env.py`) | Real Gymnasium `Env`, `Discrete(4)` action space, `AGENT_ACTION_MAP` per role including a full negotiator entry. |
| **State encoder** (`rl/state_encoder.py`) | Confirmed exact 1544-dim encoding (4×384 sentence-BERT embeddings + 5-dim phase one-hot + 3 scalars), real `all-MiniLM-L6-v2` embeddings, not mocked. |
| **Reward engine** (`rl/reward.py`) | Working immediate reward (clarity + consistency + coverage_delta), heuristic/keyword-based by design — a documented simplification, not a bug. |
| **Oracle** (`sim/oracle.py`) | Real sentence-BERT cosine-similarity scoring against ground-truth requirements (coverage, precision, conflict, NFR) — genuine semantic measurement, not mocked. |
| **Episode memory** | SQLite-backed (`rl/memory.py`), 295 episodes logged (Apr 28 – Jun 10, 2026). |
| **PPO training** | `create_ppo_policy()` (SB3), trained checkpoints exist and are current (Jun 10, 2026) for **collector, modeler, checker** — 10 intermediate + 1 final checkpoint each. |
| **Scenario generator** (`sim/scenario_gen.py`) | 55 domain templates across 14 sectors — exceeds the documented "30 templates / 6 sectors." Undocumented, but a strength, not a gap. |
| **Primary statistical result** (`eval/run_paired_eval.py` + `eval/analyze_paired.py`) | Paired t-test, Wilcoxon signed-rank, Cohen's d, 95% CI, Bonferroni + Holm correction, win/tie/loss — n=22, collector role, vs both MAREFaithfulPolicy and MARERandomPolicy. |
| **Supplementary eval** (`eval/benchmark.py`, `eval/mare_style_eval.py`) | Per-role oracle results (n=20) for collector/modeler/checker, plus MARE-style requirement P/R/F1 at exact/token/semantic match levels. |

---

## 3. What Went Well

- **The core causal claim survives scrutiny.** Correct paired design (identical
  scenarios across policies), two independent test statistics agreeing (t-test and
  Wilcoxon), large effect sizes, multiple-comparison correction applied. This is
  the strongest asset in the project.
- **The measurement instrument is credible.** Both the oracle and the state
  encoder use genuine sentence-embedding similarity rather than keyword heuristics
  — a reviewer can't dismiss the results as "just string matching."
- **Scenario coverage is generous** — 55 domains across 14 sectors gives real
  breadth for generalization claims.
- **Layer separation is respected** in the live code path (`mare/` → `rl/` →
  `sim/` → `eval/`), even though legacy modules linger alongside it.

---

## 4. What Went Wrong

- **Checker action-permission bug (all-zero results).** `mare/rl_adapter.py` maps
  checker's `approve_and_document` → `ActionType.WRITE_SRS`, but
  `mare/agents/base.py`'s `can_perform_action()` only permits `WRITE_SRS` for the
  documenter role. Every checker episode's terminal action fails
  (`AgentExecutionError`, logged 60 times in `data/logs/manual/eval_checker.log`),
  so no SRS text is ever produced — hence 0.0 across every requirement-F1 metric.
- **Negotiator isn't RL-trained**, and — architecturally — `RESimEnv` never runs
  true multi-agent episodes (the `agent_role="multi"` path exists in code but is
  never invoked by any script). Every real training/eval episode is single-role,
  so even a hypothetical negotiator-training run couldn't meaningfully exercise
  conflict resolution without first enabling that path — negotiator would be
  acting on a workspace no other RL agent had populated.
- **Modeler has no meaningful requirement-F1 metric** — its action set writes
  entities/relations, not "shall" prose, so reusing the shall-statement extractor
  guarantees zero. Needs an entity/relation-F1 metric instead.
- **Statistical treatment is uneven across roles.** Only collector has the full
  paired comparison (vs both baselines, Wilcoxon + Holm correction). Modeler and
  checker only have single-arm `benchmark.py` oracle numbers — no vs-random arm,
  no Wilcoxon. The "n=22, 80% power" methodology described in `CLAUDE.md`
  currently backs 1 of 3 trained roles.
- **`conflict` metric shows zero variance** (constant 0.400) across every
  policy/role in the paired eval — it isn't currently discriminating between
  systems, which weakens any claim specifically about conflict-handling quality.
- **Checkpoint/result mismatch.** The n=22 paired-eval numbers (Jun 8) and the n=20
  benchmark numbers (Jun 14–15) were produced by two different generations of the
  collector checkpoint (Jun 7 vs Jun 10). For a clean paper narrative these need to
  come from the same trained model.
- **No training-dynamics visibility.** No TensorBoard event files exist anywhere
  despite docs describing `tensorboard --logdir data/logs/` — reward/entropy/KL
  curves aren't currently inspectable, only final checkpoints.
- **`episodes.db` domain field is unpopulated** for 291/295 rows (a plumbing gap:
  `RESimEnv.step()`'s info dict never sets `"domain"`), which blocks any
  domain-stratified analysis of stored episodes.
- **Test coverage gap.** ~26 tests exist, all confined to reward/state-encoder/
  env/oracle/scenario-gen. Zero tests for the LLM agents, `rl_adapter.py` (the
  RL↔LLM bridge), `rl/trainer.py`, the entire `eval/` package, and all root
  scripts (`train.py`, `evaluate.py`, `srs_pipeline.py`). No CI pipeline at all.
- **Dependency drift with no lockfile.** `requirements.txt` floors
  (`langchain>=0.2.0`, `gymnasium>=0.29.0`, `sentence-transformers>=2.7.0`) are far
  below what's actually installed and validated (`langchain 1.3.4`,
  `gymnasium 1.2.3`, `sentence-transformers 5.5.1`) — these are major-version jumps
  with real breaking-change history. A fresh clone would not reproduce your
  validated environment.
- **Dead-code surface**: `mare/agents/base_agent.py` + `mare/prompts/
  prompt_builder.py` (an old, superseded agent-base pattern), `mare/pipeline.py`
  (a full orchestrator that's never imported), and `rl/actionspace.py` (an
  alternate, inconsistent action-space definition) all sit unused alongside the
  real implementation — low risk today, high risk of confusing a reviewer or a
  future collaborator.

---

## 5. What's Remaining — Recommended Path Forward

Ordered by leverage (fix correctness and result-integrity first, then breadth):

1. **Fix the checker bug.** Either allow `WRITE_SRS` for checker in
   `can_perform_action()`, or remap `approve_and_document` to a checker-appropriate
   `ActionType`. Re-run the checker benchmark afterward.
2. **Re-run the primary paired eval on current checkpoints for all three trained
   roles**, so every number in the paper comes from the same generation of models.
   Extend the `--role` support already present in `benchmark.py` into
   `run_paired_eval.py`'s CLI if it isn't already there.
3. **Make an explicit decision on negotiator** — this is the highest-leverage
   choice for how defensible your "6-agent" framing is:
   - **Option A (fast, honest):** scope the paper's claim precisely — "6-agent
     pipeline; 3 roles RL-trained (collector, modeler, checker), 3 roles use fixed
     MARE policy (stakeholder, negotiator, documenter)." This is accurate today
     and requires no new code.
   - **Option B (bigger lift, stronger claim):** enable the `"multi"` multi-agent
     episode path in `RESimEnv`, add negotiator to `training.agent_roles`, train
     it, and evaluate it like the others. This is what would let you honestly say
     "6 agents, RL-driven" — but it's a real engineering project on its own
     (multi-role episode design, reward attribution across roles, etc.), not a
     small patch.
4. **Add a modeler-specific quality metric** (entity/relation F1) instead of
   reusing the shall-statement extractor.
5. **Investigate the zero-variance `conflict` metric** in the oracle — either fix
   the scoring formula so it discriminates between policies, or drop/replace it
   before it appears in the paper as evidence of anything.
6. **Add tests** for `rl_adapter.py`, the `eval/` package's statistical functions,
   and at least a smoke test per LLM agent's prompt construction (mockable without
   live Ollama/NVIDIA calls). Wire a minimal CI workflow to run `make test-fast`
   on push.
7. **Pin dependencies** — add a lockfile or tighten `requirements.txt` ranges to
   match what's actually validated, so results are reproducible from a clean clone.
8. **Prune or clearly quarantine dead code** (`base_agent.py`, `prompt_builder.py`,
   `pipeline.py`, `actionspace.py`) so the live architecture is unambiguous.
9. **Re-enable TensorBoard logging** for future training runs so reward/entropy
   curves exist as supporting evidence of convergence, not just final checkpoints.

---

## 6. Answer to "Am I on the right path?"

**Yes.** The methodology — paired design, two independent baselines, real
statistical tests with correction, genuine embedding-based scoring — is sound, and
the collector result is a real, significant effect with a large effect size. The
risk to your central claim isn't the RL mechanism; it's scope accuracy. Negotiator
isn't RL-trained yet, checker's numbers are broken by a fixable bug, and only one
of three trained roles has the full paired statistical treatment. Closing items 1–3
above is what turns this from "promising partial result" into "defensible full
comparison" — and is achievable without rethinking the underlying approach.
