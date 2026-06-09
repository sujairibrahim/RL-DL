# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

REMARL (Multi-Agent Reinforcement Learning for Requirements Engineering) extends MARE (a multi-agent LLM framework) with PPO RL policies. Six LLM agents (stakeholder, collector, modeler, checker, negotiator, documenter) collaborate to produce IEEE 830 SRS documents. RL policies learn which agent actions to take at each step to maximise downstream document quality.

The project has **two distinct operating modes**:

1. **HITL Pipeline** — interactive, human-guided SRS generation using local Ollama: `python srs_pipeline.py`
2. **RL Training + Evaluation** — PPO policies trained on synthetic scenarios: `python train.py` / `python evaluate.py`

## Prerequisites

Ollama must be running locally before either mode can work:
```bash
ollama serve
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull llama3.2:3b
ollama pull gemma4:latest   # used by stakeholder and documenter by default
```

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# One-time setup: build scenario cache and verify the Gymnasium env
python sim/scenario_gen.py
python sim/re_env.py

# Interactive SRS generation (HITL mode)
python srs_pipeline.py
python srs_pipeline.py --resume output/srs_session_<name>_<timestamp>.json

# RL training
python train.py --config configs/remarl_config.yaml   # train collector (default)
python train.py --role modeler --episodes 500
python train.py --role all                             # train all RL roles sequentially
python train.py --resume data/checkpoints/collector_ep100

# Evaluation (simple — delegates to eval/benchmark.py)
python evaluate.py --checkpoint data/checkpoints/collector_final
python evaluate.py --checkpoint data/checkpoints/collector_final --domain patient_portal

# Primary paired evaluation: REMARL vs MAREFaithfulPolicy vs random on identical scenarios
python eval/run_paired_eval.py --checkpoint data/checkpoints/collector_final --n 22
python eval/run_paired_eval.py --checkpoint data/checkpoints/collector_final --n 22 --domain patient_portal

# Statistical analysis of paired results (t-test, Wilcoxon, Cohen's d, Bonferroni/Holm)
python eval/analyze_paired.py --input data/benchmarks/paired_eval_<timestamp>.json
python eval/analyze_paired.py --input data/benchmarks/paired_eval_<timestamp>.json --comparison remarl_vs_random

# MARE-style requirement-level evaluation (exact / token / semantic P/R/F1)
python -m eval.mare_style_eval --episodes data/benchmarks/paired_eval_<timestamp>.json

# Full benchmark with per-episode CSV and LaTeX table
python eval/benchmark.py --checkpoint data/checkpoints/collector_final --n_eval 50

# Experiment tracking
tensorboard --logdir data/logs/
```

### Makefile shortcuts
```bash
make install       # pip install
make setup         # build scenario cache + verify env
make test          # run all tests
make test-fast     # skip slow tests
make run           # python run_episode.py
make train         # default training run
make train-all     # train all roles
make eval          # evaluate collector_final checkpoint
make clean         # remove __pycache__, episodes.db, scenario cache
```

### Running a single test
```bash
pytest tests/test_rl/test_reward.py -v
pytest tests/test_sim/test_re_env.py -v -k "test_reset"
```

## Architecture

### Layer separation

```
mare/          ← Language layer (LLM agents, prompts, workspace)
rl/            ← RL layer (PPO policy, reward, state encoding, memory)
sim/           ← Simulation (Gymnasium env, scenario generator, oracle)
eval/          ← Benchmarking and metrics
```

### Data flow — RL training

1. `train.py` calls `build_env_fn()` which wires together all components.
2. `RESimEnv` ([sim/re_env.py](sim/re_env.py)) is a Gymnasium env. Each episode = one RE project.
   - `reset()` samples a `Scenario` from `ScenarioGenerator`, creates a fresh `SharedWorkspace`.
   - `step(action: int 0-3)` maps the integer to a role-specific MARE action name (via `AGENT_ACTION_MAP`), calls the agent, scores the output, and advances the phase.
3. `MARERLAgent` ([mare/rl_adapter.py](mare/rl_adapter.py)) is the bridge: translates action name strings → `ActionType` enums, builds `input_data` from the workspace, calls `agent.execute_action()`, and writes output back to the workspace.
4. `StateEncoder` ([rl/state_encoder.py](rl/state_encoder.py)) encodes 4 workspace text fields (384-dim each via `all-MiniLM-L6-v2`) + a 5-dim phase one-hot + 3 scalars = **1544-dim** float32 observation.
5. `RewardEngine` ([rl/reward.py](rl/reward.py)) computes the immediate reward (clarity + consistency + coverage_delta, each in [-1,1]).
6. `Oracle` ([sim/oracle.py](sim/oracle.py)) computes the terminal reward at episode end using cosine similarity between the produced SRS and ground-truth requirements from the scenario.
7. Stable-Baselines3 `PPO` trains an `MlpPolicy` over this environment. Episode experiences are stored in `EpisodeMemory` (SQLite at `data/episodes.db`).

### Data flow — HITL pipeline

`srs_pipeline.py` runs the same agents sequentially across 6 stages (user stories → Q&A → draft → modelling → QA check → conflict negotiation → SRS doc), prompting the human for feedback after each stage. Session state is checkpointed to `output/srs_session_*.json` after every stage so sessions can be resumed with `--resume`.

### Agent system (mare/)

- `AbstractAgent` ([mare/agents/base.py](mare/agents/base.py)) — base class; holds LangChain LLM, conversation history (capped to system prompt + last 4 messages to avoid context bloat), and action history.
- `AgentFactory.create_all_agents_from_config()` ([mare/agents/factory.py](mare/agents/factory.py)) — reads `llm.*` section of the YAML and instantiates all six agents with their per-role model and token limits.
- Provider support: `ollama` (default), `openai`, `anthropic`, `nvidia` (NVIDIA NIM via OpenAI-compatible endpoint) — controlled by `llm.provider` in the YAML config.
- `SharedWorkspace` ([mare/workspace/shared_workspace.py](mare/workspace/shared_workspace.py)) — the dict-like shared memory that all agents read/write during an episode.

### Scenario generator ([sim/scenario_gen.py](sim/scenario_gen.py))

30 domain templates across 6 sectors (e-commerce, healthcare, education, fintech, logistics/IoT, government/entertainment). Each template has ground-truth requirements, NFRs, stakeholder personas, domain entities, and intentional conflicts. On first run, templates are expanded and cached to `data/scenarios/all_scenarios_expanded_V1.json`. During training, `hide_fraction` (default 0.25) of requirements are hidden from the initial prompt — agents must elicit them.

### Evaluation layer (eval/)

- `eval/benchmark.py` — full benchmark runner; compares REMARL, `MAREFaithfulPolicy` (canonical MARE action sequence `[propose×3, draft, refine, flag]`), and `MARERandomPolicy` on oracle metrics + requirement-level F1. Outputs per-episode CSV and LaTeX table to `data/benchmarks/eval_<role>_<timestamp>/`.
- `eval/run_paired_eval.py` — **primary evaluation script for the paper**; runs N paired episodes (default 22, chosen for 80% power at d≈0.643) across all three policies on identical scenarios and saves raw results to `data/benchmarks/paired_eval_<timestamp>.json`.
- `eval/analyze_paired.py` — statistical analysis of paired results: paired t-test, Wilcoxon signed-rank, Cohen's d, 95% CI, Bonferroni/Holm corrections, and win/tie/loss counts across `total_reward`, `coverage`, `precision`, `conflict`, `nfr`.
- `eval/mare_style_eval.py` — MARE-paper-style requirement evaluation: extracts "shall/must/should" statements and computes P/R/F1 at exact, token (Jaccard ≥ 0.5), and semantic (cosine ≥ 0.65 via sentence-BERT) matching levels.
- `eval/mare_eval.py` — `MAREEvaluator` and `EvaluationSuite` used by `benchmark.py`.

Current results (collector role, n=22): REMARL vs MAREFaithfulPolicy — `total_reward` +0.056 (p<.0001, d=1.11), `precision` +0.269 (p<.0001, d=2.12). See `results_table_1.txt` and `results_table_2.txt`.

### Config ([configs/remarl_config.yaml](configs/remarl_config.yaml))

All hyperparameters live here. Key sections: `llm` (provider, models per agent), `state_encoder`, `reward` (weight breakdown), `env`, `ppo`, `training`, `eval`, `memory`, `wandb`. The trainable roles are listed under `training.agent_roles`; others use fixed MARE policies.

## Key invariants

- **State dim is 1544** (4 × 384 + 5 + 3). If you change the sentence model or add workspace fields, update `state_encoder.state_dim` in the YAML and the constant in `rl/state_encoder.py`.
- **Action space is Discrete(4)** per agent role. `AGENT_ACTION_MAP` in `sim/re_env.py` and `ACTION_TYPE_MAP` / `FIELD_MAP` in `mare/rl_adapter.py` must stay in sync when actions are added or renamed.
- **Reward weights must sum to 1.0** — enforced by assertions in `Oracle.__init__`.
- **Ollama health is checked** at pipeline start (`LLMClient.check_ollama_health()`). No LLM call will be attempted unless Ollama responds.
- Conversation history is hard-capped to system prompt + last 4 messages in `AbstractAgent._generate_response()` to prevent context explosion across long episodes.
