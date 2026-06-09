# REMARL — Multi-Agent Reinforcement Learning for Requirements Engineering

REMARL extends MARE (a multi-agent LLM framework) with PPO policies that learn agent-level action selection to improve requirements engineering (RE) outputs over time.

Key features:
- Human-in-the-loop SRS generation using local LLMs (Ollama by default).
- RL training and evaluation on synthetic scenarios with a Gymnasium environment.
- Paper-ready evaluation scripts and statistical analysis utilities.

This README covers: prerequisites, setup, common commands, architecture, and pointers to important files.

**If you need deeper developer notes**, see CLAUDE.md for an extended project guide.

**Prerequisites**

- Python 3.8+ and the Python dependencies from `requirements.txt`.
- Ollama running locally if you plan to use the HITL pipeline or any LLM-backed agents:

```bash
ollama serve
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull llama3.2:3b
ollama pull gemma4:latest
```

Note: `gemma4` is used by the stakeholder and documenter agents by default in the included configs.

**Installation**

```bash
pip install -r requirements.txt
```

**One-time setup**

Build the scenario cache and verify the environment before training or evaluation:

```bash
python sim/scenario_gen.py
python sim/re_env.py
```

**Common workflows & commands**

- Human-in-the-loop SRS generation (HITL pipeline):

```bash
python srs_pipeline.py
python srs_pipeline.py --resume output/srs_session_<name>_<timestamp>.json
```

- RL training (PPO):

```bash
python train.py --config configs/remarl_config.yaml        # default training (collector role by default)
python train.py --role modeler --episodes 500             # train specific role
python train.py --role all                                # train all trainable roles sequentially
python train.py --resume data/checkpoints/collector_ep100 # resume from a checkpoint
```

- Evaluation & benchmarking:

```bash
python evaluate.py --checkpoint data/checkpoints/collector_final
python eval/run_paired_eval.py --checkpoint data/checkpoints/collector_final --n 22
python eval/benchmark.py --checkpoint data/checkpoints/collector_final --n_eval 50
python eval/analyze_paired.py --input data/benchmarks/paired_eval_<timestamp>.json
```

- Quick experiment tracking:

```bash
tensorboard --logdir data/logs/
```

**Makefile shortcuts**

```bash
make install    # pip install -r requirements.txt
make setup      # build scenario cache + verify env
make test       # run tests
make run        # run a single episode runner
make train      # default training run
make train-all  # train all roles sequentially
make eval       # run evaluation pipeline
make clean      # remove caches and temporary files
```

**Project layout (high level)**

```text
mare/      ← language layer: agents, prompts, shared workspace
rl/        ← RL layer: PPO policies, reward, state encoder, memory
sim/       ← simulation: Gymnasium env, scenario generator, oracle
eval/      ← evaluation & benchmarking tools
configs/   ← YAML config files
data/      ← scenarios, logs, checkpoints, episode DB
output/    ← generated HITL session JSON and markdown SRS exports
tests/     ← unit and integration tests
```

**Architecture & data flow (short)**

- The environment (`sim/re_env.py`) exposes a Discrete(4) action space per role. Each integer maps to a role-specific MARE action via `AGENT_ACTION_MAP`.
- `mare/rl_adapter.py` converts action names into LLM-driven agent calls and writes outputs to the `SharedWorkspace`.
- `rl/state_encoder.py` encodes workspace text fields and phase scalars into the observation vector used by PPO (state dim: 1544 by default).
- `rl/reward.py` computes immediate reward components (clarity, consistency, coverage_delta).
- `sim/oracle.py` computes terminal reward by comparing produced SRS text against scenario ground truth.

For a detailed, developer-oriented walkthrough, see CLAUDE.md.

**Important invariants**

- State vector dimension is 1544 (4 × 384 + 5 + 3). If you change sentence encoders, update `rl/state_encoder.py` and the YAML `state_encoder.state_dim`.
- Action space is Discrete(4). Keep `AGENT_ACTION_MAP` (in `sim/re_env.py`) and `ACTION_TYPE_MAP`/`FIELD_MAP` (in `mare/rl_adapter.py`) in sync when adding/removing actions.
- Reward weights must sum to 1.0 — this is asserted in `sim/oracle.py`.

**Configuration**

The main config files are in `configs/`. The default experiment config is `configs/remarl_config.yaml`. Key sections:
- `llm` — provider and model per agent
- `state_encoder` — sentence model and embedding dim
- `reward` — weight breakdown for reward components
- `training` — PPO and training hyperparameters

**Running tests**

```bash
pytest tests/ -v --tb=short
pytest tests/test_sim/test_re_env.py -v -k "test_reset"
```

**Output & artifacts**

- Trained checkpoints: `data/checkpoints/`
- Scenario cache: `data/scenarios/all_scenarios_expanded_V1.json`
- Episodes DB: `data/episodes.db`
- HITL session snapshots and exported SRS docs: `output/`

**Further reading**

- Developer notes and deep-dive: [CLAUDE.md](CLAUDE.md)
- Configuration examples: `configs/remarl_config.yaml`

If you'd like, I can:
- Run the test suite and report failures
- Add a quick-start script that automates the one-time setup
- Create a short CONTRIBUTING.md with development checks and linting rules

Feel free to tell me which of these you'd like next.
