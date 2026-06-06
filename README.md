# REMARL - Multi-Agent Reinforcement Learning for Requirements Engineering

REMARL extends MARE (Multi-Agent RE Framework) with PPO-based policies that learn how to elicit, negotiate, and specify software requirements more effectively over time.

The project has two operating modes:

1. Human-in-the-loop SRS generation with local Ollama models.
2. RL training and evaluation on synthetic RE scenarios.

## Prerequisites

Before running either mode, make sure Ollama is installed and running locally:

```bash
ollama serve
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull llama3.2:3b
ollama pull gemma4:latest
```

Python dependencies are listed in `requirements.txt`.

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# Build the scenario cache once
python sim/scenario_gen.py

# Verify the Gymnasium environment
python sim/re_env.py

# Run the default training loop
python train.py --config configs/remarl_config.yaml
```

## Common commands

```bash
# Human-in-the-loop SRS generation
python srs_pipeline.py
python srs_pipeline.py --resume output/srs_session_<name>_<timestamp>.json

# Training
python train.py --config configs/remarl_config.yaml
python train.py --role modeler --episodes 500
python train.py --role all
python train.py --resume data/checkpoints/collector_ep100

# Evaluation
python evaluate.py --checkpoint data/checkpoints/collector_final
python evaluate.py --checkpoint data/checkpoints/collector_final --domain patient_portal

# Experiment tracking
tensorboard --logdir data/logs/
```

Makefile shortcuts are also available:

```bash
make install
make setup
make test
make test-fast
make run
make train
make train-all
make eval
make clean
```

## Project layout

```text
remarl/
├── mare/           MARE agents, prompts, and shared workspace
├── rl/             RL policies, rewards, state encoding, memory
├── sim/            Gymnasium environment, oracle, scenario generator
├── eval/           Benchmarking and metrics
├── configs/        YAML configuration files
├── data/           Scenarios, logs, and checkpoints
├── output/         Generated SRS sessions and markdown exports
└── tests/          Unit and integration tests
```

## Architecture

MARE provides the language layer: agents, prompts, and shared workspace state. REMARL adds the decision layer on top: PPO policies choose which action each role should take, the environment scores those actions, and the reward signal pushes the system toward better downstream requirements quality.

### Data flow

1. `train.py` builds the Gymnasium environment from the config.
2. `sim/re_env.py` maps discrete actions to role-specific agent actions.
3. `mare/rl_adapter.py` executes the agent action and writes results back to the shared workspace.
4. `rl/state_encoder.py` converts workspace state into the observation vector.
5. `rl/reward.py` computes the immediate reward.
6. `sim/oracle.py` computes the terminal reward against the scenario ground truth.

## Testing

```bash
pytest tests/ -v --tb=short
pytest tests/test_sim/test_re_env.py -v -k "test_reset"
```
