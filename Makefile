# remarl/Makefile
# Common development commands

.PHONY: install setup test train eval run clean

install:
	pip install -r requirements.txt

setup:
	python sim/scenario_gen.py
	python sim/re_env.py

test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v --tb=short -m "not slow"

run:
	python run_episode.py

train:
	python train.py --config configs/remarl_config.yaml

train-collector:
	python train.py --role collector --episodes 500

train-all:
	python train.py --role all

eval:
	python evaluate.py --checkpoint data/checkpoints/collector_final

tensorboard:
	tensorboard --logdir data/logs/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -f data/episodes.db
	rm -f data/scenarios/all_scenarios.json
