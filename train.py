"""
remarl/train.py
---------------
Main REMARL training entry point.

Usage:
    python train.py                                    # use default config
    python train.py --config configs/remarl_config.yaml
    python train.py --episodes 500 --role collector
    python train.py --resume data/checkpoints/collector_ep100
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

# Load environment variables from .env (NVIDIA_API_KEY etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure log directory exists
Path("data/logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/logs/train.log", mode="a"),
    ],
)
logger = logging.getLogger("remarl.train")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_env_fn(config: dict, agent_role: str):
    """Returns a zero-arg callable that creates a fresh RESimEnv."""
    from sim.scenario_gen import ScenarioGenerator
    from sim.oracle import Oracle
    from sim.re_env import RESimEnv
    from rl.state_encoder import StateEncoder
    from rl.reward import RewardEngine

    gen     = ScenarioGenerator(config["env"]["scenario_dir"])
    oracle  = Oracle(
        coverage_threshold=config["reward"]["coverage_threshold"]
    )
    encoder = StateEncoder(
        model_name=config["state_encoder"]["model"],
        max_steps=config["env"]["max_steps_per_episode"],
    )
    reward_engine = RewardEngine(
        clarity_weight=config["reward"]["clarity_weight"],
        consistency_weight=config["reward"]["consistency_weight"],
        coverage_delta_weight=config["reward"]["coverage_delta_weight"],
    )

    # Build real MARE agents once — reuse across all episodes
    # Each agent connects to Ollama with its configured model
    from mare.agents.factory import AgentFactory
    from mare.rl_adapter import MARERLAgent

    raw_agents = AgentFactory.create_all_agents_from_config(config)

    # Wrap each real MARE agent in the RL adapter
    rl_agents = {
        role_name: MARERLAgent(raw_agents[role_name])
        for role_name in raw_agents
    }

    def env_fn():
        # Reset agent conversation history at each new episode
        for agent in rl_agents.values():
            agent.reset()

        return RESimEnv(
            scenario_gen=gen,
            oracle=oracle,
            state_encoder=encoder,
            reward_engine=reward_engine,
            agents=rl_agents,
            agent_role=agent_role,
            max_steps=config["env"]["max_steps_per_episode"],
            step_penalty=config["reward"]["step_penalty"],
        )

    return env_fn


def train(config: dict, agent_role: str, resume_from: str = None):
    from rl.policy import create_ppo_policy
    from rl.memory import EpisodeMemory
    from stable_baselines3.common.callbacks import BaseCallback

    Path(config["training"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["training"]["log_dir"]).mkdir(parents=True, exist_ok=True)
    memory = EpisodeMemory(config["memory"]["db_path"])

    env_fn = build_env_fn(config, agent_role)
    model  = create_ppo_policy(env_fn, config, agent_role)

# ── NEW: Callback to store episodes in EpisodeMemory ─────────
    class MemoryCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self._ep_count = 0

        def _on_step(self) -> bool:
            # SB3 sets 'dones' when an episode ends
            infos = self.locals.get("infos", [{}])
            for info in infos:
                if info.get("oracle_result") is not None:
                    oracle = info["oracle_result"]
                    domain = info.get("domain", "unknown")
                    self._ep_count += 1
                    memory.store(
                        episode_id=self._ep_count,
                        domain=domain,
                        reward=float(oracle.total_reward),
                        experiences=info.get("experiences", []),
                        covered_pct=float(oracle.coverage_score),
                    )
            return True  # return False to stop training early
    # ─────────────────────────────────────────────────────────────
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        from stable_baselines3 import PPO
        model = PPO.load(resume_from, env=model.get_env())

    n_episodes  = config["training"]["n_episodes"]
    steps_per_ep = config["env"]["max_steps_per_episode"]
    total_steps  = n_episodes * steps_per_ep

    logger.info(
        f"Starting training | role={agent_role} "
        f"episodes={n_episodes} total_steps={total_steps}"
    )

    # SB3 PPO trains by timestep, not episode
    # We checkpoint every save_every_n_episodes * steps_per_ep steps
    save_every = config["training"]["save_every_n_episodes"] * steps_per_ep
    eval_every = config["training"]["eval_every_n_episodes"] * steps_per_ep

    model.learn(
        total_timesteps=total_steps,
        progress_bar=True,
        tb_log_name=f"ppo_{agent_role}",
    )

    # Save final model
    final_path = Path(config["training"]["checkpoint_dir"]) / f"{agent_role}_final"
    model.save(str(final_path))
    logger.info(f"Saved final model to {final_path}")

    print(f"\nTraining complete. Episode memory stats:")
    import pprint; pprint.pprint(memory.stats())


def main():
    parser = argparse.ArgumentParser(description="Train REMARL policies")
    parser.add_argument("--config",   default="configs/remarl_config.yaml")
    parser.add_argument("--role",     default="collector",
                        choices=["collector", "modeler", "checker", "all"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--resume",   default=None, help="Path to checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.episodes:
        config["training"]["n_episodes"] = args.episodes

    if args.role == "all":
        for role in config["training"]["agent_roles"]:
            logger.info(f"\n{'='*50}\nTraining role: {role}\n{'='*50}")
            train(config, role, args.resume)
    else:
        train(config, args.role, args.resume)


if __name__ == "__main__":
    main()
