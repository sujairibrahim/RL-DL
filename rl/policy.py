"""
remarl/rl/policy.py
-------------------
PPO policy creation for each REMARL agent role.
Uses Stable-Baselines3 — no manual PPO implementation needed.
"""

import logging
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

logger = logging.getLogger(__name__)


def create_ppo_policy(env_fn, config: dict, agent_role: str) -> PPO:
    """
    Create a PPO policy for one agent role.

    Args:
        env_fn:      zero-arg callable that returns a RESimEnv instance
        config:      ppo section of remarl_config.yaml (as dict)
        agent_role:  used for logging only

    Returns:
        Configured PPO model (not yet trained)
    """
    vec_env = DummyVecEnv([env_fn])

    ppo_cfg = config.get("ppo", {})
    arch    = ppo_cfg.get("policy_arch", [256, 256])

    policy_kwargs = {
        "net_arch": dict(pi=arch, vf=arch),
    }

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        n_steps=ppo_cfg.get("n_steps", 512),
        batch_size=ppo_cfg.get("batch_size", 64),
        n_epochs=ppo_cfg.get("n_epochs", 10),
        gamma=ppo_cfg.get("gamma", 0.95),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        ent_coef=ppo_cfg.get("ent_coef", 0.01),
        vf_coef=ppo_cfg.get("vf_coef", 0.5),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs,
        tensorboard_log=config.get("training", {}).get("log_dir", "data/logs/"),
        verbose=1,
    )
    logger.info(f"Created PPO policy for role={agent_role}")
    return model
