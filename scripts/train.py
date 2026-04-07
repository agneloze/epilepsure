"""
Train a PPO agent on any of the three EpilepsyEnv tasks.

Usage:
    python scripts/train.py --task task1 --steps 50000
    python scripts/train.py --task task2 --steps 50000
    python scripts/train.py --task task3 --steps 100000
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.epilepsure_environment import EpilepsyEnv
from models import EpilepsyAction

OBS_DIM = 64 * 64 * 3   # single-frame flat observation


class EpilepsyGymWrapper(gym.Env):
    """
    Wraps EpilepsyEnv in the gymnasium interface expected by Stable Baselines 3.

    observation_space: Box(0, 255, (12288,), uint8)
    action_space:      Discrete(3) for task1/2  |  Discrete(5) for task3
    """
    TASK_ACTION_COUNTS = {"task1": 3, "task2": 4, "task3": 5}

    def __init__(self, task_id: str = "task1", seed: int | None = None):
        super().__init__()
        self.task_id = task_id
        self.env = EpilepsyEnv(task_id=task_id, seed=seed)
        n_actions = self.TASK_ACTION_COUNTS[task_id]
        self.action_space      = gym.spaces.Discrete(n_actions)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(OBS_DIM,), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset(seed=seed)
        return np.array(obs.frame, dtype=np.uint8), obs.metadata or {}

    def step(self, action: int):
        obs = self.env.step(EpilepsyAction(decision=int(action)))
        return (
            np.array(obs.frame, dtype=np.uint8),
            float(obs.reward),
            bool(obs.done),
            False,
            obs.metadata or {},
        )


def train(task_id: str, total_steps: int) -> None:
    print(f"Training PPO on {task_id} for {total_steps:,} steps …")
    os.makedirs("models", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)

    def make_env():
        return EpilepsyGymWrapper(task_id=task_id)

    venv = DummyVecEnv([make_env])

    checkpoint_cb = CheckpointCallback(
        save_freq=max(total_steps // 10, 1000),
        save_path="./models/",
        name_prefix=f"epilepsy_{task_id}_ckpt",
    )

    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
    )
    model.learn(total_timesteps=total_steps, callback=checkpoint_cb)

    out_path = f"models/epilepsy_{task_id}_final"
    model.save(out_path)
    print(f"\nTraining complete. Model saved → {out_path}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",  default="task1",
                        choices=["task1", "task2", "task3"])
    parser.add_argument("--steps", type=int, default=50_000)
    args = parser.parse_args()
    train(args.task, args.steps)
