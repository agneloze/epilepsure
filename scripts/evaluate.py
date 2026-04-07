"""
Evaluate a trained PPO model on any task.

Usage:
    python scripts/evaluate.py --task task1 --model models/epilepsy_task1_final
    python scripts/evaluate.py --task task3 --episodes 50
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.epilepsure_environment import EpilepsyEnv
from models import EpilepsyAction

# Reuse wrapper from train.py
from scripts.train import EpilepsyGymWrapper


def find_latest_model(task_id: str) -> str | None:
    patterns = [
        f"models/epilepsy_{task_id}_final.zip",
        f"models/epilepsy_{task_id}_ckpt_*.zip",
    ]
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(p))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime).replace(".zip", "")


def evaluate(task_id: str, model_path: str, n_episodes: int) -> None:
    print(f"\nEvaluating {task_id} | model={model_path} | episodes={n_episodes}")
    model = PPO.load(model_path)
    env = EpilepsyGymWrapper(task_id=task_id)

    grades, rewards, stat_counts = [], [], {}

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        ep_reward = 0.0
        last_info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated
            last_info = info

        grade = last_info.get("grade", 0.0)
        grades.append(grade)
        rewards.append(ep_reward)

        stat = last_info.get("stat") or last_info.get("grade_label", "")
        stat_counts[stat] = stat_counts.get(stat, 0) + 1

    print(f"\n{'─'*50}")
    print(f"  Results over {n_episodes} episodes:")
    print(f"  avg grade  = {np.mean(grades):.4f}  (std {np.std(grades):.4f})")
    print(f"  avg reward = {np.mean(rewards):+.2f}  (std {np.std(rewards):.2f})")
    print(f"  Outcome breakdown:")
    for k, v in sorted(stat_counts.items(), key=lambda x: -x[1]):
        print(f"    {k:30s}  {v:4d}  ({100*v/n_episodes:.1f}%)")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",     default="task1",
                        choices=["task1", "task2", "task3"])
    parser.add_argument("--model",    default=None)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    model_path = args.model or find_latest_model(args.task)
    if model_path is None:
        print(f"No model found for {args.task}. Train first with scripts/train.py")
        sys.exit(1)

    evaluate(args.task, model_path, args.episodes)
