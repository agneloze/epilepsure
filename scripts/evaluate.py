import sys
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from epilepsure.env import EpilepsyEnv, EpilepsyAction

class OpenEnvToGymWrapper(gym.Env):
    def __init__(self, openenv_env):
        super().__init__()
        self.env = openenv_env
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(49152,), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset(seed=seed)
        obs_array = np.array(obs.obs, dtype=np.uint8)
        return obs_array, obs.metadata

    def step(self, action):
        openenv_action = EpilepsyAction(prediction=int(action))
        obs = self.env.step(openenv_action)
        obs_array = np.array(obs.obs, dtype=np.uint8)
        return obs_array, float(obs.reward), bool(obs.done), False, obs.metadata

def evaluate(model_path, num_episodes=100):
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    raw_env = EpilepsyEnv()
    env = OpenEnvToGymWrapper(raw_env)
    
    stats = {
        "correct_danger": 0,
        "correct_safe": 0,
        "missed_red": 0,
        "missed_bw": 0,
        "false_alarms": 0,
        "total_reward": 0.0
    }
    
    print(f"Evaluating over {num_episodes} episodes...")
    
    for i in range(num_episodes):
        obs, info = env.reset()
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        stats["total_reward"] += reward
        stat_key = info.get("stat")
        if stat_key in stats:
            stats[stat_key] += 1
            
    print("\n--- Evaluation Results ---")
    print(f"Average Reward: {stats['total_reward'] / num_episodes:.2f}")
    print(f"Correct Dangers Caught: {stats['correct_danger']}")
    print(f"Correct Safes Identified: {stats['correct_safe']}")
    print(f"False Alarms: {stats['false_alarms']}")
    print(f"Missed BW Flickers: {stats['missed_bw']}")
    print(f"Missed RED Flickers (Critical!): {stats['missed_red']}")
    
    accuracy = (stats['correct_danger'] + stats['correct_safe']) / num_episodes * 100
    print(f"Overall Accuracy: {accuracy:.1f}%")

if __name__ == "__main__":
    model_file = "models/epilepsy_agent_v2"
    if os.path.exists(model_file + ".zip"):
        evaluate(model_file)
    else:
        # Fallback to v1 if v2 doesn't exist
        model_file = "models/epilepsy_agent_v1"
        if os.path.exists(model_file + ".zip"):
            evaluate(model_file)
        else:
            print("Error: No model files found in models/ folder.")
