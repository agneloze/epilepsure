import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import openenv.core as openenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from epilepsure.env import EpilepsyEnv

def train():
    """
    Train a PPO agent on the EpilepsySafety-v0 environment.
    """
    print("Initializing EpilepsySafety-v0 environment...")
    
    import gymnasium as gym
    import numpy as np
    from epilepsure.env import EpilepsyAction
    
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
            info = obs.metadata or {}
            return obs_array, info

        def step(self, action):
            openenv_action = EpilepsyAction(prediction=int(action))
            obs = self.env.step(openenv_action)
            obs_array = np.array(obs.obs, dtype=np.uint8)
            reward = float(obs.reward)
            terminated = bool(obs.done)
            truncated = False
            info = obs.metadata or {}
            return obs_array, reward, terminated, truncated, info

    raw_env = EpilepsyEnv()
    env = OpenEnvToGymWrapper(raw_env)
    
    # Stable Baselines3 requires a vectorized environment
    # DummyVecEnv is used to wrap a single environment instance
    venv = DummyVecEnv([lambda: env])
    
    print("Starting training for 10,000 steps...")
    
    # Using MlpPolicy as a general-purpose starting point.
    model = PPO("MlpPolicy", venv, verbose=1)
    
    # Train the model
    model.learn(total_timesteps=10000)
    
    # Save the final model
    model.save("models/epilepsy_agent_v1")
    print("\nTraining complete. Model saved in 'models/epilepsy_agent_v1'.")

if __name__ == "__main__":
    train()
