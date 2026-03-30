import openenv.core as openenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from epilepsure.env import EpilepsyEnv

def train():
    """
    Train a PPO agent on the EpilepsySafety-v0 environment.
    """
    print("Initializing EpilepsySafety-v0 environment...")
    
    env = EpilepsyEnv()
    
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
