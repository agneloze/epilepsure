import random
import time
from openenv.core import SyncEnvClient, GenericEnvClient
from epilepsure.env import EpilepsyAction, EpilepsyObservation

def test_client():
    """
    Connect to the EpilepsySafety-v0 server and run 10 steps (episodes).
    """
    base_url = "http://localhost:5000"
    print(f"Connecting to {base_url}...")

    try:
        # Use GenericEnvClient which is concrete
        async_client = GenericEnvClient(base_url)
        client = SyncEnvClient(async_client)
        
        # 1. Reset the environment
        print("Resetting environment...")
        result = client.reset()
        
        # In this version of openenv/SyncEnvClient, the observation is a dict
        obs_dict = result.observation
        
        # metadata is NOT included in serialize_observation by default it seems
        # but 'obs' is there.
        obs_data = obs_dict.get("obs", [])
        print(f"Observation data length: {len(obs_data)}")
        
        # 2. Loop for 10 steps
        for i in range(1, 11):
            action_val = random.choice([0, 1])
            action = EpilepsyAction(prediction=action_val)
            
            print(f"\n--- Step {i} ---")
            print(f"Action: {'Danger' if action_val == 1 else 'Safe'} ({action_val})")
            
            step_result = client.step(action)
            
            # 3. Print reward and obs info
            reward = step_result.reward
            obs_dict = step_result.observation
            obs_data = obs_dict.get("obs", [])
            source = obs_dict.get("source", "unknown")
            print(f"Reward: {reward}")
            print(f"Obs data length: {len(obs_data)}")
            print(f"Source: {source}")
            
            if step_result.done:
                print("Episode finished. Resetting for next step...")
                result = client.reset()
                
        print("\nClient test completed successfully.")
        
    except Exception as e:
        print(f"Error during client test: {e}")

if __name__ == "__main__":
    test_client()
