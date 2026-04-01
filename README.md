# Epilepsure: AI Safety for Photosensitive Epilepsy

**Epilepsure** is an OpenEnv-compatible reinforcement learning environment designed to detect and flag video content that violates the **Harding Test** safety standards for photosensitive epilepsy (PSE).

This project was built for the **Meta PyTorch OpenEnv Hackathon** to provide a standardized way for AI agents to learn safety-critical visual detection tasks.

## Overview
Photosensitive epilepsy affects approximately 1 in 4,000 people. Certain visual triggers—such as rapid flashes or high-contrast patterns—can induce seizures. Epilepsure provides a simulated environment where agents are trained to observe 4-frame video clips and decide whether the content is `Safe` or `Danger`.

## The Harding Test (7 Core Rules)
The environment implements detection logic based on the industry-standard Harding Test (Ofcom & ITU-R BT.1702 guidelines):

1.  **The 7-Transitions Rule:** No more than 3.5 flashes per second (measured as 7 alternating luminance transitions in any 1-second window).
2.  **The 25% Area Rule:** Flashing is restricted if the affected region occupies more than 25% of the viewer's field of vision.
3.  **Luminance Threshold:** A transition is only counted as a "flash" if the change in luminance exceeds 20 cd/m².
4.  **Saturated Red Flashes:** Saturated red flashes are high-risk; transitions to/from saturated red are subject to stricter frequency limits.
5.  **Stationary Spatial Patterns:** High-contrast patterns (stripes, checks) occupying >25% of the screen with >5 cycles are flagged.
6.  **Moving Patterns:** Scrolling or rotating high-contrast patterns that create a flickering effect are restricted.
7.  **Rapid Scene Cuts:** More than 3 rapid cuts between high-contrast scenes in a single second can trigger a failure.

## Reward Logic
The environment uses a weighted reward system to prioritize safety (minimizing False Negatives, especially for red triggers):

*   **+10:** Correctly flagging a `Danger` clip.
*   **+1:** Correctly flagging a `Safe` clip.
*   **-5:** False Alarm (flagging a safe clip as dangerous).
*   **-20:** Missed Black-and-White Flicker (False Negative).
*   **-50:** **Missed Saturated Red Flicker** (Highest penalty for high-risk False Negatives).

## How to Run
The environment is containerized for easy deployment and interaction via the OpenEnv protocol.

### Prerequisites
*   Docker
*   Python 3.10+ (for client-side testing)

### Using Docker
1.  **Build the Image:**
    ```bash
    docker build -t epilepsure-v1 .
    ```
2.  **Run the Server:**
    ```bash
    docker run -p 5000:5000 epilepsure-v1
    ```
    The environment will now be served at `http://localhost:5000`.

### Local Development
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/Epilepsure-RL.git
    cd Epilepsure-RL
    ```
2.  **Set Up a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Start the Environment Server:**
    ```bash
    python server.py
    ```
5.  **Run a Client Test:**
    ```bash
    python scripts/test_client.py
    ```

## Training and Evaluation
The project includes pre-configured scripts for training a PPO (Proximal Policy Optimization) agent and evaluating its performance.

### Training the Agent
To start training the agent for 10,000 steps (default):
```bash
python scripts/train.py
```
This will save model checkpoints in the `models/` directory and logs in `tensorboard_logs/`.

### Evaluating the Agent
Once a model is trained, you can evaluate its performance across 100 episodes to see its accuracy and reward stats:
```bash
python scripts/evaluate.py
```
The script will automatically look for the latest model in the `models/` folder.

## Monitoring with TensorBoard
You can monitor training progress (reward, loss, episode length) in real-time using TensorBoard:

1.  **Start TensorBoard:**
    ```bash
    tensorboard --logdir=tensorboard_logs
    ```
2.  **Open in Browser:**
    Navigate to `http://localhost:6006` to view the dashboards.

## Project Structure
*   `epilepsure/`: Core package containing the environment logic (`env.py`).
*   `scripts/`: Utility scripts for training, testing, and checking flickers.
*   `data/`: Directory for real-world `.npy` and `.mp4` samples.
*   `models/`: Saved model weights and checkpoints.
*   `server.py`: FastAPI server entry point for the environment.
