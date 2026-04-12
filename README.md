---
title: Epilepsure
emoji: 🧠
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 5000
tags:
  - openenv
  - reinforcement-learning
  - content-moderation
  - safety
license: mit
pinned: false
---

<div align="center">

# 🧠 Epilepsure: RL Environment for Photosensitive Epilepsy Content Moderation

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-brightgreen.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-RL-orange.svg)]()

<br/>

*A safety-critical reinforcement learning framework for the detection of photosensitive epilepsy (PSE) triggers.*

</div>

<hr/>

By implementing the **Harding Test rules** (the global broadcast standard for flash and pattern safety), Epilepsure allows AI agents to make sequential, frame-by-frame moderation decisions across tasks of increasing difficulty to identify risks such as high-frequency strobes and saturated <span style="color:red; font-weight:bold;">red flickers</span>.

## ✨ Key Features

- 🔬 **Sequential Analysis**: Agents observe one frame at a time, mimicking a human reviewer's process.
- 📈 **Graded Difficulty Tasks**: From binary safe/danger classification to detailed violation type identification and triage queue management.
- 🌐 **OpenEnv Compatible**: Fully compliant with OpenEnv specifications for seamless integration and benchmarking.
- 🎯 **Dense Reward Structure**: Includes step costs to incentivize speed and heavy penalties for critical safety misses.

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Framework**: FastAPI (for the underlying server API)
- **RL Libraries**: Stable Baselines 3, Gymnasium, TensorBoard
- **Core Dependencies**: NumPy, OpenCV-Python, Pydantic, Pillow
- **Deployment**: Docker, Hugging Face Spaces (via OpenEnv)

## 📋 Prerequisites

> [!IMPORTANT]
> Ensure you have the following installed before proceeding:
> - Python 3.10 or higher
> - Docker (optional, but highly recommended for consistent deployment)
> - A virtual environment tool (`venv`, `conda`, `uv`, etc.)

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/agneloze/Epilepsure-RL.git
cd Epilepsure-RL
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv

# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

Install the project in editable mode along with its dependencies:

```bash
pip install -r requirements.txt
```

### 4. Environment Setup

Configure the following basic environment variables (you can export them or place them in an `.env` file if deploying via Docker):

| Variable        | Description                            | Default |
|-----------------|----------------------------------------|---------|
| `EPILEPSY_TASK` | The active RL task (task1, task2, task3) | `task1` |
| `PORT`          | The port the FastAPI server runs on    | `5000`  |

### 5. Start Development Server

Run the FastAPI environment server locally:

```bash
python server.py
# Or use the installed CLI command:
# epilepsure-server
```

You should see the server starting on `http://0.0.0.0:5000`.

### 6. Run RL Training or Inference

**To train an agent using Proximal Policy Optimization (PPO):**
```bash
python scripts/train.py --task task1 --steps 50000
```

**To evaluate a trained model:**
```bash
python scripts/evaluate.py --task task1 --model models/epilepsy_task1_final
```

**To run the vision-based LLM baseline:**
```bash
export OPENAI_API_KEY="your_api_key"
python inference.py
```

## 🏗️ Architecture

### Directory Structure

```text
├── server/
│   ├── app.py                      # FastAPI server logic and routing
│   └── epilepsure_environment.py   # Core Gymnasium RL environment, frame generation, rule logic
├── scripts/
│   ├── train.py                    # RL PPO training script
│   ├── evaluate.py                 # RL evaluation script
│   ├── test_client.py              # Server-client integration testing
│   └── flicker_check.py            # Static heuristic rules for flicker
├── tests/                          # Automated tests
├── models.py                       # Pydantic schemas for Actions, Observations, etc.
├── inference.py                    # Zero-shot GPT-4o-mini inference logic
├── client.py                       # SyncEnvClient for interacting with the server
├── openenv.yaml                    # OpenEnv standard metadata config
├── pyproject.toml                  # Python packaging and dependency config
├── Dockerfile                      # Container definition
└── requirements.txt                # Legacy dependencies list
```

### Request Lifecycle for the Environment Server

1. A client issues a `POST /reset` to initialize a new episode or `POST /step` to take an action.
2. The `app.py` FastAPI server parses the request and validates the action payload using Pydantic schemas (`models.py`).
3. The request is passed to the core `epilepsure_environment.py` instance representing the environment.
4. The environment computes the frame logic, generates the new observation (usually a 64x64 RGB flattened array), assigns the reward, checks if the episode is done, and provides additional metadata.
5. The result is returned via JSON to the RL algorithm or testing client.

### Task Specifications

**Task 1: Binary Moderation**
- **Action Space**: 3 discrete actions (`continue_watching`, `flag_safe`, `flag_danger`).
- **Goal**: Decide Safe vs Danger observing sequentially up to 4 frames.
- **Grader**: `epilepsure.graders.grade_task1`

**Task 2: Violation Classification**
- **Action Space**: 4 discrete actions (`continue_watching`, `flag_safe`, `flag_bw_flicker`, `flag_red_flicker`).
- **Goal**: Identify the exact violation type (or Safe).

**Task 3: Queue Triage**
- **Action Space**: 5 discrete actions (`skip_safe`, `escalate_danger`, `request_full_review`, `commit_safe`, `commit_danger`).
- **Goal**: Route a queue of 5 clips, optimizing for F1 score heavily penalizing missed red flickers.

## ⚙️ Environment Variables

### Core Configuration

| Variable          | Description                                  | Default |
|-------------------|----------------------------------------------|---------|
| `EPILEPSY_TASK`   | Dictates which of the 3 tasks the server will run | `task1` |
| `PORT`            | FastAPI binding port                         | `5000`  |
| `OPENAI_API_KEY`  | For running baseline LLM evaluation inference | -       |

## 📜 Available Scripts

| Command                                                    | Description                                           |
|------------------------------------------------------------|-------------------------------------------------------|
| `python server.py`                                         | Starts the FastAPI environment server                 |
| `epilepsure-server`                                        | (If installed) Starts the FastAPI environment server  |
| `python inference.py`                                      | Runs the vision-based LLM baseline evaluation         |
| `python scripts/train.py --task <task> --steps <N>`        | Initiates a PPO training run                          |
| `python scripts/evaluate.py --task <task> --model <path>`  | Evaluates a saved stable-baselines3 model             |

## 🧪 Testing

To run the test suite (if configured) or test integration with the running server:

```bash
# Ensure server is running in another terminal
python server.py

# Run the integration test client
python scripts/test_client.py
```

## 📦 Deployment

### Docker

The project provides a `Dockerfile` that packages the application, its OpenCV dependencies, and starts the FastAPI server natively.

```bash
# 1. Build the Docker image
docker build -t epilepsure .

# 2. Run the container
docker run -p 5000:5000 \
  -e EPILEPSY_TASK=task1 \
  -e PORT=5000 \
  epilepsure
```
You can now interface with the environment at `http://localhost:5000`.

### Hugging Face Spaces (OpenEnv)

Because of the presence of `openenv.yaml` and the standard Hugging Face headers, you can deploy this as a Docker Space on Hugging Face. When pushing to a fresh Hugging Face Space configured as `Docker`, the platform will automatically build using the supplied `Dockerfile` and expose it securely on port 5000.

## 🆘 Troubleshooting

> [!WARNING]
> ### OpenCV Library Errors
> **Error:** `ImportError: libGL.so.1: cannot open shared object file: No such file or directory` or similar
> 
> **Solution:**
> OpenCV requires certain system dependencies. The provided `Dockerfile` already installs them (`libglib2.0-0` or `libgl1-mesa-glx`), but if you are running locally without Docker on Linux, install the dependencies manually:
> ```bash
> sudo apt-get update && sudo apt-get install libgl1
> ```
> *(Alternatively, uninstall `opencv-python` and install `opencv-python-headless`)*

> [!TIP]
> ### PPO Dependencies Issue
> **Error:** Module not found for `stable_baselines3` or `gymnasium`.
> 
> **Solution:** Make sure you installed the dependencies with `pip install .` or `pip install -r requirements.txt`. Re-activate your virtual environment.

> [!CAUTION]
> ### Connection Refused
> **Error:** Client outputs `Connection refused` when starting RL training script or inference.
> 
> **Solution:** The remote RL architecture relies on the server running first. Keep `python server.py` running in a separate terminal before executing the RL or inference scripts.
