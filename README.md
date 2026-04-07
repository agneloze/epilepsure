---
title: Epilepsure
emoji: 🧠
colorFrom: red
colorTo: orange
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

# Epilepsure v2: Reinforcement Learning for Photosensitive Epilepsy Detection

**Epilepsure** is an OpenEnv-compatible reinforcement learning (RL) framework designed to train and benchmark agents on the detection of photosensitive epilepsy (PSE) triggers. It implements the **Harding Test**—the global broadcast standard for flash and pattern safety—to identify visual risks such as high-frequency strobes and saturated red flickers.

---

## Project Overview

Photosensitive epilepsy affects millions of people worldwide. Traditional moderation often relies on human review or static filters. This project treats moderation as a sequential decision-making problem:
1. **Sequential Analysis**: Agents observe one frame at a time, mimicking a human reviewer.
2. **Decision Logic**: Agents must decide whether to continue watching (costing time/resources) or commit to a "Safe" or "Danger" classification.
3. **Queue Management**: In advanced tasks, agents manage a triage queue, balancing thoroughness with efficiency.

---

## Repository Structure

```text
├── models.py                # Pydantic definitions for Observations, Actions, and Rewards
├── inference.py             # LLM-based (GPT-4o-mini) zero-shot inference script
├── openenv.yaml             # OpenEnv environment configuration metadata
├── server/
│   ├── app.py               # FastAPI server for remote RL training/inference
│   └── epilepsure_environment.py # Core environment logic, generators, and graders
├── scripts/
│   ├── train.py             # PPO training script using Stable Baselines 3
│   ├── evaluate.py          # Model evaluation and metrics reporting
│   ├── test_client.py       # Integration test for server-client communication
│   └── flicker_check.py     # Static heuristic for flicker detection
└── data/                    # Placeholder for local datasets or logs
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Virtual environment (recommended)

### Installation
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 1. Running the Local Server
The environment runs as a FastAPI server. This allows for language-agnostic interaction via HTTP.
```bash
$env:PYTHONPATH="."; .\venv\Scripts\python.exe -m server.app --task task1 --port 5000
```

### 2. Reinforcement Learning (RL)
To train an agent using Proximal Policy Optimization (PPO):
```bash
$env:PYTHONPATH="."; .\venv\Scripts\python.exe scripts/train.py --task task1 --steps 50000
```
To evaluate a trained model:
```bash
$env:PYTHONPATH="."; .\venv\Scripts\python.exe scripts/evaluate.py --task task1 --model models/epilepsy_task1_final
```

### 3. LLM-Based Inference (Baseline)
To run the vision-based LLM baseline (requires an API key):
```bash
$env:OPENAI_API_KEY="your_key_here"; $env:PYTHONPATH="."; .\venv\Scripts\python.exe inference.py
```

---

## Tasks and Evaluation

### Task 1: Binary Moderation
- **Goal**: Identify if a clip is `Safe` or `Danger`.
- **Constraint**: Maximum 4 frames.
- **Scoring**: Higher scores for early, accurate decisions.

### Task 2: Violation Classification
- **Goal**: Differentiate between Black-and-White (BW) flicker and Saturated Red flicker.
- **Constraint**: Identifying the specific Harding Rule violation.

### Task 3: Queue Triage
- **Goal**: Manage a queue of 5 clips using a "Preview" and "Full Review" system.
- **Metric**: F1-Score on danger detection with penalties for missing high-risk red flickers.

---

## Technical Specifications
- **Observation Space**: Flattened RGB frames (64x64x3), normalized to uint8.
- **Action Space**: Discrete actions (Continue, Flag Safe, Flag Danger, etc.).
- **Reward Function**: Dense rewards including step costs (-0.5) to incentivize speed and heavy penalties (-50.0) for critical safety misses.

---

## License
This project is licensed under the MIT License.
