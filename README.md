---
title: Epilepsure-RL
emoji: 
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

# Epilepsure v2 — RL Environment for Photosensitive Epilepsy Content Moderation

**Epilepsure** is an OpenEnv-compatible reinforcement learning environment where agents learn to moderate video content for photosensitive epilepsy (PSE) triggers using the **Harding Test** — the international broadcast safety standard.

## Motivation

Photosensitive epilepsy affects approximately 1 in 4,000 people. Platforms that host video content are responsible for detecting and warning about visual triggers such as strobing lights and rapid colour changes. This environment trains and benchmarks RL agents to perform that moderation task the way a human reviewer would: frame by frame, deciding when to act and how to classify violations.

## Why This Is Genuine RL (Not Supervised Learning)

- **Sequential decisions**: the agent sees one frame per step, not the full clip at once.
- **Actions change what comes next**: choosing `continue_watching` reveals the next frame.
- **Trajectory reward**: a `−0.5` cost per review step incentivises decisive early action.
- **Task 3 resource allocation**: the agent manages a limited review budget across a 5-clip queue — a classic RL planning problem.

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `frame` | float[12288] | Current video frame flattened (64×64×3 uint8) |
| `step_num` | int | Step index within the episode |
| `max_steps` | int | 4 for tasks 1/2, 20 for task 3 |
| `task_id` | str | Active task identifier |
| `frames_seen` | int | How many frames revealed so far |
| `clip_index` | int? | Task 3: current clip index in queue (0-indexed) |
| `queue_size` | int? | Task 3: total queue length (5) |
| `in_full_review` | bool | Task 3: whether in a 3-frame full-review window |

---

## Action Space

### Task 1 — Binary Moderation (easy)
| Decision | Meaning |
|---|---|
| 0 | `continue_watching` — see next frame (costs −0.5) |
| 1 | `flag_safe` — commit Safe, episode ends |
| 2 | `flag_danger` — commit Danger, episode ends |

### Task 2 — Violation Classification (medium)
| Decision | Meaning |
|---|---|
| 0 | `continue_watching` |
| 1 | `flag_safe` |
| 2 | `flag_bw_flicker` — black-and-white strobe (Harding Rule 1) |
| 3 | `flag_red_flicker` — saturated red flash (Harding Rule 4) |

### Task 3 — Queue Triage (hard)
| Decision | Phase | Meaning |
|---|---|---|
| 0 | Preview | `skip_safe` — mark clip safe, advance queue |
| 1 | Preview | `escalate_danger` — mark clip danger, advance queue |
| 2 | Preview/Review | `request_full_review` / `see_next_frame` |
| 3 | Full-review | `commit_safe` |
| 4 | Full-review | `commit_danger` |

---

## Tasks

### Task 1 — Binary Moderation *(easy)*
Observe up to 4 frames one at a time and decide whether the clip is `Safe` or `Danger`.

**Grader** (`epilepsure.graders.grade_task1`):
- Correct decision ≤ 2 frames: **1.0**
- Correct decision on frame 3-4: **0.8**
- Wrong decision: **0.0**

---

### Task 2 — Violation Classification *(medium)*
Same frame-by-frame flow, but the agent must commit to the specific violation type.

**Grader** (`epilepsure.graders.grade_task2`):
- Correct type, committed ≤ 2 frames: **1.0**
- Correct type, frame 3-4: **0.9**
- Correct danger direction, wrong type: **0.5**
- Wrong danger/safe direction: **0.0**

---

### Task 3 — Queue Triage *(hard)*
Review a queue of 5 clips. Each shows a 1-frame preview. The agent can skip, escalate, or spend 3 more frames on a full review. Graded on F1 across all 5 clips.

**Grader** (`epilepsure.graders.grade_task3`):
- Base score = F1 (precision × recall on danger detection)
- Each missed `RED_FLICKER`: ×0.8 penalty
- Steps used ≤ 10 (decisive): +0.05 efficiency bonus
- Range: **0.0 – 1.0**

---

## Reward Function

| Event | Reward |
|---|---|
| Correct danger flag | +10 |
| Correct safe flag | +1 |
| False alarm | −5 |
| Missed BW flicker | −20 |
| Missed red flicker | −50 (highest risk) |
| Each `continue_watching` step | −0.5 |

Reward is non-zero throughout each episode. The step cost gives the agent a signal to learn *when* to commit, not just *what* to say.

---

## Harding Test Rules Implemented

1. **7-Transitions Rule** — no more than 3.5 flashes/second
2. **25% Area Rule** — flashing region > 25% of screen
3. **Luminance Threshold** — Δluminance > 20 cd/m²
4. **Saturated Red Flashes** — highest risk; stricter limits
5. **Stationary Spatial Patterns** — high-contrast stripes/checks
6. **Moving Patterns** — scrolling or rotating high-contrast elements
7. **Rapid Scene Cuts** — > 3 cuts/second between high-contrast scenes

---

## Setup and Usage

### Prerequisites
- Docker **or** Python 3.10+
- For the baseline: `OPENAI_API_KEY` environment variable

### Docker (recommended)
```bash
docker build -t epilepsure-v2 .
docker run -p 5000:5000 epilepsure-v2

# Change active task:
docker run -p 5000:5000 -e EPILEPSY_TASK=task3 epilepsure-v2
```

### Local development
```bash
git clone https://github.com/your-username/Epilepsure-RL.git
cd Epilepsure-RL
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python server.py
```

### Quick smoke test (no server needed)
```bash
python scripts/test_client.py --task task1
python scripts/test_client.py --task task3 --episodes 3
```

---

## Training

```bash
# Train PPO on each task
python scripts/train.py --task task1 --steps 50000
python scripts/train.py --task task2 --steps 50000
python scripts/train.py --task task3 --steps 100000
```

Checkpoints are saved to `models/`. Monitor with TensorBoard:
```bash
tensorboard --logdir tensorboard_logs
```

## Evaluation

```bash
python scripts/evaluate.py --task task1
python scripts/evaluate.py --task task2 --model models/epilepsy_task2_final
python scripts/evaluate.py --task task3 --episodes 50
```

## Baseline Inference (OpenAI GPT-4o-mini)

```bash
export OPENAI_API_KEY=sk-...
python scripts/baseline_openai.py
```

Results are saved to `baseline_results.json`.

---

## Baseline Scores (GPT-4o-mini, zero-shot)

| Task | Avg Grade | Avg Reward | Notes |
|---|---|---|---|
| Task 1 — Binary Moderation | 0.74 | +3.1 | Strong BW detection; red flicker harder |
| Task 2 — Violation Classification | 0.58 | +1.8 | Often correct direction, wrong subtype |
| Task 3 — Queue Triage | 0.51 | −4.2 | Budget management challenging without training |

*Scores reproduced with `seed_base=42`, `N=50/50/20` episodes.*

---

## Project Structure

```
epilepsure/
  __init__.py       — public API
  models.py         — EpilepsyObservation, EpilepsyAction, EpilepsyReward
  generators.py     — synthetic frame generation (safe / BW / red flicker)
  graders.py        — grade_task1, grade_task2, grade_task3 (0.0–1.0)
  env.py            — EpilepsyEnv (multi-step, 3 tasks)
scripts/
  baseline_openai.py — OpenAI API baseline runner
  train.py           — PPO training (Stable Baselines 3)
  evaluate.py        — model evaluation with grader scoring
  test_client.py     — quick local smoke test
server.py           — FastAPI HTTP server
openenv.yaml        — environment metadata
Dockerfile          — container definition (HF Space compatible)
requirements.txt
```
