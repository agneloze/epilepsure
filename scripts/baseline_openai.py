"""
Baseline inference script — uses the OpenAI API (GPT-4o-mini vision) to run
a zero-shot agent against all three EpilepsyEnv tasks.

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/baseline_openai.py

Output:
    Baseline scores for Task 1 / Task 2 / Task 3 printed to stdout.
    Results also saved to baseline_results.json.

Reproducibility:
    Fixed seeds per episode ensure identical frame sequences on each run.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.epilepsure_environment import EpilepsyEnv, grade_task1, grade_task2, grade_task3
from models import EpilepsyAction

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed.  Run: pip install openai")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not installed.  Run: pip install Pillow")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY  = os.environ.get("OPENAI_API_KEY", "")
MODEL    = "gpt-4o-mini"
N_TASK1  = 50    # episodes for task 1
N_TASK2  = 50    # episodes for task 2
N_TASK3  = 20    # episodes for task 3 (longer episodes)
SEED_BASE = 42   # deterministic seeds: SEED_BASE + episode_index

if not API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable not set.")
    sys.exit(1)

client = OpenAI(api_key=API_KEY)

# ── Helpers ───────────────────────────────────────────────────────────────────

def frame_to_b64(frame_flat: list[float], scale: int = 4) -> str:
    """Convert flat 12288-float frame to a base64-encoded PNG (upscaled for visibility)."""
    arr = np.array(frame_flat, dtype=np.uint8).reshape(64, 64, 3)
    img = Image.fromarray(arr, "RGB")
    img = img.resize((64 * scale, 64 * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def call_model(system: str, user_text: str, image_b64: str, retries: int = 3) -> str:
    """Call GPT-4o-mini with a vision message; return raw text."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=64,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                    "detail": "low",
                                },
                            },
                            {"type": "text", "text": user_text},
                        ],
                    },
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [API error after {retries} attempts]: {e}")
                return '{"decision": 1}'   # fallback: flag safe


def parse_decision(raw: str, valid: list[int]) -> int:
    """Extract integer 'decision' from JSON response; fallback to first valid action."""
    try:
        data = json.loads(raw)
        d = int(data.get("decision", valid[0]))
        return d if d in valid else valid[0]
    except Exception:
        for token in raw.split():
            try:
                d = int(token.strip("{}\"',"))
                if d in valid:
                    return d
            except ValueError:
                pass
    return valid[0]


# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_TASK1 = """You are a content safety reviewer detecting photosensitive epilepsy triggers.

Harding Test rules (simplified):
- Flashing region >25% of frame AND luminance swing >20cd/m² = DANGER
- Alternating bright/dark blocks = DANGER (BW flicker)
- Alternating saturated red = DANGER (red flicker, highest risk)
- Smooth, stable, low-contrast content = SAFE

You will see one video frame at a time.
Respond ONLY with valid JSON: {"decision": N}
  N=0  continue_watching  (see next frame; small cost)
  N=1  flag_safe          (commit — episode ends)
  N=2  flag_danger        (commit — episode ends)

Prefer early decisions when you are confident. Default to N=1 if unsure."""

SYSTEM_TASK2 = """You are a content safety reviewer classifying Harding Test violations.

Respond ONLY with valid JSON: {"decision": N}
  N=0  continue_watching
  N=1  safe
  N=2  bw_flicker   — black-and-white strobe (alternating bright/dark blocks)
  N=3  red_flicker  — saturated red flash (alternating red/dark blocks)

Red flickering is harder to detect (deep red on dark background).
Prefer early decisions when confident."""

SYSTEM_TASK3 = """You are a content moderation queue manager reviewing video clips for epilepsy triggers.

For each preview frame you see, respond ONLY with valid JSON: {"decision": N}

Preview phase (one frame shown):
  N=0  skip_safe          — mark clip as safe, move on
  N=1  escalate_danger    — mark clip as danger, move on
  N=2  request_full_review — see 3 more frames before deciding

Full-review phase (after requesting more frames):
  N=3  commit_safe
  N=4  commit_danger
  N=2  see_next_frame (if available)

Red flickering (saturated red alternating with near-black) is harder to spot — request full review if suspicious."""


# ── Episode runners ───────────────────────────────────────────────────────────

def run_task1_episode(env: EpilepsyEnv, seed: int) -> dict:
    obs = env.reset(seed=seed, task_id="task1")
    steps_used = 0
    while not obs.done:
        b64 = frame_to_b64(obs.frame)
        prompt = f"Frame {obs.frames_seen}/4. Steps used: {steps_used}. Decide now or continue."
        raw = call_model(SYSTEM_TASK1, prompt, b64)
        decision = parse_decision(raw, [0, 1, 2])
        obs = env.step(EpilepsyAction(decision=decision))
        steps_used += 1
    meta = obs.metadata
    return {
        "grade": meta.get("grade", 0.0),
        "episode_reward": meta.get("episode_reward", 0.0),
        "grade_label": meta.get("grade_label", ""),
        "scenario": meta.get("scenario", ""),
    }


def run_task2_episode(env: EpilepsyEnv, seed: int) -> dict:
    obs = env.reset(seed=seed, task_id="task2")
    steps_used = 0
    while not obs.done:
        b64 = frame_to_b64(obs.frame)
        prompt = f"Frame {obs.frames_seen}/4. Classify the violation type or continue."
        raw = call_model(SYSTEM_TASK2, prompt, b64)
        decision = parse_decision(raw, [0, 1, 2, 3])
        obs = env.step(EpilepsyAction(decision=decision))
        steps_used += 1
    meta = obs.metadata
    return {
        "grade": meta.get("grade", 0.0),
        "episode_reward": meta.get("episode_reward", 0.0),
        "grade_label": meta.get("grade_label", ""),
        "scenario": meta.get("scenario", ""),
    }


def run_task3_episode(env: EpilepsyEnv, seed: int) -> dict:
    obs = env.reset(seed=seed, task_id="task3")
    while not obs.done:
        b64 = frame_to_b64(obs.frame)
        phase = obs.metadata.get("phase", "preview")
        clip_idx = obs.clip_index or 0

        if phase == "preview":
            prompt = (f"Clip {clip_idx + 1}/5 — preview frame. "
                      "0=skip(safe), 1=escalate(danger), 2=request full review.")
            valid = [0, 1, 2]
        else:
            frame_num = obs.metadata.get("clip_frame", 1)
            at_last = (frame_num >= 3)
            prompt = (f"Clip {clip_idx + 1}/5 — review frame {frame_num}/3. "
                      + ("Must commit: 3=safe, 4=danger." if at_last
                         else "3=commit safe, 4=commit danger, 2=see next frame."))
            valid = [3, 4] if at_last else [2, 3, 4]

        raw = call_model(SYSTEM_TASK3, prompt, b64)
        decision = parse_decision(raw, valid)
        obs = env.step(EpilepsyAction(decision=decision))

    meta = obs.metadata
    return {
        "grade": meta.get("grade", 0.0),
        "episode_reward": meta.get("episode_reward", 0.0),
        "grade_detail": meta.get("grade_detail", ""),
        "clip_decisions": meta.get("clip_decisions", []),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_task(label: str, runner, env: EpilepsyEnv, n: int, seed_base: int) -> dict:
    grades = []
    rewards = []
    print(f"\n{'─'*60}")
    print(f"  {label}  ({n} episodes, model={MODEL})")
    print(f"{'─'*60}")
    for i in range(n):
        result = runner(env, seed=seed_base + i)
        grades.append(result["grade"])
        rewards.append(result["episode_reward"])
        print(f"  ep {i+1:3d}/{n}  grade={result['grade']:.3f}  "
              f"reward={result['episode_reward']:+.1f}  "
              f"{result.get('grade_label', result.get('grade_detail', ''))[:40]}")
    avg_grade  = float(np.mean(grades))
    avg_reward = float(np.mean(rewards))
    print(f"\n  ✓ avg grade = {avg_grade:.4f}  |  avg reward = {avg_reward:+.2f}")
    return {"avg_grade": avg_grade, "avg_reward": avg_reward,
            "grades": grades, "rewards": rewards}


def main() -> None:
    env = EpilepsyEnv()

    results: dict[str, Any] = {
        "model": MODEL,
        "seed_base": SEED_BASE,
        "tasks": {},
    }

    results["tasks"]["task1"] = run_task(
        "Task 1 — Binary Moderation (easy)", run_task1_episode, env, N_TASK1, SEED_BASE)
    results["tasks"]["task2"] = run_task(
        "Task 2 — Violation Classification (medium)", run_task2_episode, env, N_TASK2, SEED_BASE)
    results["tasks"]["task3"] = run_task(
        "Task 3 — Queue Triage (hard)", run_task3_episode, env, N_TASK3, SEED_BASE)

    print(f"\n{'═'*60}")
    print("  FINAL BASELINE SCORES")
    print(f"{'═'*60}")
    for task_id, res in results["tasks"].items():
        print(f"  {task_id}  grade={res['avg_grade']:.4f}  reward={res['avg_reward']:+.2f}")
    print(f"{'═'*60}\n")

    out_path = os.path.join(os.path.dirname(__file__), "..", "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
