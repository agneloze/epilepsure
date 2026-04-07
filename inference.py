"""
inference.py — Epilepsure v2 OpenEnv Inference Script
======================================================

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL        LLM API endpoint  (default: OpenAI)
    MODEL_NAME          Model identifier  (default: gpt-4o-mini)
    HF_TOKEN            HuggingFace / API key  (also checked as API_KEY)

STDOUT FORMAT (machine-parseable, one line each):
    [START] task=<task_name> env=epilepsure model=<model_name>
    [STEP]  step=<n> action=<action_int> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Runs all three tasks in sequence.  Total runtime < 20 min on vcpu=2 / 8 GB.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import textwrap
import time
from typing import List, Optional

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from server.epilepsure_environment import EpilepsyEnv
from models import EpilepsyAction, TASK_ACTION_SPACES

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai package not installed. Run: pip install openai", flush=True)
    sys.exit(1)

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
BENCHMARK    = "epilepsure"

# Episodes per task — kept small so total runtime stays well under 20 min
N_EPISODES = {
    "task1": 10,
    "task2": 10,
    "task3": 5,
}
SEED_BASE = 42   # reproducible

if not API_KEY:
    print("[ERROR] No API key found. Set HF_TOKEN, API_KEY, or OPENAI_API_KEY.", flush=True)
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── Structured loggers ────────────────────────────────────────────────────────

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: int, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Frame encoding ────────────────────────────────────────────────────────────

def frame_to_b64(frame_flat: list, scale: int = 4) -> str:
    """Encode flat 12288-float frame as base64 PNG (upscaled for model visibility)."""
    arr = np.array(frame_flat, dtype=np.uint8).reshape(64, 64, 3)
    if _HAS_PIL:
        img = Image.fromarray(arr, "RGB")
        img = img.resize((64 * scale, 64 * scale), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    # Fallback: raw PNG via struct (no PIL dependency)
    import zlib, struct
    img_data = arr.repeat(scale, axis=0).repeat(scale, axis=1)
    h, w = img_data.shape[:2]
    def make_png(pixels):
        def pack_row(row):
            return b'\x00' + row.tobytes()
        raw = b''.join(pack_row(pixels[y]) for y in range(h))
        compressed = zlib.compress(raw)
        def chunk(name, data):
            c = name + data
            return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)
        sig = b'\x89PNG\r\n\x1a\n'
        ihdr = chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
        idat = chunk(b'IDAT', compressed)
        iend = chunk(b'IEND', b'')
        return sig + ihdr + idat + iend
    return base64.b64encode(make_png(img_data)).decode()


# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_TASK1 = textwrap.dedent("""
    You are a video content safety reviewer detecting photosensitive epilepsy triggers.

    Harding Test — key rules:
    - Flashing region >25% of frame AND luminance swing >20 cd/m² = DANGER
    - Alternating bright/dark blocks (BW strobe) = DANGER
    - Alternating saturated red flashes = DANGER (highest risk)
    - Smooth, stable, low-contrast content = SAFE

    You see one video frame per turn.
    Reply ONLY with valid JSON on one line: {"decision": N}
      N=0  continue watching (see next frame, small cost)
      N=1  flag SAFE (commit, episode ends)
      N=2  flag DANGER (commit, episode ends)

    Prefer early decisions when confident. If content is clearly stable and smooth,
    flag SAFE immediately. If you see alternating bright/dark or red flashing, flag DANGER.
""").strip()

SYSTEM_TASK2 = textwrap.dedent("""
    You are a video content safety reviewer classifying Harding Test violations.

    Reply ONLY with valid JSON on one line: {"decision": N}
      N=0  continue watching
      N=1  SAFE — no violation
      N=2  BW_FLICKER — black-and-white strobe (alternating bright/dark blocks)
      N=3  RED_FLICKER — saturated red flash (alternating deep red/dark blocks)

    Red flickering: large red region (R high, G+B near zero) alternating with near-black.
    Prefer early, specific decisions when the pattern is clear.
""").strip()

SYSTEM_TASK3 = textwrap.dedent("""
    You are a content moderation queue manager reviewing clips for epilepsy triggers.

    Reply ONLY with valid JSON on one line: {"decision": N}

    Preview phase (one frame shown per clip):
      N=0  skip_safe          — mark this clip safe, advance to next
      N=1  escalate_danger    — mark this clip danger, advance to next
      N=2  request_full_review — see up to 3 more frames before deciding

    Full-review phase (after requesting more frames):
      N=2  see_next_frame (if available)
      N=3  commit_safe
      N=4  commit_danger

    Be efficient: only request full review when the preview is ambiguous.
    Red flickering is easy to miss — request review if you see any red tones.
""").strip()


# ── Model call ────────────────────────────────────────────────────────────────

def call_model(system: str, user_text: str, frame_flat: list,
               retries: int = 3) -> str:
    b64 = frame_to_b64(frame_flat)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=32,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64}",
                                    "detail": "low",
                                },
                            },
                            {"type": "text", "text": user_text},
                        ],
                    },
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return '{"decision": 1}'   # safe fallback
    return '{"decision": 1}'


def parse_decision(raw: str, valid: list[int]) -> int:
    """Parse integer 'decision' from JSON; fallback to first valid action."""
    try:
        return int(json.loads(raw).get("decision", valid[0]))
    except Exception:
        for token in raw.split():
            try:
                d = int(token.strip('{}"\','))
                if d in valid:
                    return d
            except ValueError:
                pass
    return valid[0]


# ── Task runners ──────────────────────────────────────────────────────────────

def run_task1_episode(env: EpilepsyEnv, seed: int) -> tuple[float, float, list[float]]:
    """Returns (grade, episode_reward, per_step_rewards)."""
    obs = env.reset(seed=seed, task_id="task1")
    rewards: list[float] = []
    step = 0

    while not obs.done:
        step += 1
        user_msg = (f"Frame {obs.frames_seen}/4 shown so far. "
                    f"Step {step}. Decide or continue.")
        raw = call_model(SYSTEM_TASK1, user_msg, obs.frame)
        decision = parse_decision(raw, [0, 1, 2])
        obs = env.step(EpilepsyAction(decision=decision))
        rewards.append(obs.reward)
        log_step(step, decision, obs.reward, obs.done, None)

    grade = float(obs.metadata.get("grade", 0.0))
    ep_reward = float(obs.metadata.get("episode_reward", sum(rewards)))
    return grade, ep_reward, rewards


def run_task2_episode(env: EpilepsyEnv, seed: int) -> tuple[float, float, list[float]]:
    obs = env.reset(seed=seed, task_id="task2")
    rewards: list[float] = []
    step = 0

    while not obs.done:
        step += 1
        user_msg = (f"Frame {obs.frames_seen}/4. Step {step}. "
                    "Classify violation type or continue.")
        raw = call_model(SYSTEM_TASK2, user_msg, obs.frame)
        decision = parse_decision(raw, [0, 1, 2, 3])
        obs = env.step(EpilepsyAction(decision=decision))
        rewards.append(obs.reward)
        log_step(step, decision, obs.reward, obs.done, None)

    grade = float(obs.metadata.get("grade", 0.0))
    ep_reward = float(obs.metadata.get("episode_reward", sum(rewards)))
    return grade, ep_reward, rewards


def run_task3_episode(env: EpilepsyEnv, seed: int) -> tuple[float, float, list[float]]:
    obs = env.reset(seed=seed, task_id="task3")
    rewards: list[float] = []
    step = 0

    while not obs.done:
        step += 1
        phase = obs.metadata.get("phase", "preview")
        clip_idx = (obs.clip_index or 0) + 1

        if phase == "preview":
            user_msg = (f"Clip {clip_idx}/5 — preview frame. "
                        "0=skip_safe, 1=escalate_danger, 2=full_review")
            valid = [0, 1, 2]
        else:
            clip_frame = obs.metadata.get("clip_frame", 1)
            at_last = (clip_frame >= 3)
            user_msg = (f"Clip {clip_idx}/5 — review frame {clip_frame}/3. "
                        + ("Must commit: 3=safe, 4=danger."
                           if at_last
                           else "2=next_frame, 3=commit_safe, 4=commit_danger."))
            valid = [3, 4] if at_last else [2, 3, 4]

        raw = call_model(SYSTEM_TASK3, user_msg, obs.frame)
        decision = parse_decision(raw, valid)
        obs = env.step(EpilepsyAction(decision=decision))
        rewards.append(obs.reward)
        log_step(step, decision, obs.reward, obs.done, None)

    grade = float(obs.metadata.get("grade", 0.0))
    ep_reward = float(obs.metadata.get("episode_reward", sum(rewards)))
    return grade, ep_reward, rewards


# ── Task orchestrator ─────────────────────────────────────────────────────────

TASK_RUNNERS = {
    "task1": run_task1_episode,
    "task2": run_task2_episode,
    "task3": run_task3_episode,
}

SUCCESS_THRESHOLD = 0.5   # grade ≥ 0.5 across episodes = success


def run_task(task_id: str, n_episodes: int) -> dict:
    """
    Run n_episodes for one task.
    Emits [START] once, then [STEP] per step, then [END] once.
    Returns {"avg_grade": float, "avg_reward": float}.
    """
    log_start(task=task_id, model=MODEL_NAME)

    env = EpilepsyEnv(task_id=task_id)
    runner = TASK_RUNNERS[task_id]

    all_grades: list[float] = []
    all_ep_rewards: list[float] = []
    all_step_rewards: list[float] = []
    total_steps = 0

    for ep_idx in range(n_episodes):
        seed = SEED_BASE + ep_idx
        try:
            grade, ep_reward, step_rewards = runner(env, seed)
        except Exception as exc:
            # Emit a STEP and END even on error
            log_step(total_steps + 1, -1, 0.0, True, str(exc)[:80])
            grade, ep_reward, step_rewards = 0.0, 0.0, [0.0]

        all_grades.append(grade)
        all_ep_rewards.append(ep_reward)
        all_step_rewards.extend(step_rewards)
        total_steps += len(step_rewards)

    avg_grade  = float(np.mean(all_grades))
    avg_reward = float(np.mean(all_ep_rewards))
    success    = avg_grade >= SUCCESS_THRESHOLD

    # score is the avg grade (already in [0, 1])
    log_end(
        success=success,
        steps=total_steps,
        score=round(avg_grade, 3),
        rewards=all_step_rewards,
    )

    return {"avg_grade": avg_grade, "avg_reward": avg_reward}


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    results = {}
    for task_id, n in N_EPISODES.items():
        results[task_id] = run_task(task_id, n)

    # Summary to stderr (not stdout — keeps stdout parseable)
    print("\n# Summary", file=sys.stderr)
    for task_id, r in results.items():
        print(f"#   {task_id}  avg_grade={r['avg_grade']:.4f}  "
              f"avg_reward={r['avg_reward']:+.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
