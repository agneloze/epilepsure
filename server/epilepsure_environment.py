from __future__ import annotations
import os
import glob
import cv2
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import openenv
    from openenv.core import Environment, State
    _HAS_OPENENV = True
except ImportError:
    _HAS_OPENENV = False
    class Environment:      # type: ignore[no-redef]
        def __init__(self): pass
    class State:            # type: ignore[no-redef]
        def __init__(self, **kw): self.__dict__.update(kw)

from models import (
    EpilepsyObservation,
    EpilepsyAction,
    EpilepsyReward,
    TASK_ACTION_SPACES,
)

# ── Scenario labels ───────────────────────────────────────────────────────────
SAFE        = 0
BW_FLICKER  = 1
RED_FLICKER = 2
DANGER_LABELS = {BW_FLICKER, RED_FLICKER}
DANGER_SCENARIOS = DANGER_LABELS

QUEUE_SIZE = 5   # number of clips per Task-3 episode


def _scenario_name(s: int) -> str:
    return {SAFE: "SAFE", BW_FLICKER: "BW_FLICKER", RED_FLICKER: "RED_FLICKER"}.get(s, "?")


# ── Generators ───────────────────────────────────────────────────────────────

def make_safe_frame(rng: np.random.Generator) -> np.ndarray:
    """Smooth, low-contrast frame with gentle noise — no Harding violation."""
    base = int(rng.integers(60, 180))
    noise = rng.integers(-15, 16, size=(64, 64, 3))
    frame = np.clip(base + noise, 0, 255).astype(np.uint8)
    return frame


def make_bw_flicker_frame(rng: np.random.Generator, frame_idx: int) -> np.ndarray:
    """
    Black-and-white high-contrast strobe.
    Even frames: bright flash region.  Odd frames: dark flash region.
    Δluminance > 20 cd/m² threshold is always exceeded.
    """
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    bright = (frame_idx % 2 == 0)
    lum_value = int(rng.integers(220, 256)) if bright else int(rng.integers(0, 36))
    bg_value  = int(rng.integers(80, 120))
    frame[:, :, :] = bg_value
    frame[12:52, 12:52, :] = lum_value
    return frame


def make_red_flicker_frame(rng: np.random.Generator, frame_idx: int) -> np.ndarray:
    """
    Saturated-red strobe — highest risk per Harding Rule 4.
    Even frames: bright saturated red.  Odd frames: near-black.
    """
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :, :] = int(rng.integers(20, 50))          # dark background
    bright = (frame_idx % 2 == 0)
    r = int(rng.integers(200, 256)) if bright else int(rng.integers(0, 20))
    g = int(rng.integers(0,   20))
    b = int(rng.integers(0,   20))
    flash = np.zeros((40, 40, 3), dtype=np.uint8)
    flash[:, :, 0] = r
    flash[:, :, 1] = g
    flash[:, :, 2] = b
    frame[12:52, 12:52, :] = flash
    return frame


def generate_frames(scenario: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Return 4 frames for the given scenario (0=SAFE, 1=BW, 2=RED)."""
    frames = []
    for i in range(4):
        if scenario == SAFE:
            frames.append(make_safe_frame(rng))
        elif scenario == BW_FLICKER:
            frames.append(make_bw_flicker_frame(rng, i))
        else:
            frames.append(make_red_flicker_frame(rng, i))
    return frames


# ── Graders ───────────────────────────────────────────────────────────────────

def grade_task1(decision: int, scenario: int, steps_used: int) -> Tuple[float, str]:
    """
    Binary moderation grader.
    """
    is_danger   = scenario in DANGER_SCENARIOS
    flags_danger = (decision == 2)

    if is_danger == flags_danger:
        score = 1.0 if steps_used <= 2 else 0.8
        label = "correct_danger" if is_danger else "correct_safe"
    else:
        score = 0.0
        label = "missed_red" if scenario == RED_FLICKER else (
                "missed_bw"  if scenario == BW_FLICKER  else "false_alarm")
    return score, label


def grade_task2(decision: int, scenario: int, steps_used: int) -> Tuple[float, str]:
    """
    Violation-type classification grader.
    """
    if scenario == SAFE:
        if decision == 1:
            return (1.0 if steps_used <= 2 else 0.9), "correct_safe"
        return 0.0, "false_alarm"

    if scenario == BW_FLICKER:
        if decision == 2:
            return (1.0 if steps_used <= 2 else 0.9), "correct_bw"
        if decision == 3:
            return 0.5, "danger_wrong_type"
        return 0.0, "missed_bw"

    # RED_FLICKER
    if decision == 3:
        return (1.0 if steps_used <= 2 else 0.9), "correct_red"
    if decision == 2:
        return 0.5, "danger_wrong_type"
    return 0.0, "missed_red"


def grade_task3(
    decisions: List[int],
    scenarios: List[int],
    steps_used: int,
    max_steps: int = 20,
) -> Tuple[float, str]:
    """
    Queue triage grader.
    """
    if len(decisions) != len(scenarios):
        return 0.0, f"incomplete: got {len(decisions)} decisions for {len(scenarios)} clips"

    tp = fp = fn = tn = 0
    red_misses = 0
    detail_parts = []

    for clip_idx, (dec, scen) in enumerate(zip(decisions, scenarios)):
        is_danger = scen in DANGER_SCENARIOS
        flagged   = (dec == 1)
        if is_danger and flagged:
            tp += 1
            detail_parts.append(f"c{clip_idx}:TP")
        elif not is_danger and flagged:
            fp += 1
            detail_parts.append(f"c{clip_idx}:FP")
        elif is_danger and not flagged:
            fn += 1
            detail_parts.append(f"c{clip_idx}:FN")
            if scen == RED_FLICKER:
                red_misses += 1
        else:
            tn += 1
            detail_parts.append(f"c{clip_idx}:TN")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # Red-miss penalty (each missed red flicker: ×0.8)
    f1 *= (0.8 ** red_misses)

    # Efficiency bonus
    if steps_used <= max_steps // 2:
        f1 = min(1.0, f1 + 0.05)

    score = round(max(0.0, min(1.0, f1)), 4)
    detail = ",".join(detail_parts) + f" | P={precision:.2f} R={recall:.2f} red_miss={red_misses}"
    return score, detail


# ── Environment ───────────────────────────────────────────────────────────────

class EpilepsyEnv(Environment):
    """
    OpenEnv RL environment for photosensitive epilepsy content moderation.
    """

    REWARD_CORRECT_DANGER = +10.0
    REWARD_CORRECT_SAFE   = +1.0
    PENALTY_MISS_BW       = -20.0
    PENALTY_MISS_RED      = -50.0
    PENALTY_FALSE_ALARM   = -5.0
    PENALTY_REVIEW_STEP   = -0.5   # cost of each "continue" step

    def __init__(
        self,
        task_id: str = "task1",
        scenario_weights: list[float] | None = None,
        seed: int | None = None,
    ) -> None:
        if _HAS_OPENENV:
            super().__init__()
        self._task_id = task_id
        self._scenario_weights = scenario_weights or [1 / 3, 1 / 3, 1 / 3]
        self._rng = np.random.default_rng(seed)

        # Shared episode state
        self._episode_id: str | None = None
        self._step_count: int = 0
        self._episode_done: bool = True
        self._episode_reward: float = 0.0

        # Task 1 / 2 state
        self._frames: list[np.ndarray] = []
        self._frame_index: int = 0
        self._current_scenario: int = SAFE
        self._steps_used: int = 0
        self._final_decision: int = -1

        # Task 3 state
        self._queue_clips: list[tuple[list[np.ndarray], int]] = []  # (frames, scenario)
        self._queue_index: int = 0
        self._clip_step: int = 0          # 0=preview, 1-3=full review
        self._in_full_review: bool = False
        self._clip_decisions: list[int] = []
        self._clip_scenarios: list[int] = []
        self._task3_steps: int = 0

        self.stats: dict[str, int] = {
            "correct_danger": 0,
            "correct_safe":   0,
            "missed_red":     0,
            "missed_bw":      0,
            "false_alarms":   0,
            "episodes":       0,
        }

    @property
    def state(self) -> State:
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            episode_done=self._episode_done,
            episode_reward=self._episode_reward,
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> EpilepsyObservation:
        self._episode_id = episode_id
        self._step_count = 0
        self._episode_done = False
        self._episode_reward = 0.0
        self.stats["episodes"] += 1

        if task_id is not None:
            self._task_id = task_id
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._task_id == "task3":
            obs = self._reset_task3()
        else:
            obs = self._reset_task12()
        
        obs.task_id = self._task_id
        return obs

    def step(
        self,
        action: EpilepsyAction,
        **kwargs: Any,
    ) -> EpilepsyObservation:
        if self._episode_done:
            raise RuntimeError("Episode is finished. Call reset() before step().")
        self._step_count += 1

        if self._task_id == "task1":
            return self._step_task1(action)
        if self._task_id == "task2":
            return self._step_task2(action)
        return self._step_task3(action)

    def _reset_task12(self) -> EpilepsyObservation:
        self._current_scenario = self._rng.choice(
            [SAFE, BW_FLICKER, RED_FLICKER],
            p=self._scenario_weights,
        )
        self._frames = self._generate_frames(self._current_scenario)
        self._frame_index = 0
        self._steps_used = 0
        self._final_decision = -1

        return self._make_obs(
            frame=self._frames[0],
            step_num=0,
            max_steps=4,
            frames_seen=1,
            done=False,
            reward=0.0,
            meta={
                "task": self._task_id,
                "action_space": TASK_ACTION_SPACES[self._task_id],
                "note": "Decide: 0=continue, 1=safe, 2=danger" + (
                    ", 3=red_flicker" if self._task_id == "task2" else ""
                ),
            },
        )

    def _step_task1(self, action: EpilepsyAction) -> EpilepsyObservation:
        decision = action.decision
        self._steps_used += 1

        if decision == 0 and self._frame_index < 3:
            self._frame_index += 1
            self._episode_reward += self.PENALTY_REVIEW_STEP
            return self._make_obs(
                frame=self._frames[self._frame_index],
                step_num=self._step_count,
                max_steps=4,
                frames_seen=self._frame_index + 1,
                done=False,
                reward=self.PENALTY_REVIEW_STEP,
                meta={"action": "continue", "frame_index": self._frame_index},
            )

        if decision == 0:
            decision = 1

        reward_val, stat_key = self._reward_task1(decision, self._current_scenario)
        self.stats[stat_key] += 1
        self._episode_reward += reward_val
        self._episode_done = True
        self._final_decision = decision

        score, grade_label = grade_task1(decision, self._current_scenario, self._steps_used)

        return self._make_obs(
            frame=self._frames[self._frame_index],
            step_num=self._step_count,
            max_steps=4,
            frames_seen=self._frame_index + 1,
            done=True,
            reward=reward_val,
            meta={
                "action": TASK_ACTION_SPACES["task1"].get(decision, "?"),
                "scenario": _scenario_name(self._current_scenario),
                "stat": stat_key,
                "grade": score,
                "grade_label": grade_label,
                "episode_reward": self._episode_reward,
                "reward_detail": EpilepsyReward(
                    value=reward_val,
                    breakdown={"step_costs": self.PENALTY_REVIEW_STEP * (self._steps_used - 1),
                                "final": reward_val},
                    reason=stat_key,
                ).model_dump(),
            },
        )

    def _reward_task1(self, decision: int, scenario: int) -> tuple[float, str]:
        is_danger    = scenario in DANGER_LABELS
        flags_danger = (decision == 2)
        if is_danger and flags_danger:
            return self.REWARD_CORRECT_DANGER, "correct_danger"
        if not is_danger and not flags_danger:
            return self.REWARD_CORRECT_SAFE, "correct_safe"
        if not is_danger and flags_danger:
            return self.PENALTY_FALSE_ALARM, "false_alarms"
        if scenario == RED_FLICKER:
            return self.PENALTY_MISS_RED, "missed_red"
        return self.PENALTY_MISS_BW, "missed_bw"

    def _step_task2(self, action: EpilepsyAction) -> EpilepsyObservation:
        decision = action.decision
        self._steps_used += 1

        if decision == 0 and self._frame_index < 3:
            self._frame_index += 1
            self._episode_reward += self.PENALTY_REVIEW_STEP
            return self._make_obs(
                frame=self._frames[self._frame_index],
                step_num=self._step_count,
                max_steps=4,
                frames_seen=self._frame_index + 1,
                done=False,
                reward=self.PENALTY_REVIEW_STEP,
                meta={"action": "continue", "frame_index": self._frame_index},
            )

        if decision == 0:
            decision = 1

        reward_val, stat_key = self._reward_task2(decision, self._current_scenario)
        self.stats[stat_key] += 1
        self._episode_reward += reward_val
        self._episode_done = True
        self._final_decision = decision

        score, grade_label = grade_task2(decision, self._current_scenario, self._steps_used)

        return self._make_obs(
            frame=self._frames[self._frame_index],
            step_num=self._step_count,
            max_steps=4,
            frames_seen=self._frame_index + 1,
            done=True,
            reward=reward_val,
            meta={
                "action": TASK_ACTION_SPACES["task2"].get(decision, "?"),
                "scenario": _scenario_name(self._current_scenario),
                "stat": stat_key,
                "grade": score,
                "grade_label": grade_label,
                "episode_reward": self._episode_reward,
                "reward_detail": EpilepsyReward(
                    value=reward_val,
                    breakdown={"step_costs": self.PENALTY_REVIEW_STEP * (self._steps_used - 1),
                                "final": reward_val},
                    reason=stat_key,
                ).model_dump(),
            },
        )

    def _reward_task2(self, decision: int, scenario: int) -> tuple[float, str]:
        is_danger    = scenario in DANGER_LABELS
        flags_danger = decision in (2, 3)

        if not is_danger and decision == 1:
            return self.REWARD_CORRECT_SAFE, "correct_safe"
        if not is_danger and flags_danger:
            return self.PENALTY_FALSE_ALARM, "false_alarms"
        if is_danger and not flags_danger:
            return (self.PENALTY_MISS_RED if scenario == RED_FLICKER
                    else self.PENALTY_MISS_BW), (
                "missed_red" if scenario == RED_FLICKER else "missed_bw")

        correct_type = (scenario == BW_FLICKER and decision == 2) or \
                       (scenario == RED_FLICKER and decision == 3)
        if correct_type:
            return self.REWARD_CORRECT_DANGER, "correct_danger"
        return self.REWARD_CORRECT_DANGER * 0.4, "correct_danger"

    def _reset_task3(self) -> EpilepsyObservation:
        self._queue_clips = []
        for _ in range(QUEUE_SIZE):
            scen = self._rng.choice([SAFE, BW_FLICKER, RED_FLICKER],
                                    p=self._scenario_weights)
            frames = self._generate_frames(scen)
            self._queue_clips.append((frames, scen))

        self._queue_index = 0
        self._clip_step = 0
        self._in_full_review = False
        self._clip_decisions = []
        self._clip_scenarios = [scen for _, scen in self._queue_clips]
        self._task3_steps = 0

        preview_frame = self._queue_clips[0][0][0]
        return self._make_obs(
            frame=preview_frame,
            step_num=0,
            max_steps=20,
            frames_seen=1,
            done=False,
            reward=0.0,
            clip_index=0,
            queue_size=QUEUE_SIZE,
            in_full_review=False,
            meta={
                "task": "task3",
                "action_space": TASK_ACTION_SPACES["task3"],
                "phase": "preview",
                "note": "0=skip(safe), 1=escalate(danger), 2=full_review",
                "clips_remaining": QUEUE_SIZE,
            },
        )

    def _step_task3(self, action: EpilepsyAction) -> EpilepsyObservation:
        decision = action.decision
        self._task3_steps += 1
        frames, scenario = self._queue_clips[self._queue_index]

        if not self._in_full_review:
            if decision == 0:
                return self._task3_commit_clip(safe=True, scenario=scenario)
            if decision == 1:
                return self._task3_commit_clip(safe=False, scenario=scenario)
            if decision == 2:
                self._in_full_review = True
                self._clip_step = 1
                self._episode_reward += self.PENALTY_REVIEW_STEP
                next_frame = frames[1]
                return self._make_obs(
                    frame=next_frame,
                    step_num=self._step_count,
                    max_steps=20,
                    frames_seen=2,
                    done=False,
                    reward=self.PENALTY_REVIEW_STEP,
                    clip_index=self._queue_index,
                    queue_size=QUEUE_SIZE,
                    in_full_review=True,
                    meta={
                        "phase": "full_review",
                        "clip_frame": self._clip_step,
                        "action_space": {3: "commit_safe", 4: "commit_danger",
                                         **({2: "see_next_frame"} if self._clip_step < 3 else {})},
                    },
                )
            return self._task3_commit_clip(safe=True, scenario=scenario)

        if decision == 3:
            return self._task3_commit_clip(safe=True, scenario=scenario)
        if decision == 4:
            return self._task3_commit_clip(safe=False, scenario=scenario)

        if self._clip_step < 3:
            self._clip_step += 1
            self._episode_reward += self.PENALTY_REVIEW_STEP
            next_frame = frames[self._clip_step]
            at_last = (self._clip_step == 3)
            return self._make_obs(
                frame=next_frame,
                step_num=self._step_count,
                max_steps=20,
                frames_seen=self._clip_step + 1,
                done=False,
                reward=self.PENALTY_REVIEW_STEP,
                clip_index=self._queue_index,
                queue_size=QUEUE_SIZE,
                in_full_review=True,
                meta={
                    "phase": "full_review",
                    "clip_frame": self._clip_step,
                    "action_space": ({3: "commit_safe", 4: "commit_danger"}
                                     if at_last else
                                     {2: "see_next_frame", 3: "commit_safe", 4: "commit_danger"}),
                },
            )

        return self._task3_commit_clip(safe=True, scenario=scenario)

    def _task3_commit_clip(self, safe: bool, scenario: int) -> EpilepsyObservation:
        clip_decision = 0 if safe else 1
        self._clip_decisions.append(clip_decision)

        is_danger   = scenario in DANGER_LABELS
        flags_danger = not safe
        reviewed_extra = self._in_full_review

        if is_danger and flags_danger:
            r = self.REWARD_CORRECT_DANGER * (0.8 if reviewed_extra else 1.0)
            self.stats["correct_danger"] += 1
        elif not is_danger and not flags_danger:
            r = self.REWARD_CORRECT_SAFE * (0.5 if reviewed_extra else 1.0)
            self.stats["correct_safe"] += 1
        elif not is_danger and flags_danger:
            r = self.PENALTY_FALSE_ALARM
            self.stats["false_alarms"] += 1
        elif scenario == RED_FLICKER:
            r = self.PENALTY_MISS_RED
            self.stats["missed_red"] += 1
        else:
            r = self.PENALTY_MISS_BW
            self.stats["missed_bw"] += 1

        self._episode_reward += r
        self._queue_index += 1
        self._in_full_review = False
        self._clip_step = 0

        if self._queue_index >= QUEUE_SIZE:
            self._episode_done = True
            score, detail = grade_task3(
                self._clip_decisions,
                self._clip_scenarios,
                self._task3_steps,
            )
            return self._make_obs(
                frame=self._queue_clips[-1][0][-1],
                step_num=self._step_count,
                max_steps=20,
                frames_seen=1,
                done=True,
                reward=r,
                clip_index=QUEUE_SIZE - 1,
                queue_size=QUEUE_SIZE,
                in_full_review=False,
                meta={
                    "phase": "done",
                    "clip_decisions": self._clip_decisions,
                    "clip_scenarios": [_scenario_name(s) for s in self._clip_scenarios],
                    "grade": score,
                    "grade_detail": detail,
                    "episode_reward": self._episode_reward,
                },
            )

        preview_frame = self._queue_clips[self._queue_index][0][0]
        clips_left = QUEUE_SIZE - self._queue_index
        return self._make_obs(
            frame=preview_frame,
            step_num=self._step_count,
            max_steps=20,
            frames_seen=1,
            done=False,
            reward=r,
            clip_index=self._queue_index,
            queue_size=QUEUE_SIZE,
            in_full_review=False,
            meta={
                "phase": "preview",
                "clips_remaining": clips_left,
                "action_space": {0: "skip_safe", 1: "escalate_danger", 2: "request_full_review"},
                "clip_decisions_so_far": self._clip_decisions,
            },
        )

    def _generate_frames(self, scenario: int) -> list[np.ndarray]:
        frames = []
        for i in range(4):
            if scenario == SAFE:
                frames.append(make_safe_frame(self._rng))
            elif scenario == BW_FLICKER:
                frames.append(make_bw_flicker_frame(self._rng, i))
            else:
                frames.append(make_red_flicker_frame(self._rng, i))
        return frames

    @staticmethod
    def _make_obs(
        frame: np.ndarray,
        step_num: int,
        max_steps: int,
        frames_seen: int,
        done: bool,
        reward: float,
        clip_index: int | None = None,
        queue_size: int | None = None,
        in_full_review: bool = False,
        meta: dict | None = None,
        task_id: str = "",
    ) -> EpilepsyObservation:
        return EpilepsyObservation(
            frame=frame.flatten().tolist(),
            step_num=step_num,
            max_steps=max_steps,
            task_id=task_id,
            frames_seen=frames_seen,
            clip_index=clip_index,
            queue_size=queue_size,
            in_full_review=in_full_review,
            done=done,
            reward=float(reward),
            metadata=meta or {},
        )
