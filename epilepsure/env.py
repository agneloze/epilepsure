"""
EpilepsyEnv - An OpenEnv-compatible Reinforcement Learning Environment
for the Epilepsy Trigger Warning Generator (Meta PyTorch OpenEnv Hackathon).

Harding Test Criteria implemented:
  - Flash Frequency : 3 Hz – 30 Hz
  - Luminance       : Contrast > 20 cd/m²
  - Area            : Flashing region > 25 % of screen
  - Red Flash       : Saturated red flashes are high-risk

Observation Space : 4 stacked RGB frames (64 × 64 × 12)
Action Space      : Discrete(2)   0 = Safe | 1 = Danger

Reward Structure:
  +10  correct Danger flag
  +1   correct Safe  flag
  -50  missed RED trigger
  -20  missed BW  trigger
  -5   false alarm (flagged a safe clip as dangerous)
"""

from __future__ import annotations

import os
import glob
import cv2
import random
from typing import Any, Dict, List, Optional, Union

import numpy as np
import openenv
from openenv.core import Environment, Observation, Action, State


# ─────────────────────────────────────────────
#  Scenario labels
# ─────────────────────────────────────────────
SAFE        = 0   # no Harding violation
BW_FLICKER  = 1   # black-and-white high-contrast flicker
RED_FLICKER = 2   # saturated red flicker (highest risk)

DANGER_LABELS = {BW_FLICKER, RED_FLICKER}

# ─────────────────────────────────────────────
#  OpenEnv Types
# ─────────────────────────────────────────────

class EpilepsyObservation(Observation):
    """
    Observation for EpilepsyEnv.
    Contains the 4-frame stack as a flattened or structured array in metadata.
    """
    obs: List[float]  # Flattened (64, 64, 12) array or similar
    source: str = "synthetic"
    # Inherited fields: done, reward, metadata

class EpilepsyAction(Action):
    """
    Action for EpilepsyEnv.
    0 = Safe, 1 = Danger.
    """
    prediction: int  # 0 or 1

# ─────────────────────────────────────────────
#  Frame-generation helpers
# ─────────────────────────────────────────────

def _make_safe_frame(rng: np.random.Generator) -> np.ndarray:
    base = rng.integers(60, 180, dtype=np.uint8)
    noise = rng.integers(-15, 16, size=(64, 64, 3))
    frame = np.clip(base + noise, 0, 255).astype(np.uint8)
    return frame


def _make_bw_flicker_frame(rng: np.random.Generator,
                            frame_idx: int) -> np.ndarray:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    bright = (frame_idx % 2 == 0)
    lum_value  = rng.integers(220, 256) if bright else rng.integers(0, 36)
    bg_value   = rng.integers(80, 120)
    frame[:, :, :] = bg_value
    flash_lum = np.full((40, 40, 3), lum_value, dtype=np.uint8)
    frame[12:52, 12:52, :] = flash_lum
    return frame


def _make_red_flicker_frame(rng: np.random.Generator,
                             frame_idx: int) -> np.ndarray:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    bright = (frame_idx % 2 == 0)
    frame[:, :, :] = rng.integers(20, 50)
    r_val = rng.integers(200, 256) if bright else rng.integers(0, 20)
    g_val = rng.integers(0,  20)
    b_val = rng.integers(0,  20)
    flash_region = np.zeros((40, 40, 3), dtype=np.uint8)
    flash_region[:, :, 0] = r_val
    flash_region[:, :, 1] = g_val
    flash_region[:, :, 2] = b_val
    frame[12:52, 12:52, :] = flash_region
    return frame


# ─────────────────────────────────────────────
#  Main Environment
# ─────────────────────────────────────────────

class EpilepsyEnv(Environment[EpilepsyAction, EpilepsyObservation, State]):
    """
    OpenEnv environment for learning to detect Harding-Test violations.
    """

    # ── reward constants ─────────────────────────
    REWARD_CORRECT_DANGER  = +10
    REWARD_CORRECT_SAFE    = +1
    PENALTY_MISS_RED       = -50
    PENALTY_MISS_BW        = -20
    PENALTY_FALSE_ALARM    = -5

    def __init__(
        self,
        scenario_weights: list[float] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self._scenario_weights = scenario_weights or [1 / 3, 1 / 3, 1 / 3]
        self._master_seed = seed
        self._rng = np.random.default_rng(seed)

        # State fields
        self._episode_id: str | None = None
        self._step_count: int = 0

        # Internal state
        self._current_scenario: int = SAFE
        self._current_obs_array: np.ndarray = np.zeros((64, 64, 12), dtype=np.uint8)
        self._episode_done: bool = True

        self.stats: dict[str, int] = {
            "correct_danger": 0,
            "correct_safe":   0,
            "missed_red":     0,
            "missed_bw":      0,
            "false_alarms":   0,
        }

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count
        )

    def _load_frames_from_file(self, file_path: str) -> list[np.ndarray] | None:
        frames = []
        if file_path.endswith('.npy'):
            try:
                data = np.load(file_path)
                if data.ndim == 4 and data.shape[0] >= 4:
                    start_idx = self._rng.integers(0, data.shape[0] - 3)
                    for i in range(4):
                        frame = data[start_idx + i]
                        if frame.shape != (64, 64, 3):
                            frame = cv2.resize(frame, (64, 64))
                        frames.append(frame.astype(np.uint8))
                else:
                    return None
            except Exception:
                return None
        elif file_path.endswith('.mp4'):
            try:
                cap = cv2.VideoCapture(file_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames < 4:
                    cap.release()
                    return None
                start_frame = self._rng.integers(0, total_frames - 3)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                for _ in range(4):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (64, 64))
                    frames.append(frame.astype(np.uint8))
                cap.release()
                if len(frames) < 4:
                    return None
            except Exception:
                return None
        return frames if len(frames) == 4 else None

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> EpilepsyObservation:
        self._episode_id = episode_id
        self._step_count = 0
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # ── prioritize loading from data/ ───────
        safe_path = os.path.join('data', 'safe')
        danger_path = os.path.join('data', 'danger')
        
        safe_files = []
        danger_files = []
        if os.path.exists(safe_path):
            safe_files = glob.glob(os.path.join(safe_path, "*.npy")) + \
                         glob.glob(os.path.join(safe_path, "*.mp4"))
        if os.path.exists(danger_path):
            danger_files = glob.glob(os.path.join(danger_path, "*.npy")) + \
                           glob.glob(os.path.join(danger_path, "*.mp4"))
        
        frames = None
        all_files = safe_files + danger_files
        
        if all_files:
            # Pick exactly one random file from all found
            file_to_load = self._rng.choice(all_files)
            frames = self._load_frames_from_file(file_to_load)
            if frames:
                # Map file location to scenario
                if file_to_load in safe_files:
                    self._current_scenario = SAFE
                else:
                    self._current_scenario = self._rng.choice([BW_FLICKER, RED_FLICKER])

        # ── fallback to synthetic generator ─────
        if frames is None:
            self._current_scenario = self._rng.choice(
                [SAFE, BW_FLICKER, RED_FLICKER],
                p=self._scenario_weights,
            )
            frames = self._generate_frames(self._current_scenario)

        self._current_obs_array = np.concatenate(frames, axis=2)
        self._episode_done = False
        self._current_source = "file" if all_files and frames else "synthetic"

        return EpilepsyObservation(
            obs=self._current_obs_array.flatten().tolist(),
            source=self._current_source,
            done=False,
            reward=0.0,
            metadata={
                "scenario": self._scenario_name(self._current_scenario),
                "is_danger": self._current_scenario in DANGER_LABELS,
                "shape": [64, 64, 12]
            }
        )

    def step(
        self,
        action: EpilepsyAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> EpilepsyObservation:
        if self._episode_done:
            raise RuntimeError("Called step() on a finished episode.")

        self._step_count += 1
        pred = action.prediction
        reward, stat_key = self._compute_reward(pred, self._current_scenario)
        self.stats[stat_key] += 1
        self._episode_done = True

        return EpilepsyObservation(
            obs=self._current_obs_array.flatten().tolist(),
            source=self._current_source,
            done=True,
            reward=float(reward),
            metadata={
                "action": "Danger" if pred == 1 else "Safe",
                "scenario": self._scenario_name(self._current_scenario),
                "stat": stat_key
            }
        )

    def _generate_frames(self, scenario: int) -> list[np.ndarray]:
        frames = []
        for i in range(4):
            if scenario == SAFE:
                frame = _make_safe_frame(self._rng)
            elif scenario == BW_FLICKER:
                frame = _make_bw_flicker_frame(self._rng, frame_idx=i)
            else:  # RED_FLICKER
                frame = _make_red_flicker_frame(self._rng, frame_idx=i)
            frames.append(frame)
        return frames

    def _compute_reward(self, action: int, scenario: int) -> tuple[float, str]:
        is_danger = scenario in DANGER_LABELS
        agent_flags_danger = action == 1
        if is_danger and agent_flags_danger:
            return self.REWARD_CORRECT_DANGER, "correct_danger"
        if not is_danger and not agent_flags_danger:
            return self.REWARD_CORRECT_SAFE, "correct_safe"
        if not is_danger and agent_flags_danger:
            return self.PENALTY_FALSE_ALARM, "false_alarms"
        if scenario == RED_FLICKER:
            return self.PENALTY_MISS_RED, "missed_red"
        return self.PENALTY_MISS_BW, "missed_bw"

    @staticmethod
    def _scenario_name(scenario: int) -> str:
        return {SAFE: "SAFE", BW_FLICKER: "BW_FLICKER", RED_FLICKER: "RED_FLICKER"}.get(scenario, "UNKNOWN")

