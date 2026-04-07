"""
Typed Pydantic models for EpilepsyEnv per OpenEnv spec.

  EpilepsyObservation  — what the agent sees each step
  EpilepsyAction       — what the agent decides
  EpilepsyReward       — structured reward breakdown
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

try:
    from openenv.core import Observation, Action, State
except ImportError:
    class Observation(BaseModel):       # type: ignore[no-redef]
        done: bool = False
        reward: float = 0.0
        metadata: Dict[str, Any] = {}
    class Action(BaseModel): pass       # type: ignore[no-redef]
    class State(BaseModel): pass        # type: ignore[no-redef]


# ── Action-space documentation ────────────────────────────────────────────────

TASK_ACTION_SPACES: Dict[str, Dict[int, str]] = {
    "task1": {
        0: "continue_watching",   # see next frame (costs −0.5)
        1: "flag_safe",           # commit Safe  → episode ends
        2: "flag_danger",         # commit Danger → episode ends
    },
    "task2": {
        0: "continue_watching",
        1: "flag_safe",
        2: "flag_bw_flicker",     # Danger: black-and-white strobe
        3: "flag_red_flicker",    # Danger: saturated-red flash
    },
    "task3": {
        # ---- Preview phase (1 frame shown per clip) ----
        0: "skip_safe",           # mark clip Safe,   advance queue
        1: "escalate_danger",     # mark clip Danger, advance queue
        2: "request_full_review", # spend 3 more frames before deciding
        # ---- Full-review phase (frames 1-3 of clip) ----
        3: "commit_safe",         # end review, mark Safe
        4: "commit_danger",       # end review, mark Danger
    },
}

TASK_DESCRIPTIONS: Dict[str, str] = {
    "task1": (
        "Binary Moderation (easy): observe up to 4 frames one at a time, "
        "decide Safe or Danger. Early correct decisions score higher."
    ),
    "task2": (
        "Violation Classification (medium): same frame-by-frame flow, "
        "but commit to the exact violation type (BW Flicker vs Red Flicker) "
        "for full credit. Correct danger/safe direction earns partial credit."
    ),
    "task3": (
        "Queue Triage (hard): 5-clip review queue. For each clip you see one "
        "preview frame and must decide: skip (safe), escalate (danger), or "
        "spend 3 extra frames for a full review. Budget is tight — "
        "graded on F1 across all 5 clips."
    ),
}


# ── Typed models ──────────────────────────────────────────────────────────────

class EpilepsyReward(BaseModel):
    """Structured reward — returned in observation.metadata['reward_detail']."""
    value: float
    breakdown: Dict[str, float] = {}
    reason: str = ""


class EpilepsyObservation(Observation):
    """
    One-frame observation.  frame = 64 × 64 × 3 = 12 288 uint8 values (flattened).

    Extra fields for Task 3 queue context:
      clip_index      — which clip in the queue (0-indexed)
      queue_size      — total clips in queue
      in_full_review  — True while in the 3-frame full-review window
    """
    frame: List[float]
    step_num: int = 0
    max_steps: int = 4
    task_id: str = "task1"
    frames_seen: int = 0
    clip_index: Optional[int] = None
    queue_size: Optional[int] = None
    in_full_review: bool = False


class EpilepsyAction(Action):
    """
    Unified action.  See TASK_ACTION_SPACES for per-task semantics.

    decision    — integer from the task's action space
    confidence  — optional agent confidence in [0, 1]
    """
    decision: int
    confidence: float = 1.0
