"""
EpilepsyEnv FastAPI server (OpenEnv HTTP protocol).

By default serves task1.  Pass ?task_id=task2 in the reset request body
or set the EPILEPSY_TASK env var to change the active task.

Start:
    python server.py
    python server.py --task task3 --port 5000
"""

from __future__ import annotations

import argparse
import os

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from server.epilepsure_environment import EpilepsyEnv
from models import EpilepsyAction, EpilepsyObservation

try:
    from openenv.core.env_server.http_server import create_fastapi_app
    _HAS_OPENENV_SERVER = True
except ImportError:
    _HAS_OPENENV_SERVER = False


# ── Fallback minimal server (if openenv server module not available) ──────────

def _create_minimal_app(env: EpilepsyEnv) -> FastAPI:
    app = FastAPI(title="EpilepsyEnv", version="2.0")

    class ResetRequest(BaseModel):
        seed: int | None = None
        task_id: str | None = None
        episode_id: str | None = None

    @app.get("/health")
    def health():
        return {"status": "ok", "env": "EpilepsyEnv-v2"}

    @app.post("/reset", response_model=dict)
    def reset(req: ResetRequest = ResetRequest()):
        obs = env.reset(
            seed=req.seed,
            task_id=req.task_id,
            episode_id=req.episode_id,
        )
        return obs.model_dump()

    @app.post("/step", response_model=dict)
    def step(action: EpilepsyAction):
        obs = env.step(action)
        return obs.model_dump()

    @app.get("/state")
    def state():
        s = env.state
        return s.__dict__ if hasattr(s, "__dict__") else {"state": str(s)}

    @app.get("/info")
    def info():
        from models import TASK_ACTION_SPACES, TASK_DESCRIPTIONS
        return {
            "tasks": list(TASK_ACTION_SPACES.keys()),
            "action_spaces": TASK_ACTION_SPACES,
            "task_descriptions": TASK_DESCRIPTIONS,
            "observation_shape": [64, 64, 3],
            "observation_flat_dim": 12288,
        }

    return app


def build_app(task_id: str) -> FastAPI:
    env = EpilepsyEnv(task_id=task_id)
    if _HAS_OPENENV_SERVER:
        try:
            return create_fastapi_app(lambda: env, EpilepsyAction, EpilepsyObservation)
        except Exception:
            pass
    return _create_minimal_app(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=os.environ.get("EPILEPSY_TASK", "task1"),
                        choices=["task1", "task2", "task3"])
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    print(f"Starting EpilepsyEnv server | task={args.task} | port={args.port}")
    app = build_app(args.task)
    uvicorn.run(app, host=args.host, port=args.port)
