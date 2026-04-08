"""
EpilepsyEnv FastAPI server (OpenEnv HTTP protocol).

By default serves task1. Set EPILEPSY_TASK env var to change the active task.
"""

from __future__ import annotations

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


# ── Build app at module level so uvicorn can import it ───────────────────────

def _create_minimal_app(env: EpilepsyEnv) -> FastAPI:
    app = FastAPI(title="EpilepsyEnv", version="2.0")

    class ResetRequest(BaseModel):
        seed: int | None = None
        task_id: str | None = None
        episode_id: str | None = None

    @app.get("/")
    def read_root():
        return {"status": "Epilepsure-RL is running", "active_task": env._task_id}

    @app.get("/health")
    def health():
        return {"status": "ok", "env": "EpilepsyEnv-v2"}

    @app.post("/reset")
    def reset(req: ResetRequest = ResetRequest()):
        obs = env.reset(
            seed=req.seed,
            task_id=req.task_id,
            episode_id=req.episode_id,
        )
        return obs.model_dump()

    @app.post("/step")
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


def _build_app(task_id: str) -> FastAPI:
    env = EpilepsyEnv(task_id=task_id)
    if _HAS_OPENENV_SERVER:
        try:
            return create_fastapi_app(lambda: env, EpilepsyAction, EpilepsyObservation)
        except Exception:
            pass
    return _create_minimal_app(env)


# Module-level `app` — required by uvicorn "server.app:app" and openenv validate
app: FastAPI = _build_app(os.environ.get("EPILEPSY_TASK", "task1"))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
