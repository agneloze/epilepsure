import openenv
import uvicorn
from epilepsure.env import EpilepsyEnv, EpilepsyAction, EpilepsyObservation
from openenv.core.env_server.http_server import create_fastapi_app

def serve(env_instance, port=5000):
    """
    Simplified serve function to start a local OpenEnv server.
    """
    # create_fastapi_app(env_factory, action_cls, observation_cls, ...)
    app = create_fastapi_app(
        lambda: env_instance,
        EpilepsyAction,
        EpilepsyObservation
    )
    print(f"Starting EpilepsySafety-v0 server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    # Mocking openenv.serve for the purpose of this script if it's expected at top level
    if not hasattr(openenv, 'serve'):
        openenv.serve = serve
    
    env = EpilepsyEnv()
    openenv.serve(env, port=5000)
