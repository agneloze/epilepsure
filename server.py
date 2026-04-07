import os
import sys
from server.app import build_app
import uvicorn

if __name__ == "__main__":
    task_id = os.environ.get("EPILEPSY_TASK", "task1")
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0"
    
    print(f"Starting Epilepsure Server via root server.py | task={task_id} | port={port}")
    app = build_app(task_id)
    uvicorn.run(app, host=host, port=port)
