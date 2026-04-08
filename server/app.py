"""
FastAPI application for the NexusGrid-CyberPhysEnv.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /health: Health check (returns immediately)
    - GET /schema: Get action/observation schemas
    - GET /web: Gradio visual dashboard
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import GridAction, GridObservation
    from .nexusgrid_environment import NexusgridEnvironment
except (ImportError, ModuleNotFoundError):
    from models import GridAction, GridObservation
    from server.nexusgrid_environment import NexusgridEnvironment


# Create the OpenEnv app
app = create_app(
    NexusgridEnvironment,
    GridAction,
    GridObservation,
    env_name="nexusgrid",
    max_concurrent_envs=1,
)


# Add /health endpoint — returns immediately, no computation
@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns HTTP 200 immediately — independent of environment state.
    Required for HF Space HEALTHCHECK and Phase 1 automated validation.
    """
    return JSONResponse(
        content={
            "status": "ok",
            "environment": "NexusGrid-CyberPhysEnv",
            "version": "1.0.0",
        },
        status_code=200,
    )


# ---------------------------------------------------------------------------
# Mount Gradio dashboard at /web
# ---------------------------------------------------------------------------
try:
    import gradio as gr
    from server.dashboard import create_dashboard

    dashboard = create_dashboard()
    app = gr.mount_gradio_app(app, dashboard, path="/web")
    print("[NexusGrid] Gradio dashboard mounted at /web")
except ImportError:
    print("[NexusGrid] Gradio not installed — dashboard disabled. Install with: pip install gradio plotly")
except Exception as e:
    print(f"[NexusGrid] Dashboard mount failed: {e} — API-only mode")


def main():
    """
    Entry point for direct execution.

    Usage:
        uv run --project . server
        python -m server.app
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="NexusGrid-CyberPhysEnv Server")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=30,
    )


if __name__ == "__main__":
    main()
