"""
FastAPI application for the NexusGrid-CyberPhysEnv.

Endpoints:
    - WS /ws: persistent sessions (reset/step/state) - use this for multi-step episodes
    - POST /reset, POST /step: stateless HTTP calls (each request gets a fresh environment)
    - GET /state, GET /schema, GET /metadata: OpenEnv introspection
    - GET /health: health check (returns immediately)
    - GET /tasks: task list with rubric weights and horizons
    - GET /web: Gradio dashboard

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi.responses import JSONResponse

ENV_VERSION = "3.0.0"

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import GridAction, GridObservation
    from .nexusgrid_environment import NexusgridEnvironment
    from .rubric import get_task_rubric_specs
    from .scenarios import EVAL_SEED_START, MAX_TICKS, TASK_DIFFICULTIES, TASK_NAMES
except (ImportError, ModuleNotFoundError):
    from models import GridAction, GridObservation
    from server.nexusgrid_environment import NexusgridEnvironment
    from server.rubric import get_task_rubric_specs
    from server.scenarios import EVAL_SEED_START, MAX_TICKS, TASK_DIFFICULTIES, TASK_NAMES


# Create the OpenEnv app
app = create_app(
    NexusgridEnvironment,
    GridAction,
    GridObservation,
    env_name="nexusgrid",
    max_concurrent_envs=16,
)


if not any(getattr(route, "path", None) == "/health" for route in app.routes):
    @app.get("/health")
    async def health_check():
        """
        Health check endpoint.
        Returns HTTP 200 immediately — independent of environment state.
        Required for HF Space HEALTHCHECK and Phase 1 automated validation.
        """
        return JSONResponse(
            content={
                "status": "healthy",
                "environment": "NexusGrid-CyberPhysEnv",
                "version": ENV_VERSION,
            },
            status_code=200,
        )


@app.get("/tasks")
async def tasks():
    """Tasks, horizons and rubric weights. Seeds >= eval_seed_start are held out for evaluation."""
    return JSONResponse(content={
        "version": ENV_VERSION,
        "eval_seed_start": EVAL_SEED_START,
        "tasks": [
            {
                "id": task_id,
                "name": TASK_NAMES[task_id],
                "difficulty": TASK_DIFFICULTIES[task_id],
                "max_ticks": MAX_TICKS[task_id],
                "rubrics": get_task_rubric_specs(task_id),
            }
            for task_id in sorted(TASK_NAMES)
        ],
    })


@app.get("/manifest.json")
async def manifest():
    return JSONResponse(content={
        "name": "NexusGrid — CyberPhys Env",
        "short_name": "NexusGrid",
        "start_url": "/web/",
        "display": "standalone",
        "background_color": "#030711",
        "theme_color": "#00f5d4",
    })


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
