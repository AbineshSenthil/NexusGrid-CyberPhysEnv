"""NexusGrid-CyberPhysEnv server components.
Keep package import side effects minimal so utility modules like
``server.scenarios`` and ``server.grid_engine`` can be imported without
requiring the full OpenEnv runtime.
"""

__all__ = ["CurriculumManager", "NexusgridEnvironment", "TrainingLogger"]


def __getattr__(name: str):
    if name == "CurriculumManager":
        from .curriculum import CurriculumManager

        return CurriculumManager
    if name == "NexusgridEnvironment":
        from .nexusgrid_environment import NexusgridEnvironment

        return NexusgridEnvironment
    if name == "TrainingLogger":
        from .training_logger import TrainingLogger

        return TrainingLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
