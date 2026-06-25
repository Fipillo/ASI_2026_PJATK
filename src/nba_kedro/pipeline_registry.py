from kedro.pipeline import Pipeline

from .pipeline import create_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the Kedro pipelines for this project."""
    return {
        "__default__": create_pipeline(),
    }
