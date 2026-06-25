"""Run Kedro pipeline for NBA game prediction."""
from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


def run_pipeline():
    """Execute NBA prediction pipeline using Kedro."""
    project_path = Path(__file__).parent.parent.parent
    
    bootstrap_project(project_path)
    
    with KedroSession.create(project_path=project_path, env="base") as session:
        pipeline = session.run()
    
    print("\n✅ Pipeline completed successfully!")
    print(f"Results saved to: reports/baseline_metrics.yml")
    print(f"Model saved to: models/baseline_random_forest.pkl")


if __name__ == "__main__":
    run_pipeline()
