"""Initialize Kedro project structure for NBA pipeline."""
import os
from pathlib import Path


def initialize_kedro():
    """Create necessary directories and files for Kedro project."""
    project_root = Path(__file__).parent.parent.parent
    
    # Create necessary directories
    directories = [
        project_root / "mlruns",  # MLflow directory
        project_root / "logs",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create .kedro.yml if it doesn't exist
    kedro_yml = project_root / ".kedro.yml"
    if not kedro_yml.exists():
        kedro_yml.write_text("""project_name: nba_prediction
project_version: 1.0.0
""")
        print(f"✅ Created .kedro.yml")
    
    print("\n✅ Kedro project initialized successfully!")
    print(f"📁 Project root: {project_root}")
    print("\n📚 Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run pipeline: python src/nba_kedro/runner.py")
    print("3. View experiments: mlflow ui")


if __name__ == "__main__":
    initialize_kedro()
