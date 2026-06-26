from src.features import main as run_feature_engineering
from src.train import main as run_training


def main() -> None:
    """Run the full ML pipeline: feature engineering and model training."""
    print("=" * 80)
    print("NBA Game Winner Prediction — Local ML Pipeline")
    print("=" * 80)

    print("\n[1/2] Running feature engineering...")
    run_feature_engineering()

    print("\n[2/2] Running model training and evaluation...")
    run_training()

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
