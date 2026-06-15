from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    games = pd.read_csv(RAW_DIR / "games.csv")
    details = pd.read_csv(RAW_DIR / "games_details.csv", low_memory=False)

    print("Loaded datasets:")
    print(f"games: {games.shape}")
    print(f"games_details: {details.shape}")

    games.head(100).to_csv(PROCESSED_DIR / "games_sample.csv", index=False)
    details.head(100).to_csv(PROCESSED_DIR / "games_details_sample.csv", index=False)

    with open(PROCESSED_DIR / "games_columns.txt", "w") as f:
        for col in games.columns:
            f.write(col + "\n")

    with open(PROCESSED_DIR / "games_details_columns.txt", "w") as f:
        for col in details.columns:
            f.write(col + "\n")

    print("Saved sample files:")
    print(PROCESSED_DIR / "games_sample.csv")
    print(PROCESSED_DIR / "games_details_sample.csv")
    print(PROCESSED_DIR / "games_columns.txt")
    print(PROCESSED_DIR / "games_details_columns.txt")


if __name__ == "__main__":
    main()
