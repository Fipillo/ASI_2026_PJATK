"""Data loading nodes for NBA pipeline."""
from pathlib import Path

import pandas as pd


def load_games(raw_data_path: str) -> pd.DataFrame:
    """
    Load NBA games dataset.
    
    Args:
        raw_data_path: Path to raw data directory
        
    Returns:
        DataFrame containing games data with dates sorted
    """
    games_path = Path(raw_data_path) / "games.csv"
    
    if not games_path.exists():
        raise FileNotFoundError(f"Games file not found: {games_path}")
    
    games = pd.read_csv(games_path)
    games["GAME_DATE_EST"] = pd.to_datetime(games["GAME_DATE_EST"], errors="coerce")
    games = games.sort_values("GAME_DATE_EST").reset_index(drop=True)
    
    return games


def load_details(raw_data_path: str) -> pd.DataFrame:
    """
    Load NBA game details dataset.
    
    Args:
        raw_data_path: Path to raw data directory
        
    Returns:
        DataFrame containing player-level game statistics
    """
    details_path = Path(raw_data_path) / "games_details.csv"
    
    if not details_path.exists():
        raise FileNotFoundError(f"Details file not found: {details_path}")
    
    details = pd.read_csv(details_path, low_memory=False)
    
    return details
