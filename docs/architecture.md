# Project architecture

## Overview

This project predicts whether the home team wins an NBA game.

The project contains two main parts:

1. research and experimentation in Jupyter/Colab notebook,
2. local Python pipeline for feature engineering, model training and evaluation.

## Architecture diagram

```text
Raw NBA datasets
games.csv + games_details.csv
        |
        v
data/raw/
        |
        v
src/features.py
        |
        v
Pre-game feature engineering
- rest days
- rolling team statistics
- season win rate
- last game result
- Elo ratings
- home vs visitor differences
        |
        v
data/processed/nba_features_pregame.csv
        |
        v
src/train.py
        |
        v
Random Forest baseline model
        |
        v
reports/baseline_metrics.csv
models/baseline_random_forest.pkl
