"""Drift detection for input features."""
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DRIFT_STATS_FILE = Path("logs/drift_stats.json")


class DriftDetector:
    """Detects data drift in NBA prediction features using statistical thresholds."""

    def __init__(self, stats_file: Path = DRIFT_STATS_FILE):
        """Initialize with optional reference statistics file."""
        self.stats_file = stats_file
        self.reference_stats = self._load_stats()

    def _load_stats(self) -> dict[str, Any]:
        """Load reference statistics from file if it exists."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Could not load drift stats: %s", e)
        return {}

    def check_drift(self, features: dict[str, float], threshold: float = 0.2) -> dict[str, Any]:
        """
        Check if features show drift from reference values.

        Args:
            features: Current prediction features
            threshold: Fraction change threshold for drift (0.2 = 20% change)

        Returns:
            Dictionary with drift results for each feature
        """
        if not self.reference_stats:
            return {
                "error": "No reference statistics available",
                "drifted_features": [],
                "num_drifted": 0,
                "total_features": len(features),
                "drift_ratio": 0.0,
                "details": {},
            }

        drifted_features = []
        drift_details = {}

        for feature_name, current_value in features.items():
            if feature_name not in self.reference_stats:
                continue

            ref_stats = self.reference_stats[feature_name]
            ref_mean = ref_stats.get("mean", 0)
            ref_std = ref_stats.get("std", 1)

            if ref_std == 0:
                ref_std = 1

            # Z-score based drift detection
            z_score = abs((current_value - ref_mean) / ref_std) if ref_std > 0 else 0

            # If z-score > 2, it's outside 95% confidence interval
            is_drifted = z_score > 2.0

            drift_details[feature_name] = {
                "current_value": float(current_value),
                "reference_mean": float(ref_mean),
                "reference_std": float(ref_std),
                "z_score": float(z_score),
                "is_drifted": is_drifted,
            }

            if is_drifted:
                drifted_features.append(feature_name)

        return {
            "total_features": len(features),
            "drifted_features": drifted_features,
            "num_drifted": len(drifted_features),
            "drift_ratio": len(drifted_features) / max(len(features), 1),
            "details": drift_details,
        }

    def update_reference_stats(self, features_df) -> None:
        """Update reference statistics from feature data."""
        stats = {}
        for col in features_df.columns:
            stats[col] = {
                "mean": float(features_df[col].mean()),
                "std": float(features_df[col].std()),
                "min": float(features_df[col].min()),
                "max": float(features_df[col].max()),
            }

        self.reference_stats = stats

        # Save to file
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        logger.info("Updated reference statistics with %d features", len(stats))
