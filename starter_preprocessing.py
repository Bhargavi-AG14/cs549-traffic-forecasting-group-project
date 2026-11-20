"""Starter preprocessing pipeline for the traffic-speed forecasting project.

The pipeline keeps the released train/validation split, fits all statistics on
training data only, preserves missingness indicators, and excludes invalid
sensor windows from training/scoring.

Example:
    from starter_preprocessing import prepare_data
    data = prepare_data("upload")
    X_train = data["train"]["X"]
    y_train = data["train"]["Y"]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


HISTORY = 12
HORIZON = 12


@dataclass
class SensorScaler:
    """Per-sensor robust scaling parameters learned from training inputs."""

    center: np.ndarray       # [num_sensors], training median in mph
    scale: np.ndarray        # [num_sensors], training IQR, at least 1 mph

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.center[None, None, :]) / self.scale[None, None, :]

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale[None, None, :] + self.center[None, None, :]


def load_data(data_dir: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Load the supplied files, accepting the uploaded '(2)' filenames."""
    data_dir = Path(data_dir)

    def choose(*names: str) -> Path:
        for name in names:
            path = data_dir / name
            if path.exists():
                return path
        raise FileNotFoundError(f"None of these files was found: {names}")

    files = {
        "train": choose("train.npz"),
        "validation": choose("validation.npz"),
        "test": choose("test_features.npz"),
    }
    output = {name: dict(np.load(path, allow_pickle=False)) for name, path in files.items()}
    return output


def fit_scaler(train_x: np.ndarray, train_valid_mask: np.ndarray) -> SensorScaler:
    """Fit robust per-sensor scaling using valid training history only."""
    x = train_x.astype(np.float64, copy=True)
    mask = np.repeat(train_valid_mask[:, None, :], x.shape[1], axis=1)
    x[~mask] = np.nan

    center = np.nanmedian(x, axis=(0, 1))
    q25 = np.nanpercentile(x, 25, axis=(0, 1))
    q75 = np.nanpercentile(x, 75, axis=(0, 1))
    scale = q75 - q25

    # A constant or nearly constant sensor should not cause division by zero.
    center = np.nan_to_num(center, nan=0.0)
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
    scale = np.maximum(scale, 1.0)
    return SensorScaler(center=center.astype(np.float32), scale=scale.astype(np.float32))


def impute_history(x: np.ndarray, valid_mask: np.ndarray, scaler: SensorScaler) -> tuple[np.ndarray, np.ndarray]:
    """Impute invalid history values and return (imputed_x, missing_indicator).

    Invalid windows are still imputed so a neural network receives finite input,
    but the returned valid_mask must be used to exclude those windows from loss.
    """
    x = x.astype(np.float32, copy=True)
    missing = (~np.isfinite(x)).astype(np.float32)
    invalid_window = (~np.repeat(valid_mask[:, None, :], x.shape[1], axis=1)).astype(np.float32)
    missing = np.maximum(missing, invalid_window)
    fill = scaler.center[None, None, :]
    x[~np.isfinite(x)] = np.broadcast_to(fill, x.shape)[~np.isfinite(x)]
    return x, missing


def time_features(timestamps: np.ndarray) -> np.ndarray:
    """Create cyclical calendar features from forecast-origin timestamps.

    Output columns are sin/cos for time of day and day of week, plus a weekend
    indicator. These are known at prediction time and therefore do not leak Y.
    """
    ts = pd.to_datetime(timestamps)
    minute_of_day = ts.hour.to_numpy() * 60 + ts.minute.to_numpy()
    day_of_week = ts.dayofweek.to_numpy()
    return np.column_stack([
        np.sin(2 * np.pi * minute_of_day / 1440),
        np.cos(2 * np.pi * minute_of_day / 1440),
        np.sin(2 * np.pi * day_of_week / 7),
        np.cos(2 * np.pi * day_of_week / 7),
        (day_of_week >= 5).astype(np.float32),
    ]).astype(np.float32)


def engineer_features(x_scaled: np.ndarray, missing: np.ndarray, timestamps: np.ndarray,
                       metadata: pd.DataFrame | None = None) -> dict[str, np.ndarray]:
    """Build useful baseline features while retaining sequence structure.

    Sequence features have shape [samples, history, sensors]. Static sensor
    coordinates and calendar features are supplied separately for flexible use
    with an MLP, RNN, temporal CNN, or graph model.
    """
    # Recent level, local variability, and short-term trend from the history.
    last_speed = x_scaled[:, -1, :]
    mean_3 = x_scaled[:, -3:, :].mean(axis=1)
    mean_6 = x_scaled[:, -6:, :].mean(axis=1)
    std_6 = x_scaled[:, -6:, :].std(axis=1)
    trend_3 = x_scaled[:, -1, :] - x_scaled[:, -4, :]

    features = {
        "history": x_scaled,
        "missing_history": missing,
        "last_speed": last_speed,
        "mean_last_3": mean_3,
        "mean_last_6": mean_6,
        "std_last_6": std_6,
        "trend_last_15min": trend_3,
        "time": time_features(timestamps),
    }
    if metadata is not None:
        features["sensor_coordinates"] = metadata[["latitude", "longitude"]].to_numpy(dtype=np.float32)
    return features


def prepare_data(data_dir: str | Path) -> dict[str, object]:
    """Run loading, cleaning, normalization, and feature engineering."""
    raw = load_data(data_dir)
    meta_path = Path(data_dir) / "sensor_metadata(2).csv"
    if not meta_path.exists():
        meta_path = Path(data_dir) / "sensor_metadata.csv"
    metadata = pd.read_csv(meta_path)

    scaler = fit_scaler(raw["train"]["X"], raw["train"]["valid_mask"])
    processed = {}
    for split in ("train", "validation", "test"):
        part = raw[split]
        x_imputed, missing = impute_history(part["X"], part["valid_mask"], scaler)
        x_scaled = scaler.transform(x_imputed)
        item = {
            "X": x_scaled.astype(np.float32),
            "valid_mask": part["valid_mask"],
            "timestamps": part["timestamps"],
            "sample_id": part["sample_id"],
            "sensor_ids": part["sensor_ids"],
            "features": engineer_features(x_scaled, missing, part["timestamps"], metadata),
        }
        if "Y" in part:
            # Targets are kept in original mph for evaluation and submission.
            y = part["Y"].astype(np.float32, copy=True)
            y[~np.isfinite(y)] = 0.0
            item["Y_mph"] = y
            # Optional normalized target for models trained with normalized loss.
            item["Y_scaled"] = scaler.transform(y[:, :, :]).astype(np.float32)
        processed[split] = item

    processed["scaler"] = scaler
    processed["metadata"] = metadata
    return processed


def masked_mae(y_true: np.ndarray, y_pred: np.ndarray, valid_mask: np.ndarray) -> float:
    """MAE over valid (sample, sensor) pairs only."""
    expanded = np.repeat(valid_mask[:, None, :], y_true.shape[1], axis=1)
    return float(np.mean(np.abs(y_true[expanded] - y_pred[expanded])))


if __name__ == "__main__":
    data = prepare_data("upload")
    for name in ("train", "validation", "test"):
        item = data[name]
        print(f"{name:10s} X={item['X'].shape}, valid={item['valid_mask'].mean():.3%}")
    print("Per-sensor scaling and feature engineering completed successfully.")
