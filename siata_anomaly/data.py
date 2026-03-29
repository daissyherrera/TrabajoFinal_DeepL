"""Data loading, preprocessing and windowing for temperature anomaly detection."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ['t']
LABEL_COL = 'temperatura_dudosa'


def load_csv(path, stations=None):
    """Load the CSV and return a clean DataFrame.

    Args:
        path: Path to temperatura_estaciones_2025.csv.
        stations: List of station codes to keep. None = all stations.

    Returns:
        DataFrame sorted by station and timestamp.
    """
    df = pd.read_csv(path, parse_dates=['fecha_hora'])
    if df[LABEL_COL].dtype == object:
        df[LABEL_COL] = df[LABEL_COL].map({'True': True, 'False': False})
    df[LABEL_COL] = df[LABEL_COL].astype(bool)
    df = df.dropna(subset=FEATURE_COLS)
    if stations is not None:
        df = df[df['codigo'].isin(stations)]
    df = df.sort_values(['codigo', 'fecha_hora']).reset_index(drop=True)
    return df


def preprocess(df, scaler=None, feature_cols=FEATURE_COLS):
    """Normalize features with StandardScaler.

    Args:
        df: DataFrame from load_csv.
        scaler: Fitted StandardScaler to reuse. None = fit a new one.
        feature_cols: Columns to normalize.

    Returns:
        (df_scaled, scaler)
    """
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    return df, scaler


def make_windows(df, window_size=30, step=1, feature_cols=FEATURE_COLS, label_col=LABEL_COL):
    """Create sliding windows from time series data.

    Each label corresponds to the last timestep in the window.
    Windows do NOT cross station boundaries.

    Args:
        df: Preprocessed DataFrame sorted by station and time.
        window_size: Number of timesteps per window.
        step: Stride between consecutive windows (use > 1 to downsample).
        feature_cols: Feature columns to include.
        label_col: Boolean column used as anomaly label.

    Returns:
        (X, y): X shape [n_windows, window_size, n_features], y shape [n_windows]
    """
    X_list, y_list = [], []
    for _, station_df in df.groupby('codigo'):
        values = station_df[feature_cols].values.astype(np.float32)
        labels = station_df[label_col].values.astype(np.float32)
        for i in range(0, len(values) - window_size, step):
            X_list.append(values[i:i + window_size])
            y_list.append(labels[i + window_size - 1])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def split_data(df, val_size=0.20, test_size=0.45):
    """Temporal train/val/test split that respects time order within each station.

    Splits each station's records chronologically (no shuffling), then
    concatenates the per-station splits. This prevents data leakage that
    would occur if sliding windows from the future were mixed into training.

    Args:
        df: DataFrame sorted by station and timestamp (output of load_csv).
        val_size: Fraction of each station's records reserved for validation.
        test_size: Fraction of each station's records reserved for testing.

    Returns:
        (df_train, df_val, df_test)
    """
    train_parts, val_parts, test_parts = [], [], []
    for _, station_df in df.groupby('codigo'):
        n = len(station_df)
        train_end = int(n * (1 - val_size - test_size))
        val_end   = int(n * (1 - test_size))
        train_parts.append(station_df.iloc[:train_end])
        val_parts.append(station_df.iloc[train_end:val_end])
        test_parts.append(station_df.iloc[val_end:])
    return (
        pd.concat(train_parts).reset_index(drop=True),
        pd.concat(val_parts).reset_index(drop=True),
        pd.concat(test_parts).reset_index(drop=True),
    )


def compute_class_weight(y_train):
    """Return pos_weight = n_negative / n_positive for weighted loss."""
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    return float(n_neg / n_pos)
