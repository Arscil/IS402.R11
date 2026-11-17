"""
Pure ETL logic used by the Azure Function (no Azure imports here, so it can be
tested locally).

It reproduces prepare_data.py step by step with the same seed, so the cloud ETL
and the local script produce identical train/test sets and client partitions:
1. validate schema, drop duplicate rows
2. stratified 80/20 train/test split (random_state=42)
3. StandardScaler (or MinMaxScaler) on Time and Amount, fit on train only
4. 5 clients IID (StratifiedKFold) and non-IID (Dirichlet, >= 30 frauds per client)
"""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SEED = 42
TEST_SIZE = 0.20
NUM_CLIENTS = 5
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"
COLUMNS_TO_SCALE = ["Time", "Amount"]
MIN_FRAUD_PER_CLIENT = 30


def validate(df: pd.DataFrame) -> None:
    expected = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột: {sorted(missing)}")
    if df[list(expected)].isnull().any().any():
        raise ValueError("Dữ liệu có giá trị thiếu (NaN)")
    labels = set(df[TARGET_COLUMN].unique())
    if not labels <= {0, 1}:
        raise ValueError(f"Nhãn Class không hợp lệ: {labels}")


def make_scaler(scaler_type: str):
    return MinMaxScaler() if scaler_type == "minmax" else StandardScaler()


def partition_iid(X_train, y_train) -> Dict[str, np.ndarray]:
    skf = StratifiedKFold(n_splits=NUM_CLIENTS, shuffle=True, random_state=SEED)
    clients = {}
    for i, (_, idx) in enumerate(skf.split(X_train, y_train)):
        clients[f"X_{i}"] = X_train[idx].astype(np.float32)
        clients[f"y_{i}"] = y_train[idx].astype(np.float32)
    return clients


def partition_noniid(X_train, y_train) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    fraud_idx = np.where(y_train == 1)[0]
    legit_idx = np.where(y_train == 0)[0]
    rng.shuffle(fraud_idx)
    rng.shuffle(legit_idx)

    remaining_fraud = len(fraud_idx) - MIN_FRAUD_PER_CLIENT * NUM_CLIENTS
    fraud_p = rng.dirichlet([0.5] * NUM_CLIENTS)
    fraud_counts = [MIN_FRAUD_PER_CLIENT + int(np.round(p * remaining_fraud)) for p in fraud_p]
    fraud_counts[-1] += len(fraud_idx) - sum(fraud_counts)

    legit_p = rng.dirichlet([20.0] * NUM_CLIENTS)
    legit_counts = [int(np.round(p * len(legit_idx))) for p in legit_p]
    legit_counts[-1] += len(legit_idx) - sum(legit_counts)

    clients, f_ptr, l_ptr = {}, 0, 0
    for i in range(NUM_CLIENTS):
        idx = np.concatenate([fraud_idx[f_ptr:f_ptr + fraud_counts[i]],
                              legit_idx[l_ptr:l_ptr + legit_counts[i]]])
        f_ptr += fraud_counts[i]
        l_ptr += legit_counts[i]
        rng.shuffle(idx)
        clients[f"X_{i}"] = X_train[idx].astype(np.float32)
        clients[f"y_{i}"] = y_train[idx].astype(np.float32)
    return clients


def client_stats(clients: Dict[str, np.ndarray]):
    stats = []
    for i in range(NUM_CLIENTS):
        y = clients[f"y_{i}"]
        fraud = int((y == 1).sum())
        stats.append({"client": i + 1, "total": int(len(y)), "fraud": fraud,
                      "fraud_pct": round(fraud / len(y) * 100, 4)})
    return stats


def run(df: pd.DataFrame, scaler_type: str = "standard") -> dict:
    """Return processed arrays, the fitted scaler and summary statistics."""
    validate(df)
    rows_in = len(df)
    df = df.drop_duplicates().reset_index(drop=True)

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    scaler = make_scaler(scaler_type)
    cols = [FEATURE_COLUMNS.index(c) for c in COLUMNS_TO_SCALE]
    X_train[:, cols] = scaler.fit_transform(X_train[:, cols])
    X_test[:, cols] = scaler.transform(X_test[:, cols])

    clients_iid = partition_iid(X_train, y_train)
    clients_noniid = partition_noniid(X_train, y_train)

    return {
        "train": {"X": X_train.astype(np.float32), "y": y_train.astype(np.float32)},
        "test": {"X": X_test.astype(np.float32), "y": y_test.astype(np.float32)},
        "clients_iid": clients_iid,
        "clients_noniid": clients_noniid,
        "scaler": scaler,
        "stats": {
            "rows_in": rows_in,
            "duplicates_removed": rows_in - len(df),
            "rows_clean": len(df),
            "fraud_clean": int((y == 1).sum()),
            "train_rows": int(len(y_train)), "train_fraud": int((y_train == 1).sum()),
            "test_rows": int(len(y_test)), "test_fraud": int((y_test == 1).sum()),
            "scaler": type(scaler).__name__,
            "clients_iid": client_stats(clients_iid),
            "clients_noniid": client_stats(clients_noniid),
        },
    }
