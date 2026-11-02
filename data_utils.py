"""
Data utilities for Federated Learning Credit Card Fraud Detection.
Provides functions to load processed data, build TFF-compatible tf.data.Dataset,
and compute class weights for imbalanced learning.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf

import config


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed centralized train and test sets.

    Returns:
        X_train: Features for training (N_train, 30) - float32
        y_train: Labels for training (N_train,) - int32 / float32
        X_test: Features for testing (N_test, 30) - float32
        y_test: Labels for testing (N_test,) - int32 / float32
    """
    if not config.TRAIN_DATA_FILE.exists() or not config.TEST_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Preprocessed train/test data not found at {config.PROCESSED_DATA_DIR}. "
            "Please run prepare_data.py or notebooks/01_data.ipynb first."
        )

    train_data = np.load(config.TRAIN_DATA_FILE)
    test_data = np.load(config.TEST_DATA_FILE)

    X_train = train_data["X"].astype(np.float32)
    y_train = train_data["y"].astype(np.float32)
    X_test = test_data["X"].astype(np.float32)
    y_test = test_data["y"].astype(np.float32)

    return X_train, y_train, X_test, y_test


def load_client_data(scenario: str = "iid") -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load partitioned client datasets from saved .npz files.

    Args:
        scenario: 'iid' or 'noniid'

    Returns:
        List of (X_i, y_i) tuples for each client (5 clients total)
    """
    scenario_clean = scenario.lower().replace("-", "").replace("_", "")
    if scenario_clean == "iid":
        filepath = config.CLIENTS_IID_FILE
    elif scenario_clean == "noniid":
        filepath = config.CLIENTS_NONIID_FILE
    else:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose 'iid' or 'noniid'.")

    if not filepath.exists():
        raise FileNotFoundError(
            f"Client data file not found at {filepath}. "
            "Please run prepare_data.py or notebooks/01_data.ipynb first."
        )

    client_npz = np.load(filepath)
    clients_data = []
    for i in range(config.NUM_CLIENTS):
        X_i = client_npz[f"X_{i}"].astype(np.float32)
        y_i = client_npz[f"y_{i}"].astype(np.float32)
        clients_data.append((X_i, y_i))

    return clients_data


def make_client_datasets(
    scenario: str = "iid",
    batch_size: int = config.CLIENT_BATCH_SIZE,
    epochs: int = config.CLIENT_EPOCHS,
    shuffle: bool = True,
    seed: int = config.SEED,
) -> List[tf.data.Dataset]:
    """
    Package client partitions into a list of tf.data.Dataset ready for TFF simulation.
    Ensures correct element_spec: (TensorSpec(None, 30), TensorSpec(None, 1)).

    Args:
        scenario: 'iid' or 'noniid'
        batch_size: Batch size for local client training (default: 32)
        epochs: Number of local epochs (default: 1)
        shuffle: Whether to shuffle client data
        seed: Random seed for shuffling

    Returns:
        List of tf.data.Dataset objects for 5 clients.
    """
    client_raw = load_client_data(scenario=scenario)
    client_datasets = []

    for i, (X_i, y_i) in enumerate(client_raw):
        # Ensure labels have shape (N, 1) for binary crossentropy in TFF
        y_col = y_i.reshape(-1, 1).astype(np.float32)
        X_mat = X_i.astype(np.float32)

        ds = tf.data.Dataset.from_tensor_slices((X_mat, y_col))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(y_i), 10000), seed=seed + i)
        if epochs > 1:
            ds = ds.repeat(epochs)
        ds = ds.batch(batch_size)
        client_datasets.append(ds)

    return client_datasets


def get_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calculate balanced class weights to handle severe fraud imbalance.
    weight_c = total_samples / (n_classes * count_c)

    Args:
        y: Binary labels array (0 and 1)

    Returns:
        Dictionary mapping class {0: weight_0, 1: weight_1}
    """
    y_flat = y.ravel().astype(int)
    n_samples = len(y_flat)
    n_classes = 2
    n_class_0 = np.sum(y_flat == 0)
    n_class_1 = np.sum(y_flat == 1)

    if n_class_0 == 0 or n_class_1 == 0:
        raise ValueError("Cannot calculate class weights with only one class present.")

    weight_0 = float(n_samples / (n_classes * n_class_0))
    weight_1 = float(n_samples / (n_classes * n_class_1))

    return {0: weight_0, 1: weight_1}


def get_client_stats(scenario: str = "iid") -> List[Dict[str, float]]:
    """
    Get summary statistics for each client in the specified scenario.

    Returns:
        List of dicts with client statistics (id, total, legit, fraud, fraud_pct).
    """
    client_raw = load_client_data(scenario=scenario)
    stats = []

    for i, (X_i, y_i) in enumerate(client_raw):
        total = len(y_i)
        fraud = int(np.sum(y_i == 1))
        legit = int(np.sum(y_i == 0))
        fraud_pct = (fraud / total * 100) if total > 0 else 0.0

        stats.append({
            "client_id": i,
            "total_samples": total,
            "legit_samples": legit,
            "fraud_samples": fraud,
            "fraud_rate_pct": round(fraud_pct, 4)
        })

    return stats
