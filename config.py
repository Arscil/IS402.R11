"""
Global configuration file for Federated Learning Credit Card Fraud Detection project.
Centralized source of truth for paths, seeds, column definitions, and hyperparameters.
"""

from pathlib import Path

# Random seed for reproducibility
SEED = 42

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_FILE = RAW_DATA_DIR / "creditcard.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.npz"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.npz"
CLIENTS_IID_FILE = PROCESSED_DATA_DIR / "clients_iid.npz"
CLIENTS_NONIID_FILE = PROCESSED_DATA_DIR / "clients_noniid.npz"
SCALER_FILE = PROCESSED_DATA_DIR / "scaler.pkl"

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Features and Column definitions
# Fixed column order: Time, V1..V28, Amount (30 features total)
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"
NUM_FEATURES = len(FEATURE_COLUMNS)  # 30

# Data preprocessing parameters
TEST_SIZE = 0.20
COLUMNS_TO_SCALE = ["Time", "Amount"]
# Đề bài yêu cầu StandardScaler hoặc MinMaxScaler cho Time và Amount.
# "standard" -> StandardScaler, "minmax" -> MinMaxScaler
SCALER_TYPE = "standard"

# Federated Learning Setup
NUM_CLIENTS = 5
NUM_ROUNDS = 30
CLIENT_BATCH_SIZE = 32
CLIENT_EPOCHS = 1

# Optimization & Model Hyperparameters
CLIENT_LEARNING_RATE = 0.02
SERVER_LEARNING_RATE = 1.0

# Neural Network Architecture
# Input(30) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, sigmoid)
HIDDEN_UNITS = [64, 32]
ACTIVATION = "relu"
OUTPUT_ACTIVATION = "sigmoid"

# Centralized baseline training
CENTRAL_EPOCHS = 10
CENTRAL_BATCH_SIZE = 256
CENTRAL_LEARNING_RATE = 0.001

# Class weighting for the fraud class (dùng chung cho baseline và FL)
USE_CLASS_WEIGHT = True

# Evaluation
DEFAULT_THRESHOLD = 0.5

# Model and result files
MODEL_NAMES = ["centralized", "fl_iid", "fl_noniid"]


def model_file(name: str) -> Path:
    """Keras model file for a given experiment name (e.g. 'fl_iid')."""
    return MODELS_DIR / f"{name}.keras"


def history_file(name: str) -> Path:
    """Training history (JSON) for a given experiment name."""
    return RESULTS_DIR / f"history_{name}.json"

