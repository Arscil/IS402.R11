"""
Model and metric helpers shared by the centralized baseline, the federated
training (TFF) and the evaluation step.

- build_keras_model(): Keras Dense network defined in config.py
- WeightedBinaryCrossentropy: class-weighted loss usable both in Keras fit()
  and inside TFF (tff.learning.models.from_keras_model does not accept the
  class_weight argument of fit(), so the weighting lives in the loss).
- compute_metrics(): Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
"""

from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config


def build_keras_model() -> tf.keras.Model:
    """Input(30) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, sigmoid)."""
    tf.keras.utils.set_random_seed(config.SEED)
    layers = [tf.keras.layers.InputLayer(input_shape=(config.NUM_FEATURES,))]
    for units in config.HIDDEN_UNITS:
        layers.append(tf.keras.layers.Dense(units, activation=config.ACTIVATION))
    layers.append(tf.keras.layers.Dense(1, activation=config.OUTPUT_ACTIVATION))
    return tf.keras.Sequential(layers, name="fraud_mlp")


@tf.keras.utils.register_keras_serializable(package="fl_fraud")
class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    """Binary cross-entropy with a separate weight for each class."""

    def __init__(self, pos_weight: float = 1.0, neg_weight: float = 1.0,
                 name: str = "weighted_bce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        loss = -(self.pos_weight * y_true * tf.math.log(y_pred)
                 + self.neg_weight * (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(loss, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"pos_weight": self.pos_weight, "neg_weight": self.neg_weight})
        return cfg


def make_loss(class_weights: Optional[Dict[int, float]]) -> tf.keras.losses.Loss:
    """Weighted BCE when class weights are given, plain BCE otherwise."""
    if class_weights is None:
        return tf.keras.losses.BinaryCrossentropy()
    return WeightedBinaryCrossentropy(pos_weight=class_weights[1],
                                      neg_weight=class_weights[0])


def make_keras_metrics():
    """Fresh metric objects (TFF needs new instances on every model_fn call)."""
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
    ]


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = config.DEFAULT_THRESHOLD) -> Dict[str, float]:
    """Classification metrics on predicted fraud probabilities."""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
    }
