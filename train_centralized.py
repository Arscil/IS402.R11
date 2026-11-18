"""
Centralized baseline: train the same Keras model on the whole training set.

Usage (inside the `fl` environment, from the project root):
    python train_centralized.py
    python train_centralized.py --epochs 5
Outputs:
    models/centralized.keras
    results/history_centralized.json
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensorflow as tf

import config
import data_utils
import model_utils


class TestMetricsCallback(tf.keras.callbacks.Callback):
    """Evaluate on the test set after every epoch, like FL does after every round."""

    def __init__(self, X_test, y_test, log):
        super().__init__()
        self.X_test, self.y_test, self.log = X_test, y_test, log

    def on_epoch_end(self, epoch, logs=None):
        prob = self.model.predict(self.X_test, batch_size=4096, verbose=0)
        test = model_utils.compute_metrics(self.y_test, prob)
        entry = {"step": epoch + 1, "train_loss": float(logs["loss"]),
                 "elapsed_sec": round(time.perf_counter() - self.log["_t0"], 2),
                 **{f"test_{k}": v for k, v in test.items()}}
        self.log["steps"].append(entry)
        print(f"Epoch {epoch + 1}: loss={logs['loss']:.4f} | test precision={test['precision']:.4f} "
              f"recall={test['recall']:.4f} f1={test['f1']:.4f} pr_auc={test['pr_auc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Centralized baseline training")
    parser.add_argument("--epochs", type=int, default=config.CENTRAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.CENTRAL_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.CENTRAL_LEARNING_RATE)
    parser.add_argument("--class-weight", choices=["balanced", "sqrt", "none"],
                        default=config.CLASS_WEIGHT_MODE)
    parser.add_argument("--name", default="centralized", help="Experiment name")
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = data_utils.load_data()
    class_weights = data_utils.get_class_weights(y_train, args.class_weight)
    print(f"Train: {X_train.shape} | Test: {X_test.shape} | class weights: {class_weights}")

    model = model_utils.build_keras_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss=model_utils.make_loss(class_weights),
                  metrics=model_utils.make_keras_metrics())

    log = {"name": args.name, "type": "centralized", "step_unit": "epoch",
           "config": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                      "class_weight_mode": args.class_weight, "class_weights": class_weights,
                      "scaler": config.SCALER_TYPE},
           "machine": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
           "steps": [], "_t0": time.perf_counter()}

    model.fit(X_train, y_train.reshape(-1, 1), epochs=args.epochs, batch_size=args.batch_size,
              shuffle=True, verbose=0,
              callbacks=[TestMetricsCallback(X_test, y_test, log)])
    log["train_time_sec"] = round(time.perf_counter() - log.pop("_t0"), 2)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(config.model_file(args.name))
    with open(config.history_file(args.name), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"Thời gian huấn luyện: {log['train_time_sec']} giây")
    print(f"Đã lưu mô hình: {config.model_file(args.name)}")
    print(f"Đã lưu lịch sử: {config.history_file(args.name)}")


if __name__ == "__main__":
    main()
