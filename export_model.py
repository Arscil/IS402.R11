"""
Export a trained Keras model for the prediction API.

The network is small (Input 30 -> Dense 64 ReLU -> Dense 32 ReLU -> Dense 1 sigmoid), so the
API runs it with plain NumPy instead of TensorFlow: the Docker image stays small and starts
fast (important for scale-to-zero cold starts). This script writes everything the API needs
into one JSON file and checks that NumPy gives the same probabilities as Keras.

Usage (inside the `fl` environment, from the project root):
    python export_model.py --model models/fl_iid_e5.keras --metrics results/vm_4vcpu/metrics_summary.json \
        --name fl_iid_e5
Output:
    api/model/model.json   weights, scaler (Time, Amount), threshold, metrics, column order
"""

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import tensorflow as tf

import config
import data_utils
import model_utils  # noqa: F401  (registers the custom loss)

OUT = Path(__file__).resolve().parent / "api" / "model" / "model.json"


def numpy_forward(layers, X):
    h = X
    for i, layer in enumerate(layers):
        h = h @ np.asarray(layer["W"]) + np.asarray(layer["b"])
        h = np.maximum(h, 0.0) if layer["activation"] == "relu" else 1.0 / (1.0 + np.exp(-h))
    return h.ravel()


def main():
    parser = argparse.ArgumentParser(description="Export a Keras model to JSON for the API")
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True, help="Experiment name used in metrics_summary.json")
    parser.add_argument("--metrics", required=True, help="metrics_summary.json from evaluate.py")
    parser.add_argument("--scaler", default=str(config.SCALER_FILE))
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            W, b = layer.get_weights()
            layers.append({"W": W.astype(float).tolist(), "b": b.astype(float).tolist(),
                           "activation": layer.get_config()["activation"]})

    with open(args.scaler, "rb") as f:
        scaler = pickle.load(f)
    metrics = next(m for m in json.loads(Path(args.metrics).read_text(encoding="utf-8"))
                   if m["name"] == args.name)

    payload = {
        "model_name": args.name,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_columns": config.FEATURE_COLUMNS,
        "scaled_columns": config.COLUMNS_TO_SCALE,
        "scaler": {"type": type(scaler).__name__,
                   "mean": getattr(scaler, "mean_", np.zeros(2)).tolist(),
                   "scale": scaler.scale_.tolist(),
                   "min": getattr(scaler, "min_", np.zeros(2)).tolist()},
        "threshold": metrics["threshold_tuned"],
        "default_threshold": 0.5,
        "metrics_test": {k: metrics[k] for k in ("precision_tuned", "recall_tuned", "f1_tuned",
                                                 "pr_auc", "roc_auc")},
        "layers": layers,
    }

    # Check: NumPy forward pass == Keras on the whole (already scaled) test set
    _, _, X_test, _ = data_utils.load_data()
    keras_prob = model.predict(X_test, batch_size=4096, verbose=0).ravel()
    numpy_prob = numpy_forward(layers, X_test.astype(np.float64))
    max_diff = float(np.max(np.abs(keras_prob - numpy_prob)))
    same_label = float(np.mean((keras_prob >= payload["threshold"]) == (numpy_prob >= payload["threshold"])))
    print(f"So khớp NumPy và Keras trên {len(X_test):,} mẫu test: sai lệch lớn nhất {max_diff:.2e}, "
          f"trùng nhãn {same_label * 100:.3f}%")
    if max_diff > 1e-4:
        raise SystemExit("Sai lệch quá lớn, không xuất mô hình.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Đã xuất {args.name} (ngưỡng {payload['threshold']:.4f}) -> {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
