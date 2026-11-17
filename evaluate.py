"""
Evaluate every trained model on the test set and draw the comparison charts.

Usage (inside the `fl` environment, from the project root, after training):
    python evaluate.py
    python evaluate.py --models centralized fl_iid fl_noniid
Outputs (results/):
    metrics_summary.csv, metrics_summary.json
    fig_metrics_comparison.png   Accuracy / Precision / Recall / F1 / PR-AUC per model
    fig_confusion_matrices.png   Confusion matrix per model
    fig_roc_pr_curves.png        ROC and Precision-Recall curves
    fig_training_curves.png      Train loss and test metrics per round / epoch
    fig_training_time.png        Training time per model
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, roc_curve

import config
import data_utils
import model_utils  # noqa: F401  (registers the custom loss for model loading)

DISPLAY_NAMES = {"centralized": "Tập trung", "fl_iid": "FL-IID", "fl_noniid": "FL-non-IID"}
MAIN_METRICS = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                ("f1", "F1-Score"), ("pr_auc", "PR-AUC")]


def label(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def find_models(requested):
    if requested:
        return requested
    available = {p.stem for p in config.MODELS_DIR.glob("*.keras")}
    ordered = [n for n in config.MODEL_NAMES if n in available]
    return ordered + sorted(available - set(ordered))


def load_history(name):
    path = config.history_file(name)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_metrics_comparison(summary: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    n = len(summary)
    width = 0.8 / max(n, 1)
    x = np.arange(len(MAIN_METRICS))
    for i, (_, row) in enumerate(summary.iterrows()):
        values = [row[key] for key, _ in MAIN_METRICS]
        bars = ax.bar(x + (i - (n - 1) / 2) * width, values, width, label=row["model"])
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([title for _, title in MAIN_METRICS])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Giá trị")
    ax.set_title("So sánh chỉ số trên tập test (ngưỡng 0,5)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_confusion_matrices(summary: pd.DataFrame, out: Path):
    n = len(summary)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), squeeze=False)
    for ax, (_, row) in zip(axes[0], summary.iterrows()):
        cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Hợp lệ", "Gian lận"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Hợp lệ", "Gian lận"])
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        ax.set_title(row["model"], fontweight="bold")
    fig.suptitle("Ma trận nhầm lẫn trên tập test", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_roc_pr(curves, y_test, out: Path):
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    for name, prob, roc_auc, pr_auc in curves:
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax_roc.plot(fpr, tpr, label=f"{label(name)} (AUC={roc_auc:.3f})")
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ax_pr.plot(rec, prec, label=f"{label(name)} (AP={pr_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.set_title("Đường ROC", fontweight="bold")
    ax_pr.axhline(y_test.mean(), color="k", linestyle="--", linewidth=1,
                  label=f"Ngẫu nhiên ({y_test.mean():.4f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Đường Precision-Recall", fontweight="bold")
    for ax in (ax_roc, ax_pr):
        ax.legend(loc="best", fontsize=9)
        ax.grid(linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_training_curves(histories, out: Path):
    panels = [("train_loss", "Loss huấn luyện"), ("test_f1", "F1-Score trên test"),
              ("test_recall", "Recall trên test"), ("test_pr_auc", "PR-AUC trên test")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, title) in zip(axes.ravel(), panels):
        for name, hist in histories.items():
            steps = [s["step"] for s in hist["steps"] if key in s]
            values = [s[key] for s in hist["steps"] if key in s]
            if steps:
                unit = "round" if hist["step_unit"] == "round" else "epoch"
                ax.plot(steps, values, marker="o", markersize=3, label=f"{label(name)} ({unit})")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Round (FL) / Epoch (tập trung)")
        ax.grid(linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)
    fig.suptitle("Quá trình huấn luyện", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_training_time(histories, out: Path):
    names = [n for n, h in histories.items() if "train_time_sec" in h]
    if not names:
        return
    times = [histories[n]["train_time_sec"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([label(n) for n in names], times, color="#4c72b0")
    ax.bar_label(bars, fmt="%.1f s", padding=2)
    machine = histories[names[0]].get("machine", {})
    ax.set_ylabel("Giây")
    ax.set_title(f"Thời gian huấn luyện ({machine.get('cpu_count', '?')} CPU)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on the test set")
    parser.add_argument("--models", nargs="*", help="Model names (default: every models/*.keras)")
    parser.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD)
    args = parser.parse_args()

    names = find_models(args.models)
    if not names:
        raise SystemExit("Chưa có mô hình nào trong models/. Hãy chạy train_centralized.py "
                         "và train_federated.py trước.")
    _, _, X_test, y_test = data_utils.load_data()
    print(f"Tập test: {len(y_test):,} mẫu, {int(y_test.sum())} gian lận")

    rows, curves, histories = [], [], {}
    for name in names:
        model = tf.keras.models.load_model(config.model_file(name), compile=False)
        prob = model.predict(X_test, batch_size=4096, verbose=0).ravel()
        metrics = model_utils.compute_metrics(y_test, prob, args.threshold)
        hist = load_history(name)
        if hist:
            histories[name] = hist
            metrics["train_time_sec"] = hist.get("train_time_sec")
        rows.append({"model": label(name), "name": name, **metrics})
        curves.append((name, prob, metrics["roc_auc"], metrics["pr_auc"]))

    summary = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.RESULTS_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_json(config.RESULTS_DIR / "metrics_summary.json", orient="records",
                    force_ascii=False, indent=2)

    plot_metrics_comparison(summary, config.RESULTS_DIR / "fig_metrics_comparison.png")
    plot_confusion_matrices(summary, config.RESULTS_DIR / "fig_confusion_matrices.png")
    plot_roc_pr(curves, y_test, config.RESULTS_DIR / "fig_roc_pr_curves.png")
    if histories:
        plot_training_curves(histories, config.RESULTS_DIR / "fig_training_curves.png")
        plot_training_time(histories, config.RESULTS_DIR / "fig_training_time.png")

    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "tp", "fp", "fn"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Đã lưu bảng và biểu đồ vào: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
