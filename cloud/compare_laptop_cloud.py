"""
Compare the laptop (WSL) with the Azure VMs: training time and Blob read/write speed
(rubric 1.3 and 3.1).

Inputs (already produced by the other scripts):
    results/exp_class_weight/history_{central,fl_iid,fl_noniid}_sqrt.json  laptop, 12 CPU
    results/vm_4vcpu/history_{centralized,fl_iid,fl_noniid}.json           VM D4s_v4
    results/vm_2vcpu/history_{centralized,fl_iid,fl_noniid}.json           VM D2s_v4
    results/blob_benchmark_laptop*.csv, results/vm_4vcpu/blob_benchmark_vm_4vcpu*.csv
All runs use the same data, class weight "sqrt", 10 epochs / 30 rounds.

Usage:  python cloud/compare_laptop_cloud.py
Outputs: results/compare_laptop_cloud.csv, results/fig_compare_training_time.png,
         results/fig_compare_blob_speed.png
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))

MACHINES = [
    ("Laptop (WSL, 12 CPU)", RES / "exp_class_weight", {"centralized": "central_sqrt",
                                                        "fl_iid": "fl_iid_sqrt",
                                                        "fl_noniid": "fl_noniid_sqrt"}),
    ("Azure VM D2s_v4 (2 vCPU)", RES / "vm_2vcpu", None),
    ("Azure VM D4s_v4 (4 vCPU)", RES / "vm_4vcpu", None),
]
MODELS = [("centralized", "Tập trung"), ("fl_iid", "FL-IID"), ("fl_noniid", "FL-non-IID")]


def training_times() -> pd.DataFrame:
    rows = []
    for machine, folder, names in MACHINES:
        for key, label in MODELS:
            path = folder / f"history_{(names or {}).get(key, key)}.json"
            if not path.exists():
                print(f"Bỏ qua (chưa có): {path.relative_to(ROOT)}")
                continue
            hist = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"machine": machine, "model": label,
                         "train_time_sec": hist["train_time_sec"],
                         "cpu_count": hist.get("machine", {}).get("cpu_count"),
                         "test_f1": hist["steps"][-1]["test_f1"],
                         "test_pr_auc": hist["steps"][-1]["test_pr_auc"]})
    return pd.DataFrame(rows)


def blob_speeds() -> pd.DataFrame:
    frames = []
    for path, machine in [(RES / "blob_benchmark_laptop.csv", "Laptop (mạng gia đình)"),
                          (RES / "blob_benchmark_laptop_gb.csv", "Laptop (mạng gia đình)"),
                          (RES / "vm_4vcpu" / "blob_benchmark_vm_4vcpu.csv", "VM cùng region"),
                          (RES / "vm_4vcpu" / "blob_benchmark_vm_4vcpu_gb.csv", "VM cùng region")]:
        if path.exists():
            df = pd.read_csv(path)
            df["machine"] = machine
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_training(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    machines = list(dict.fromkeys(df["machine"]))
    x = np.arange(len(MODELS))
    width = 0.8 / len(machines)
    for i, m in enumerate(machines):
        sub = df[df["machine"] == m].set_index("model")
        vals = [sub["train_time_sec"].get(label, np.nan) for _, label in MODELS]
        bars = ax.bar(x + (i - (len(machines) - 1) / 2) * width, vals, width, label=m)
        ax.bar_label(bars, fmt="%.0f s", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in MODELS])
    ax.set_ylabel("Thời gian huấn luyện (giây)")
    ax.set_title("Thời gian huấn luyện: laptop và Azure VM (10 epoch / 30 round)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_blob(df: pd.DataFrame, out: Path):
    sub = df[df["concurrency"] == 4]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, col, title in ((axes[0], "upload_mbps", "Ghi (upload) lên Blob"),
                           (axes[1], "download_mbps", "Đọc (download) từ Blob")):
        pivot = sub.pivot_table(index="file", columns="machine", values=col)
        pivot.plot(kind="bar", ax=ax, rot=0)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.0f", fontsize=8, padding=2)
        ax.set_title(f"{title}, 4 kết nối song song", fontweight="bold")
        ax.set_ylabel("MB/s")
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    t = training_times()
    b = blob_speeds()
    t.to_csv(RES / "compare_laptop_cloud.csv", index=False, encoding="utf-8-sig")
    if not t.empty:
        plot_training(t, RES / "fig_compare_training_time.png")
        print(t.pivot_table(index="model", columns="machine", values="train_time_sec").round(1).to_string())
    if not b.empty:
        plot_blob(b, RES / "fig_compare_blob_speed.png")
        cols = ["machine", "file", "size_mb", "concurrency", "upload_mbps", "download_mbps", "pandas_read_sec"]
        print(b[cols].to_string(index=False))
    print(f"Đã lưu vào {RES}")


if __name__ == "__main__":
    main()
