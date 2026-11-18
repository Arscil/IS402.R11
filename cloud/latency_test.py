"""
Latency test for the fraud API deployed in several Azure regions (rubric 3.3 and 3.4).

For each endpoint it measures, from this machine:
  keepalive  N requests of 1 transaction reusing one HTTPS connection (normal client)
  newconn    N requests of 1 transaction, new HTTPS connection each time (TLS handshake every call)
  batch      requests carrying 100 transactions each (latency per transaction)
and reports mean / p50 / p95 / min / max round-trip time plus the server-side processing time.

Usage:
    python cloud/latency_test.py --label warm \
        --url eastasia=https://... --url centralindia=https://...
    python cloud/latency_test.py --label cold --cold --url eastasia=https://...   # one request only
Outputs (results/latency/):
    latency_<label>.json, latency_<label>.csv, fig_latency_<label>.png
"""

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "api" / "samples" / "samples.json"
OUT = ROOT / "results" / "latency"


def pct(values, q):
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def summary(name, region, rtts, server_ms, per_tx=1):
    return {"region": region, "mode": name, "n": len(rtts), "tx_per_request": per_tx,
            "mean_ms": round(statistics.mean(rtts), 1), "p50_ms": round(pct(rtts, 0.5), 1),
            "p95_ms": round(pct(rtts, 0.95), 1), "min_ms": round(min(rtts), 1), "max_ms": round(max(rtts), 1),
            "server_p50_ms": round(pct(server_ms, 0.5), 3) if server_ms else None,
            "per_tx_p50_ms": round(pct(rtts, 0.5) / per_tx, 3)}


def timed_post(client, url, body):
    t0 = time.perf_counter()
    r = client.post(url, json=body)
    rtt = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return rtt, r.json()["server_time_ms"]


def run_region(region, base, n, batch_n):
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    one = {"transactions": [samples[0]["transaction"]]}
    hundred = {"transactions": [samples[i % len(samples)]["transaction"] for i in range(100)]}
    url = f"{base.rstrip('/')}/predict"
    rows = []

    with httpx.Client(timeout=60) as client:
        client.get(f"{base}/health")  # warm up the replica and the connection
        rtts, srv = zip(*[timed_post(client, url, one) for _ in range(n)])
        rows.append(summary("keepalive", region, rtts, srv))
        b_rtts, b_srv = zip(*[timed_post(client, url, hundred) for _ in range(batch_n)])
        rows.append(summary("batch100", region, b_rtts, b_srv, per_tx=100))

    rtts, srv = [], []
    for _ in range(n):
        with httpx.Client(timeout=60) as client:
            rtt, s = timed_post(client, url, one)
        rtts.append(rtt)
        srv.append(s)
    rows.append(summary("newconn", region, rtts, srv))
    return rows


def run_cold(region, base):
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    with httpx.Client(timeout=120) as client:
        t0 = time.perf_counter()
        r = client.post(f"{base.rstrip('/')}/predict", json={"transactions": [samples[0]["transaction"]]})
        rtt = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
    return [summary("cold_first_request", region, [rtt], [r.json()["server_time_ms"]])]


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    modes = list(dict.fromkeys(r["mode"] for r in rows))
    regions = list(dict.fromkeys(r["region"] for r in rows))
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(modes))
    width = 0.8 / len(regions)
    for i, reg in enumerate(regions):
        p50 = [next((r["p50_ms"] for r in rows if r["region"] == reg and r["mode"] == m), np.nan) for m in modes]
        p95 = [next((r["p95_ms"] for r in rows if r["region"] == reg and r["mode"] == m), np.nan) for m in modes]
        pos = x + (i - (len(regions) - 1) / 2) * width
        bars = ax.bar(pos, p50, width, label=f"{reg} (p50)")
        ax.errorbar(pos, p50, yerr=[np.zeros(len(p50)), np.array(p95) - np.array(p50)], fmt="none",
                    ecolor="black", capsize=4, linewidth=1)
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    labels = {"keepalive": "Giữ kết nối\n(1 giao dịch)", "newconn": "Kết nối mới mỗi lần\n(1 giao dịch)",
              "batch100": "Gửi lô\n(100 giao dịch)", "cold_first_request": "Request đầu\n(cold start)"}
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(m, m) for m in modes])
    ax.set_ylabel("Thời gian khứ hồi (ms), cột = p50, vạch = tới p95")
    ax.set_title("Độ trễ API theo region (đo từ Việt Nam)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Latency test for the fraud API")
    parser.add_argument("--url", action="append", required=True, help="region=https://... (repeatable)")
    parser.add_argument("--label", default="warm")
    parser.add_argument("-n", type=int, default=100, help="Requests per mode")
    parser.add_argument("--batch-n", type=int, default=20)
    parser.add_argument("--cold", action="store_true", help="Only time the first request (after scale-to-zero)")
    args = parser.parse_args()

    rows = []
    for item in args.url:
        region, base = item.split("=", 1)
        rows += run_cold(region, base) if args.cold else run_region(region, base, args.n, args.batch_n)

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"label": args.label, "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "client": "laptop, Viet Nam", "rows": rows}
    (OUT / f"latency_{args.label}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    import csv
    with open(OUT / f"latency_{args.label}.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot(rows, OUT / f"fig_latency_{args.label}.png")
    cols = ["region", "mode", "n", "mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "server_p50_ms", "per_tx_p50_ms"]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r[c]) for c in cols))
    print(f"Đã lưu vào {OUT}")


if __name__ == "__main__":
    main()
