"""
Smoke test for the fraud API, locally or on Azure.

Usage:
    python api/test_api.py --url http://127.0.0.1:8000
    python api/test_api.py --url https://<app>.<region>.azurecontainerapps.io
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

SAMPLES = Path(__file__).resolve().parent / "samples" / "samples.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))

    with httpx.Client(timeout=60) as client:
        t0 = time.perf_counter()
        health = client.get(f"{base}/health").json()
        print(f"GET /health ({(time.perf_counter() - t0) * 1000:.0f} ms): {health}")
        info = client.get(f"{base}/model-info").json()
        print(f"GET /model-info: model={info['model']} threshold={info['threshold']:.4f} "
              f"region={info['region']} metrics={info['metrics_test']}")

        body = {"transactions": [s["transaction"] for s in samples]}
        t0 = time.perf_counter()
        resp = client.post(f"{base}/predict", json=body)
        rtt = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        out = resp.json()
        print(f"POST /predict ({len(samples)} giao dịch): {rtt:.0f} ms khứ hồi, "
              f"server xử lý {out['server_time_ms']} ms, phát hiện {out['fraud_count']} gian lận")
        print(f"{'Thực tế':<10} {'Dự đoán':<10} {'Xác suất':>10}  Amount")
        correct = 0
        for s, r in zip(samples, out["results"]):
            correct += s["actual"] == r["label"]
            print(f"{s['actual']:<10} {r['label']:<10} {r['fraud_probability']:>10.4f}  {s['transaction']['Amount']}")
        print(f"Đúng {correct}/{len(samples)}")

        bad = client.post(f"{base}/predict", json={"transactions": [{"Time": 1}]})
        print(f"Kiểm tra dữ liệu thiếu cột -> HTTP {bad.status_code} (mong đợi 422)")
        return 0 if resp.status_code == 200 and bad.status_code == 422 else 1


if __name__ == "__main__":
    sys.exit(main())
