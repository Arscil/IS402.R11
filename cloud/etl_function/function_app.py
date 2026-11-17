"""
Azure Function (Python v2 model): automatic ETL for the credit-card dataset.

Trigger: a new .csv blob in container `raw` under `incoming/`
         (e.g. raw/incoming/creditcard.csv). Other folders in `raw`, such as the
         GB-scale benchmark files, do not trigger the function.
Output:  container `processed`
         runs/<run_id>/...  every run is kept
         latest/...         always the most recent run (read by the training VM)
Files:   train.npz, test.npz, clients_iid.npz, clients_noniid.npz, scaler.pkl
         (same names and format as prepare_data.py, so train_*.py can use them directly),
         test.parquet, clients_*/client_<k>.parquet (columnar copy), manifest.json
"""

import io
import json
import logging
import os
import pickle
import time
from datetime import datetime, timezone

import azure.functions as func
import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient

import etl_core

app = func.FunctionApp()

OUTPUT_CONTAINER = os.environ.get("ETL_OUTPUT_CONTAINER", "processed")
SCALER_TYPE = os.environ.get("ETL_SCALER_TYPE", "standard")


def _npz_bytes(arrays: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _parquet_bytes(X: np.ndarray, y: np.ndarray) -> bytes:
    df = pd.DataFrame(X, columns=etl_core.FEATURE_COLUMNS)
    df[etl_core.TARGET_COLUMN] = y
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="snappy")
    return buf.getvalue()


def build_outputs(result: dict) -> dict:
    files = {
        "train.npz": _npz_bytes(result["train"]),
        "test.npz": _npz_bytes(result["test"]),
        "clients_iid.npz": _npz_bytes(result["clients_iid"]),
        "clients_noniid.npz": _npz_bytes(result["clients_noniid"]),
        "scaler.pkl": pickle.dumps(result["scaler"]),
        "test.parquet": _parquet_bytes(result["test"]["X"], result["test"]["y"]),
    }
    for scenario in ("iid", "noniid"):
        clients = result[f"clients_{scenario}"]
        for i in range(etl_core.NUM_CLIENTS):
            files[f"clients_{scenario}/client_{i + 1}.parquet"] = _parquet_bytes(
                clients[f"X_{i}"], clients[f"y_{i}"])
    return files


@app.blob_trigger(arg_name="blob", path="raw/incoming/{name}", connection="AzureWebJobsStorage")
def etl_creditcard(blob: func.InputStream):
    name = blob.name or ""
    if not name.lower().endswith(".csv"):
        logging.info("Bỏ qua %s (không phải file .csv)", name)
        return

    t_start = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logging.info("ETL bắt đầu: %s (%s bytes), run_id=%s", name, blob.length, run_id)

    t0 = time.perf_counter()
    df = pd.read_csv(io.BytesIO(blob.read()))
    t_extract = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = etl_core.run(df, scaler_type=SCALER_TYPE)
    t_transform = time.perf_counter() - t0
    logging.info("Transform xong: %s", json.dumps(result["stats"], ensure_ascii=False))

    t0 = time.perf_counter()
    files = build_outputs(result)
    service = BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])
    container = service.get_container_client(OUTPUT_CONTAINER)
    for prefix in (f"runs/{run_id}", "latest"):
        for path, data in files.items():
            container.upload_blob(f"{prefix}/{path}", data, overwrite=True)
    t_load = time.perf_counter() - t0

    manifest = {
        "run_id": run_id,
        "source_blob": name,
        "source_bytes": blob.length,
        "timings_sec": {"extract": round(t_extract, 2), "transform": round(t_transform, 2),
                        "load": round(t_load, 2), "total": round(time.perf_counter() - t_start, 2)},
        "outputs": {path: len(data) for path, data in files.items()},
        **result["stats"],
    }
    body = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
    for prefix in (f"runs/{run_id}", "latest"):
        container.upload_blob(f"{prefix}/manifest.json", body, overwrite=True)
    logging.info("ETL hoàn tất: %s", json.dumps(manifest["timings_sec"]))
