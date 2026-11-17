"""
Read/write speed benchmark against Azure Blob Storage (rubric 3.1, 3.4, 2.1).

For each file it measures:
- upload to Blob and download from Blob (MB/s), with several parallel-connection settings,
- reading the downloaded file into pandas (CSV vs Parquet).

Authentication uses Microsoft Entra ID (no account key in code): the signed-in user
needs the "Storage Blob Data Contributor" role on the storage account. Sign in once with
`az login`, or let the script open a browser window (InteractiveBrowserCredential).

Usage:
    python cloud/benchmark_blob.py --account stis402flfraud \
        --files data/raw/creditcard.csv data/expanded/creditcard_x14.parquet
    python cloud/benchmark_blob.py --account stis402flfraud --files ... --concurrency 1 4 8
Outputs (results/):
    blob_benchmark.json, blob_benchmark.csv, fig_blob_benchmark.png
"""

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


AUTH_RECORD = Path.home() / ".is402_azure_auth.json"
STORAGE_SCOPE = "https://storage.azure.com/.default"


def get_credential():
    """Sign in once in the browser, then reuse the cached token on later runs.

    The token cache is stored by the OS (Windows: DPAPI-encrypted); only a small
    authentication record (account name, tenant, no secret) is kept in AUTH_RECORD.
    """
    from azure.identity import (AuthenticationRecord, AzureCliCredential, ChainedTokenCredential,
                                InteractiveBrowserCredential, TokenCachePersistenceOptions)
    cache = TokenCachePersistenceOptions(name="is402_fl_fraud")
    record = None
    if AUTH_RECORD.exists():
        record = AuthenticationRecord.deserialize(AUTH_RECORD.read_text(encoding="utf-8"))
    browser = InteractiveBrowserCredential(cache_persistence_options=cache,
                                           authentication_record=record)
    if record is None:
        record = browser.authenticate(scopes=[STORAGE_SCOPE])
        AUTH_RECORD.write_text(record.serialize(), encoding="utf-8")
    return ChainedTokenCredential(AzureCliCredential(), browser)


def read_local(path: Path) -> float:
    t0 = time.perf_counter()
    if path.suffix == ".parquet":
        pd.read_parquet(path)
    else:
        pd.read_csv(path)
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description="Azure Blob read/write benchmark")
    parser.add_argument("--account", required=True, help="Storage account name")
    parser.add_argument("--container", default="raw")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--label", default=platform.node(), help="Where the benchmark runs (laptop, vm, ...)")
    parser.add_argument("--skip-read", action="store_true", help="Skip pandas read timing")
    args = parser.parse_args()

    from azure.storage.blob import BlobServiceClient
    service = BlobServiceClient(f"https://{args.account}.blob.core.windows.net",
                                credential=get_credential())
    container = service.get_container_client(args.container)
    if not container.exists():
        container.create_container()

    rows = []
    for f in args.files:
        path = Path(f)
        size_mb = path.stat().st_size / 1024 ** 2
        blob = container.get_blob_client(f"benchmark/{path.name}")
        read_sec = None if args.skip_read else read_local(path)
        for conc in args.concurrency:
            with open(path, "rb") as fh:
                t0 = time.perf_counter()
                blob.upload_blob(fh, overwrite=True, max_concurrency=conc)
                up = time.perf_counter() - t0
            with tempfile.NamedTemporaryFile(delete=False, suffix=path.suffix) as tmp:
                t0 = time.perf_counter()
                blob.download_blob(max_concurrency=conc).readinto(tmp)
                down = time.perf_counter() - t0
            os.unlink(tmp.name)
            row = {"label": args.label, "file": path.name, "format": path.suffix.lstrip("."),
                   "size_mb": round(size_mb, 1), "concurrency": conc,
                   "upload_sec": round(up, 2), "upload_mbps": round(size_mb / up, 2),
                   "download_sec": round(down, 2), "download_mbps": round(size_mb / down, 2),
                   "pandas_read_sec": None if read_sec is None else round(read_sec, 2)}
            rows.append(row)
            print(f"{path.name:<32} {size_mb:>8.1f} MB | conc={conc:<2} | upload {row['upload_mbps']:>7.2f} MB/s "
                  f"| download {row['download_mbps']:>7.2f} MB/s | pandas read {row['pandas_read_sec']}s")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"blob_benchmark_{args.label}"
    with open(config.RESULTS_DIR / f"{stem}.json", "w", encoding="utf-8") as fh:
        json.dump({"account": args.account, "machine": platform.platform(), "rows": rows}, fh, indent=2)
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / f"{stem}.csv", index=False)
    plot(df, config.RESULTS_DIR / f"fig_{stem}.png", args.label)
    print(f"Đã lưu kết quả vào {config.RESULTS_DIR}")


def plot(df: pd.DataFrame, out: Path, label: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, col, title in ((axes[0], "upload_mbps", "Tốc độ ghi (upload)"),
                           (axes[1], "download_mbps", "Tốc độ đọc (download)")):
        pivot = df.pivot_table(index="file", columns="concurrency", values=col)
        pivot.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(f"{title} - {label}", fontweight="bold")
        ax.set_ylabel("MB/s")
        ax.set_xlabel("")
        ax.legend(title="Số kết nối song song")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
