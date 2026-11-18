"""
Download / upload files between Azure Blob Storage and the training VM.

Authentication: DefaultAzureCredential, i.e. the VM's system-assigned managed identity
(role "Storage Blob Data Contributor" on the storage account). No key in code.
On a laptop the same script works after `az login`.

Usage:
    # ETL output -> data/processed (train.npz, test.npz, clients_*.npz, scaler.pkl)
    python cloud/vm/blob_sync.py download --container processed --prefix latest/ \
        --dest data/processed --suffix .npz .pkl
    # trained models and results -> container `models`
    python cloud/vm/blob_sync.py upload --container models --src models --prefix runs/<run_id>/models/
"""

import argparse
import time
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


def service(account: str) -> BlobServiceClient:
    return BlobServiceClient(f"https://{account}.blob.core.windows.net",
                             credential=DefaultAzureCredential())


def download(args):
    container = service(args.account).get_container_client(args.container)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    total, t0 = 0, time.perf_counter()
    for blob in container.list_blobs(name_starts_with=args.prefix):
        rel = blob.name[len(args.prefix):]
        if "/" in rel or (args.suffix and not rel.endswith(tuple(args.suffix))):
            continue
        with open(dest / rel, "wb") as fh:
            container.download_blob(blob.name, max_concurrency=args.concurrency).readinto(fh)
        total += blob.size
        print(f"  tải {blob.name} ({blob.size / 1e6:.1f} MB)")
    sec = time.perf_counter() - t0
    print(f"Tải xong {total / 1e6:.1f} MB trong {sec:.1f}s ({total / 1e6 / max(sec, 1e-9):.1f} MB/s)")


def upload(args):
    container = service(args.account).get_container_client(args.container)
    src = Path(args.src)
    total, t0 = 0, time.perf_counter()
    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        if args.suffix and not path.name.endswith(tuple(args.suffix)):
            continue
        name = args.prefix + path.relative_to(src).as_posix()
        with open(path, "rb") as fh:
            container.upload_blob(name, fh, overwrite=True, max_concurrency=args.concurrency)
        total += path.stat().st_size
        print(f"  đẩy {name} ({path.stat().st_size / 1e6:.2f} MB)")
    sec = time.perf_counter() - t0
    print(f"Đẩy xong {total / 1e6:.1f} MB trong {sec:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Sync files with Azure Blob Storage")
    parser.add_argument("action", choices=["download", "upload"])
    parser.add_argument("--account", default="stis402flfraud")
    parser.add_argument("--container", required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--dest", default=".")
    parser.add_argument("--src", default=".")
    parser.add_argument("--suffix", nargs="*", default=None, help="Only files ending with these")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    download(args) if args.action == "download" else upload(args)


if __name__ == "__main__":
    main()
