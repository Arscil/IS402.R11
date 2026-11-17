"""
Expand creditcard.csv to a multi-GB dataset for the storage / speed experiments
(rubric 1.3: data size at GB level, and laptop vs cloud processing speed).

Each copy k = 1..N-1 of the original data gets:
- Time shifted by k * 2 days (a later "period" of transactions),
- small Gaussian noise on V1..V28 and Amount (1% of each column's std),
- the same Class label, so the fraud rate stays ~0.17%.
Copy 0 is the original data unchanged.

The expanded data is ONLY for storage, ETL and speed benchmarks. Model training
and evaluation keep using the original dataset (ghi rõ trong báo cáo).

Usage:
    python cloud/expand_dataset.py --target-gb 2
    python cloud/expand_dataset.py --copies 14 --no-parquet
Outputs (data/expanded/, gitignored):
    creditcard_x<N>.csv, creditcard_x<N>.parquet
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

OUT_DIR = config.DATA_DIR / "expanded"
NOISE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TIME_SHIFT_SEC = 2 * 24 * 3600


def make_copy(df: pd.DataFrame, k: int, stds: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    if k == 0:
        return df
    part = df.copy()
    part["Time"] = part["Time"] + k * TIME_SHIFT_SEC
    noise = rng.normal(0.0, 0.01, size=(len(part), len(NOISE_COLUMNS))) * stds.values
    part[NOISE_COLUMNS] = part[NOISE_COLUMNS].values + noise
    part["Amount"] = part["Amount"].clip(lower=0).round(2)
    return part


def main():
    parser = argparse.ArgumentParser(description="Expand creditcard.csv to GB scale")
    parser.add_argument("--target-gb", type=float, default=2.0,
                        help="Approximate CSV size to reach (ignored if --copies is set)")
    parser.add_argument("--copies", type=int, default=None, help="Number of copies incl. original")
    parser.add_argument("--no-parquet", action="store_true", help="Only write the CSV file")
    args = parser.parse_args()

    src = config.RAW_DATA_FILE
    if not src.exists():
        raise SystemExit(f"Không tìm thấy {src}. Đặt creditcard.csv vào data/raw/ trước.")
    src_gb = src.stat().st_size / 1024 ** 3
    copies = args.copies or max(1, int(np.ceil(args.target_gb / src_gb)))

    df = pd.read_csv(src)
    stds = df[NOISE_COLUMNS].std()
    rng = np.random.default_rng(config.SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"creditcard_x{copies}.csv"
    parquet_path = OUT_DIR / f"creditcard_x{copies}.parquet"
    print(f"Nguồn: {src} ({src_gb * 1024:.1f} MB, {len(df):,} dòng) -> {copies} bản sao")

    writer = None
    if not args.no_parquet:
        import pyarrow as pa
        import pyarrow.parquet as pq

    t0 = time.perf_counter()
    for k in range(copies):
        part = make_copy(df, k, stds, rng)
        part.to_csv(csv_path, mode="w" if k == 0 else "a", header=(k == 0), index=False)
        if not args.no_parquet:
            table = pa.Table.from_pandas(part, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, table.schema, compression="snappy")
            writer.write_table(table)
        print(f"  Bản sao {k + 1}/{copies} xong ({time.perf_counter() - t0:.1f}s)")
    if writer is not None:
        writer.close()

    rows = len(df) * copies
    print(f"Tổng: {rows:,} dòng, gian lận ~{df['Class'].mean() * 100:.3f}%")
    print(f"CSV:     {csv_path} ({csv_path.stat().st_size / 1024 ** 3:.2f} GB)")
    if parquet_path.exists():
        print(f"Parquet: {parquet_path} ({parquet_path.stat().st_size / 1024 ** 3:.2f} GB)")
    print(f"Thời gian: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
