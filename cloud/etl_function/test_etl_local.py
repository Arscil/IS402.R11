"""
Local check that the cloud ETL (etl_core.run) gives exactly the same arrays as
prepare_data.py.

Usage (from the project root):
    python prepare_data.py                      # reference output in data/processed/
    python cloud/etl_function/test_etl_local.py --reference data/processed
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import etl_core  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(HERE.parent.parent / "data" / "raw" / "creditcard.csv"))
    parser.add_argument("--reference", required=True, help="Folder with prepare_data.py outputs")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = etl_core.run(pd.read_csv(args.csv))
    print(f"etl_core.run: {time.perf_counter() - t0:.1f}s")

    ref = Path(args.reference)
    checks = 0
    for name in ("train", "test", "clients_iid", "clients_noniid"):
        expected = np.load(ref / f"{name}.npz")
        actual = result[name]
        assert set(expected.files) == set(actual), f"{name}: khác danh sách mảng"
        for key in expected.files:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=f"{name}/{key}")
            checks += 1
    print(f"OK: {checks} mảng giống hệt prepare_data.py")

    try:
        import function_app
        files = function_app.build_outputs(result)
        print(f"OK: build_outputs tạo {len(files)} file, tổng {sum(map(len, files.values())) / 1e6:.1f} MB")
    except ImportError as e:
        print(f"Bỏ qua build_outputs (thiếu thư viện: {e})")
    print(result["stats"])


if __name__ == "__main__":
    main()
