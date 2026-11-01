"""
Smoke test script for environment verification and TFF FedAvg pipeline check.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

def main():
    print("=" * 60)
    print("Bắt đầu kiểm tra môi trường (Smoke Test)...")
    print("=" * 60)

    # 1. Kiểm tra phiên bản TensorFlow và TFF
    try:
        import tensorflow as tf
        import tensorflow_federated as tff
        print(f"TensorFlow: {tf.__version__} | TFF: {tff.__version__}")
        print("Hello FL")
    except ImportError as e:
        print(f"[LỖI] Không thể import tensorflow hoặc tensorflow_federated: {e}")
        print("Vui lòng kích hoạt đúng môi trường (conda activate fl) theo SETUP.md.")
        sys.exit(1)

    # 2. Kiểm tra dữ liệu creditcard.csv
    raw_csv = config.RAW_DATA_FILE
    if raw_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(raw_csv)
            print(f"Du lieu OK: {df.shape}")
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể đọc file dữ liệu {raw_csv}: {e}")
    else:
        print(f"[THÔNG BÁO] Chưa tìm thấy file dữ liệu tại {raw_csv}.")
        print("  -> Bạn có thể tải creditcard.csv từ Kaggle và đặt vào thư mục data/raw/.")
        print("  -> Đang tiếp tục test luồng FedAvg với dữ liệu giả lập (synthetic data)...")

    # 3. Kiểm tra luồng FedAvg
    try:
        import numpy as np

        # Tạo dummy dataset cho 3 clients giả lập
        def create_dummy_client_dataset():
            x = np.random.randn(32, config.NUM_FEATURES).astype(np.float32)
            y = np.random.randint(0, 2, size=(32, 1)).astype(np.float32)
            ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(16)
            return ds

        dummy_client_datasets = [create_dummy_client_dataset() for _ in range(3)]
        sample_element_spec = dummy_client_datasets[0].element_spec

        def model_fn():
            keras_model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(config.NUM_FEATURES,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            return tff.learning.models.from_keras_model(
                keras_model=keras_model,
                input_spec=sample_element_spec,
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy()]
            )

        fed_avg_process = tff.learning.algorithms.build_weighted_fed_avg(
            model_fn=model_fn,
            client_optimizer_fn=tff.learning.optimizers.build_sgdm(
                learning_rate=config.CLIENT_LEARNING_RATE
            ),
            server_optimizer_fn=tff.learning.optimizers.build_sgdm(
                learning_rate=config.SERVER_LEARNING_RATE
            )
        )

        state = fed_avg_process.initialize()

        for round_num in range(1, 4):
            state, metrics = fed_avg_process.next(state, dummy_client_datasets)
            print(f"Round {round_num}: OK | client_work metrics computed")

        print("Smoke test FedAvg: OK")
        print("=" * 60)
        print("CHÚC MỪNG: Môi trường TFF và FedAvg hoạt động hoàn toàn chính xác!")
        print("=" * 60)

    except Exception as e:
        print(f"[LỖI khi chạy FedAvg]: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

