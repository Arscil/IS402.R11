"""
Data preparation script for Credit Card Fraud Detection.
Performs data validation, train/test splitting, RobustScaler fitting,
and client partitioning (IID & Non-IID with Dirichlet distribution).
Saves processed files to data/processed/ and visualization figures to results/.
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def run_pipeline():
    print("=" * 70)
    print("BẮT ĐẦU QUY TRÌNH TIỀN XỬ LÝ & CHIA CLIENT (GIAI ĐOẠN 2)")
    print("=" * 70)

    # 1. Đọc dữ liệu thô
    raw_path = config.RAW_DATA_FILE
    if not raw_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {raw_path}")

    print(f"\n[1/6] Đọc file dữ liệu từ: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"  -> Kích thước ban đầu: {df.shape[0]:,} dòng, {df.shape[1]} cột")

    # Kiểm tra thiếu giá trị
    null_counts = df.isnull().sum().sum()
    print(f"  -> Số giá trị thiếu (NaN): {null_counts}")

    # Kiểm tra dòng trùng lặp
    dup_counts = df.duplicated().sum()
    print(f"  -> Số dòng trùng lặp phát hiện: {dup_counts:,}")
    if dup_counts > 0:
        # Loại bỏ dòng trùng lặp theo quy ước dự án và ghi nhận vào báo cáo
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"  -> Đã loại bỏ dòng trùng. Kích thước sau làm sạch: {df.shape[0]:,} dòng")

    # Phân bố nhãn Class
    class_counts = df[config.TARGET_COLUMN].value_counts()
    fraud_count = class_counts.get(1, 0)
    legit_count = class_counts.get(0, 0)
    fraud_pct = (fraud_count / len(df)) * 100
    print(f"  -> Hợp lệ (Class 0): {legit_count:,} ({100 - fraud_pct:.2f}%)")
    print(f"  -> Gian lận (Class 1): {fraud_count:,} ({fraud_pct:.3f}%)")

    # 2. Tạo biểu đồ EDA nhanh và lưu vào results/
    print("\n[2/6] Tạo biểu đồ trực quan hóa EDA...")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Biểu đồ 1: Phân bố nhãn (log scale)
    sns.countplot(x=config.TARGET_COLUMN, data=df, ax=axes[0], palette=["#2ecc71", "#e74c3c"])
    axes[0].set_yscale("log")
    axes[0].set_title("Phân bố nhãn Class (Log Scale)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Nhãn (0: Hợp lệ, 1: Gian lận)")
    axes[0].set_ylabel("Số lượng mẫu (log)")
    for p in axes[0].patches:
        height = p.get_height()
        axes[0].annotate(f"{int(height):,}", (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=11, xytext=(0, 5),
                         textcoords='offset points')

    # Biểu đồ 2: Phân bố giá trị Amount
    sns.histplot(df["Amount"], bins=50, ax=axes[1], color="#3498db", kde=True)
    axes[1].set_yscale("log")
    axes[1].set_title("Phân bố số tiền giao dịch Amount (Log Scale)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Số tiền (Amount)")
    axes[1].set_ylabel("Tần suất (log)")

    # Biểu đồ 3: Phân bố thời gian Time
    sns.histplot(df["Time"], bins=50, ax=axes[2], color="#9b59b6", kde=True)
    axes[2].set_title("Phân bố thời gian Time (Giây)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Thời gian (Time)")
    axes[2].set_ylabel("Tần suất")

    plt.tight_layout()
    eda_fig_path = config.RESULTS_DIR / "eda_distribution.png"
    plt.savefig(eda_fig_path, dpi=300)
    plt.close()
    print(f"  -> Đã lưu biểu đồ EDA tại: {eda_fig_path}")

    # 3. Tách X, y và chia tập Train/Test 80/20 Stratified
    print("\n[3/6] Phân chia tập Train/Test (80/20, stratify=y)...")
    # Đảm bảo thứ tự cột đúng theo config: Time, V1..V28, Amount (30 cột)
    X = df[config.FEATURE_COLUMNS].values
    y = df[config.TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
        stratify=y
    )
    print(f"  -> Train set: {X_train.shape[0]:,} mẫu | Gian lận: {int(np.sum(y_train == 1)):,}")
    print(f"  -> Test set:  {X_test.shape[0]:,} mẫu | Gian lận: {int(np.sum(y_test == 1)):,}")

    # 4. Chuẩn hóa RobustScaler cho cột Time (index 0) và Amount (index 29)
    print("\n[4/6] Fit RobustScaler chỉ trên tập Train (Time & Amount)...")
    scale_indices = [0, 29]  # Cột 0: Time, Cột 29: Amount
    
    scaler = RobustScaler()
    # Fit và transform trên train
    X_train[:, scale_indices] = scaler.fit_transform(X_train[:, scale_indices])
    # Transform trên test
    X_test[:, scale_indices] = scaler.transform(X_test[:, scale_indices])

    # Lưu train.npz, test.npz, scaler.pkl
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.TRAIN_DATA_FILE, X=X_train.astype(np.float32), y=y_train.astype(np.float32))
    np.savez_compressed(config.TEST_DATA_FILE, X=X_test.astype(np.float32), y=y_test.astype(np.float32))
    with open(config.SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  -> Đã lưu: {config.TRAIN_DATA_FILE}")
    print(f"  -> Đã lưu: {config.TEST_DATA_FILE}")
    print(f"  -> Đã lưu: {config.SCALER_FILE}")

    # 5. Phân chia 5 clients - Kịch bản IID
    print(f"\n[5/6] Phân chia {config.NUM_CLIENTS} clients theo kịch bản IID...")
    skf = StratifiedKFold(n_splits=config.NUM_CLIENTS, shuffle=True, random_state=config.SEED)
    clients_iid = {}
    
    for i, (_, client_idx) in enumerate(skf.split(X_train, y_train)):
        clients_iid[f"X_{i}"] = X_train[client_idx].astype(np.float32)
        clients_iid[f"y_{i}"] = y_train[client_idx].astype(np.float32)

    np.savez_compressed(config.CLIENTS_IID_FILE, **clients_iid)
    print(f"  -> Đã lưu kịch bản IID: {config.CLIENTS_IID_FILE}")

    # 6. Phân chia 5 clients - Kịch bản Non-IID (Dirichlet alpha=0.5 với min fraud sample constraint)
    print(f"\n[6/6] Phân chia {config.NUM_CLIENTS} clients theo kịch bản Non-IID (Dirichlet)...")
    rng = np.random.default_rng(config.SEED)

    fraud_indices = np.where(y_train == 1)[0]
    legit_indices = np.where(y_train == 0)[0]
    rng.shuffle(fraud_indices)
    rng.shuffle(legit_indices)

    n_fraud = len(fraud_indices)
    n_legit = len(legit_indices)
    
    # Ràng buộc: Mỗi client có tối thiểu 30 mẫu gian lận để đảm bảo có tín hiệu học tập
    min_fraud_per_client = 30
    fraud_reserved = min_fraud_per_client * config.NUM_CLIENTS
    remaining_fraud = n_fraud - fraud_reserved

    # Phân phối Dirichlet cho số mẫu gian lận còn lại (alpha=0.5)
    fraud_proportions = rng.dirichlet([0.5] * config.NUM_CLIENTS)
    fraud_counts = [min_fraud_per_client + int(np.round(p * remaining_fraud)) for p in fraud_proportions]
    
    # Điều chỉnh chênh lệch làm tròn cho client cuối
    diff_fraud = n_fraud - sum(fraud_counts)
    fraud_counts[-1] += diff_fraud

    # Phân chia mẫu hợp lệ (cho phân phối có độ lệch nhẹ)
    legit_proportions = rng.dirichlet([20.0] * config.NUM_CLIENTS)
    legit_counts = [int(np.round(p * n_legit)) for p in legit_proportions]
    diff_legit = n_legit - sum(legit_counts)
    legit_counts[-1] += diff_legit

    clients_noniid = {}
    f_ptr = 0
    l_ptr = 0

    for i in range(config.NUM_CLIENTS):
        c_f_indices = fraud_indices[f_ptr : f_ptr + fraud_counts[i]]
        c_l_indices = legit_indices[l_ptr : l_ptr + legit_counts[i]]
        f_ptr += fraud_counts[i]
        l_ptr += legit_counts[i]

        c_all_indices = np.concatenate([c_f_indices, c_l_indices])
        rng.shuffle(c_all_indices)

        clients_noniid[f"X_{i}"] = X_train[c_all_indices].astype(np.float32)
        clients_noniid[f"y_{i}"] = y_train[c_all_indices].astype(np.float32)

    np.savez_compressed(config.CLIENTS_NONIID_FILE, **clients_noniid)
    print(f"  -> Đã lưu kịch bản Non-IID: {config.CLIENTS_NONIID_FILE}")

    # Bảng thống kê so sánh phân phối giữa 2 kịch bản
    print("\n" + "=" * 80)
    print("BẢNG THỐNG KÊ PHÂN PHỐI CLIENT GIỮA 2 KỊCH BẢN (IID vs NON-IID)")
    print("=" * 80)
    headers = ["Client", "IID Total", "IID Fraud", "IID Fraud%", "Non-IID Total", "Non-IID Fraud", "Non-IID Fraud%"]
    print(f"{headers[0]:<8} | {headers[1]:<10} | {headers[2]:<10} | {headers[3]:<12} | {headers[4]:<14} | {headers[5]:<14} | {headers[6]:<14}")
    print("-" * 88)

    stats_for_plot = []

    for i in range(config.NUM_CLIENTS):
        y_iid = clients_iid[f"y_{i}"]
        total_iid = len(y_iid)
        fraud_iid = int(np.sum(y_iid == 1))
        pct_iid = (fraud_iid / total_iid) * 100

        y_noniid = clients_noniid[f"y_{i}"]
        total_noniid = len(y_noniid)
        fraud_noniid = int(np.sum(y_noniid == 1))
        pct_noniid = (fraud_noniid / total_noniid) * 100

        stats_for_plot.append({
            "client": f"Client {i+1}",
            "iid_total": total_iid,
            "iid_fraud": fraud_iid,
            "iid_pct": pct_iid,
            "noniid_total": total_noniid,
            "noniid_fraud": fraud_noniid,
            "noniid_pct": pct_noniid,
        })

        print(f"Client {i+1:<2} | {total_iid:<10,} | {fraud_iid:<10} | {pct_iid:<11.4f}% | {total_noniid:<14,} | {fraud_noniid:<14} | {pct_noniid:<13.4f}%")

    print("-" * 88)

    # Biểu đồ so sánh phân phối Client và lưu vào results/client_distribution.png
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    clients_label = [s["client"] for s in stats_for_plot]
    
    # Plot 1: Số mẫu gian lận (Fraud count)
    iid_frauds = [s["iid_fraud"] for s in stats_for_plot]
    noniid_frauds = [s["noniid_fraud"] for s in stats_for_plot]
    x = np.arange(len(clients_label))
    width = 0.35

    axes[0].bar(x - width/2, iid_frauds, width, label='IID', color='#2ecc71')
    axes[0].bar(x + width/2, noniid_frauds, width, label='Non-IID', color='#e74c3c')
    axes[0].set_ylabel('Số mẫu gian lận (Fraud count)')
    axes[0].set_title('So sánh số lượng mẫu gian lận theo Client', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(clients_label)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot 2: Tỷ lệ phần trăm gian lận (%)
    iid_pcts = [s["iid_pct"] for s in stats_for_plot]
    noniid_pcts = [s["noniid_pct"] for s in stats_for_plot]

    axes[1].bar(x - width/2, iid_pcts, width, label='IID', color='#2ecc71')
    axes[1].bar(x + width/2, noniid_pcts, width, label='Non-IID', color='#e74c3c')
    axes[1].set_ylabel('Tỷ lệ gian lận (%)')
    axes[1].set_title('So sánh tỷ lệ % gian lận theo Client', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(clients_label)
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    dist_fig_path = config.RESULTS_DIR / "client_distribution.png"
    plt.savefig(dist_fig_path, dpi=300)
    plt.close()
    print(f"\n  -> Đã lưu biểu đồ phân phối client tại: {dist_fig_path}")

    print("\n" + "=" * 70)
    print("HOÀN TẤT GIAI ĐOẠN 2: DỮ LIỆU ĐÃ SẴN SÀNG CHO BƯỚC HUẤN LUYỆN (GIAI ĐOẠN 3)")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
