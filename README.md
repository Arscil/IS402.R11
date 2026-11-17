# Federated Learning for Credit Card Fraud Detection
## Dự án: Học liên kết cho phát hiện gian lận thẻ tín dụng (Bài tập 22)

### 1. Giới thiệu
Dự án nghiên cứu và triển khai mô hình học liên kết (Federated Learning - FL) sử dụng thuật toán **FedAvg** để phát hiện giao dịch gian lận thẻ tín dụng trên bộ dữ liệu Credit Card Fraud Detection (Kaggle). Mô hình được huấn luyện trên 5 clients giả lập (IID và Non-IID) và so sánh với mô hình học tập trung (Centralized baseline).

### 2. Cấu trúc thư mục
```
├── config.py             # Cấu hình chung tập trung (Seed, Paths, Hyperparameters)
├── requirements.txt      # Danh sách thư viện cố định phiên bản
├── smoke_test.py         # Script kiểm tra môi trường và FedAvg
├── SETUP.md              # Hướng dẫn chi tiết cài đặt môi trường
├── PROJECT_SUMMARY.md    # Tóm tắt toàn bộ yêu cầu và lộ trình dự án
├── prepare_data.py       # Tiền xử lý: StandardScaler cho Time/Amount, chia train/test, chia 5 client
├── data_utils.py         # Đọc dữ liệu đã xử lý, tạo tf.data.Dataset cho TFF, class weight
├── model_utils.py        # Mô hình Keras Dense, loss có trọng số lớp, hàm tính chỉ số
├── train_centralized.py  # Baseline huấn luyện tập trung
├── train_federated.py    # Huấn luyện liên kết FedAvg bằng TFF (IID hoặc non-IID)
├── evaluate.py           # Accuracy, Precision, Recall, F1, ROC/PR-AUC và biểu đồ
├── data/
│   ├── raw/              # Chứa file creditcard.csv gốc
│   └── processed/        # Dữ liệu train, test, client IID/Non-IID sau tiền xử lý
├── models/               # Mô hình đã huấn luyện (.keras)
├── results/              # Lịch sử huấn luyện (history_*.json), bảng chỉ số và biểu đồ (.png)
├── notebooks/
│   ├── 01_data.ipynb     # EDA, tiền xử lý và chia dữ liệu client
│   ├── 02_model_fl.ipynb # Xây dựng mô hình, baseline & huấn luyện TFF
│   └── 03_eval.ipynb     # Đánh giá kết quả và trực quan hóa
└── report/               # Tài liệu báo cáo cuối kỳ
```

### 3. Cài đặt và Chạy thử
1. Đọc hướng dẫn chi tiết tại `SETUP.md` (yêu cầu chạy trên WSL2 / Linux x86_64, Python 3.10).
2. Tạo môi trường conda `fl`:
   ```bash
   conda create -n fl python=3.10 -y
   conda activate fl
   pip install -r requirements.txt
   ```
3. Tải dataset `creditcard.csv` từ Kaggle và đặt vào `data/raw/creditcard.csv`.
4. Chạy kiểm tra môi trường:
   ```bash
   python smoke_test.py
   ```

### 3.1. Chạy toàn bộ quy trình
Chạy trong môi trường `fl`, tại thư mục gốc của dự án:

```bash
export MPLBACKEND=Agg
python prepare_data.py                          # Bước 4: tiền xử lý và chia 5 client
python train_centralized.py                     # Baseline tập trung
python train_federated.py --scenario iid        # FL, dữ liệu đồng nhất
python train_federated.py --scenario noniid     # FL, dữ liệu không đồng nhất
python evaluate.py                              # Bước 6: chỉ số và biểu đồ
```

Thí nghiệm thêm (so sánh theo tham số) chỉ cần đặt tên khác, `evaluate.py` sẽ tự đưa vào biểu đồ:

```bash
python train_federated.py --scenario iid --local-epochs 3 --name fl_iid_e3
python train_federated.py --scenario iid --rounds 50 --name fl_iid_r50
```

Đối chiếu với đề bài:

| Yêu cầu | Thực hiện ở |
|---|---|
| Chuẩn hóa `Time`, `Amount` bằng StandardScaler hoặc MinMaxScaler | `prepare_data.py` (`config.SCALER_TYPE`) |
| Chia dữ liệu thành 4-5 client | `prepare_data.py` (5 client, IID và non-IID) |
| Mô hình Keras với các tầng Dense | `model_utils.build_keras_model` |
| Huấn luyện liên kết bằng TensorFlow Federated | `train_federated.py` (`build_weighted_fed_avg`) |
| Đánh giá bằng Accuracy, Precision, Recall, F1-Score | `evaluate.py`, `model_utils.compute_metrics` |
| Trực quan bằng biểu đồ | `evaluate.py` (5 biểu đồ trong `results/`) |

**Đơn giản hóa của mô phỏng (cần ghi vào báo cáo):** scaler và class weight được tính trên toàn bộ tập train trước khi chia client. Trong triển khai thật, mỗi client chỉ biết thống kê dữ liệu của chính mình.

### 4. Quy ước làm việc nhóm
- **Quy ước nhánh Git:**
  - `main`: Nhánh chính, ổn định.
  - `data`: Người 2 phụ trách tiền xử lý và chia dữ liệu.
  - `model`: Người 3 phụ trách mô hình và huấn luyện TFF.
  - `eval`: Người 4 phụ trách đánh giá và biểu đồ.
- **Quy ước cấu hình:** Toàn bộ tham số seed (`SEED = 42`), thứ tự cột, đường dẫn file bắt buộc import từ `config.py`.
- **Dữ liệu lớn và Trọng số:** Không push file dữ liệu (`.csv`, `.npz`) và model weights lên Git (đã được cấu hình trong `.gitignore`).

