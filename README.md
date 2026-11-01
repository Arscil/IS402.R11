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
├── data/
│   ├── raw/              # Chứa file creditcard.csv gốc
│   └── processed/        # Dữ liệu train, test, client IID/Non-IID sau tiền xử lý
├── models/               # Lưu file trọng số mô hình (.npz, .h5)
├── results/              # Lịch sử training (loss, metrics) và biểu đồ (.png)
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

### 4. Quy ước làm việc nhóm
- **Quy ước nhánh Git:**
  - `main`: Nhánh chính, ổn định.
  - `data`: Người 2 phụ trách tiền xử lý và chia dữ liệu.
  - `model`: Người 3 phụ trách mô hình và huấn luyện TFF.
  - `eval`: Người 4 phụ trách đánh giá và biểu đồ.
- **Quy ước cấu hình:** Toàn bộ tham số seed (`SEED = 42`), thứ tự cột, đường dẫn file bắt buộc import từ `config.py`.
- **Dữ liệu lớn và Trọng số:** Không push file dữ liệu (`.csv`, `.npz`) và model weights lên Git (đã được cấu hình trong `.gitignore`).

