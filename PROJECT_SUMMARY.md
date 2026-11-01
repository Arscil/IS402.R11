# Tóm tắt dự án: Học liên kết cho phát hiện gian lận thẻ tín dụng

**Bài tập 22** | Nhóm 4 người | Công cụ: Python 3.10, TensorFlow Federated 0.87.0, Keras

---

## 1. Mục tiêu

Huấn luyện mô hình phát hiện giao dịch thẻ tín dụng gian lận bằng **học liên kết (Federated Learning)**: dữ liệu nằm lại ở từng client (node), chỉ có trọng số mô hình được gửi về server để tổng hợp. Sau đó đánh giá mô hình và so sánh với huấn luyện tập trung.

Nội dung chính cần thể hiện trong bài:

- Hiểu và trình bày được học liên kết, thuật toán FedAvg.
- Xử lý dữ liệu mất cân bằng và không đồng nhất (non-IID).
- Huấn luyện phân tán trên 5 client giả lập.
- Đánh giá bằng Accuracy, Precision, Recall, F1-Score, kèm biểu đồ.

---

## 2. Dữ liệu

| Mục | Giá trị |
|---|---|
| Tên | Credit Card Fraud Detection (Kaggle, `mlg-ulb/creditcardfraud`) |
| File | `creditcard.csv` |
| Kích thước | 284.807 dòng, 31 cột |
| Cột | `Time`, `V1`-`V28` (đã qua PCA, ẩn danh), `Amount`, `Class` |
| Nhãn | `Class` = 1 là gian lận (492 mẫu, khoảng 0,17%), `Class` = 0 là hợp lệ |
| Đặc điểm | Toàn bộ là số, không thiếu giá trị, mất cân bằng lớp rất nặng |

---

## 3. Quy trình tiền xử lý (phiên bản chốt)

| Bước | Việc làm |
|---|---|
| 1 | Đọc dữ liệu, kiểm tra kích thước, kiểu dữ liệu, giá trị thiếu, dòng trùng (khoảng 1.000, tùy chọn loại bỏ và ghi rõ trong báo cáo) |
| 2 | EDA ngắn: phân bố lớp, `Amount`, `Time` |
| 3 | Tách X (30 đặc trưng) và y (`Class`) |
| 4 | Chia train/test 80/20 bằng `stratify=y`, `random_state=42` |
| 5 | Fit **RobustScaler** (hoặc StandardScaler) chỉ trên train, chỉ cho `Time` và `Amount`; áp dụng cho test. V1-V28 giữ nguyên |
| 6 | Chia tập train thành 5 client, hai kịch bản: **IID** và **non-IID** |
| 7 | Xử lý mất cân bằng bằng class weight, hoặc oversample cục bộ trong từng client sau khi chia |
| 8 | Đóng gói đầu ra và bàn giao |

**Đã bỏ so với quy trình cũ (Home Credit):** Grouping, One-Hot, Align columns (không có cột phân loại); Median Imputation (không thiếu giá trị); Outlier Capping (xóa mất tín hiệu gian lận); Top 20 correlated và Polynomial + Interaction (V1-V28 đã trực giao do PCA, tăng số đặc trưng làm tăng số trọng số phải truyền mỗi round).

**Hai kịch bản chia client:**

- **IID:** xáo trộn và chia stratified, mỗi client có tỷ lệ gian lận gần bằng nhau.
- **Non-IID:** chia lệch tỷ lệ gian lận giữa các client (phân phối Dirichlet trên nhãn, hoặc chia thủ công theo tỷ lệ). Mỗi client có ít nhất vài chục mẫu gian lận.

**Thứ tự cột cố định:** `Time, V1, ..., V28, Amount`.

---

## 4. Mô hình và huấn luyện

| Mục | Giá trị |
|---|---|
| Kiến trúc | Input(30) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, sigmoid) |
| Loss | Binary cross-entropy (kèm class weight) |
| Baseline | Cùng kiến trúc, huấn luyện tập trung trên toàn bộ train |
| Học liên kết | `tff.learning.models.from_keras_model` + `build_weighted_fed_avg`, client optimizer SGD |
| Số client | 5 |
| Số round | khoảng 20-50 (`NUM_ROUNDS = 30` trong `config.py`) |
| Kịch bản | FL-IID và FL-non-IID |

Ba mô hình cần có để so sánh: **Centralized**, **FL-IID**, **FL-non-IID**.

---

## 5. Đánh giá

- Gán trọng số mô hình toàn cục vào mô hình Keras, dự đoán trên tập test (ngưỡng mặc định 0,5, có thể thử điều chỉnh).
- Chỉ số: Accuracy, Precision, Recall, F1-Score, thêm ROC-AUC và PR-AUC.
- Vì dữ liệu mất cân bằng nặng, Accuracy luôn rất cao dù mô hình kém. Phân tích tập trung vào **Recall, Precision, F1 và PR-AUC**.
- Biểu đồ: loss/metric theo round, confusion matrix, đường ROC, đường Precision-Recall, biểu đồ cột so sánh 3 mô hình.

---

## 6. Phân công 4 người

| Người | Vai trò | Việc |
|---|---|---|
| **1** | Nền tảng và báo cáo | Việc 1, 2, 3, 14: lý thuyết, môi trường, tải dữ liệu, ghép báo cáo |
| **2** | Dữ liệu | Việc 4, 5, 6, 7: EDA, tiền xử lý, chia train/test, chia client |
| **3** | Lõi huấn luyện | Việc 8, 9, 10: mô hình Keras, baseline, huấn luyện TFF (giữ liền mạch) |
| **4** | Đánh giá và phân tích | Việc 11, 12, 13: chỉ số, biểu đồ, phân tích kết quả |

Người 3 là điểm nghẽn (TFF dễ lỗi phiên bản và chạy chậm). Người 2 xong sớm nên hỗ trợ debug và chạy thử cấu hình cho người 3, nhưng không chia nhỏ phần code lõi.

---

## 7. Phần việc của Người 1 (chi tiết)

### Giai đoạn A: Dựng nền tảng (ưu tiên cao nhất, xong trong nửa ngày đầu)

1. Tự dựng môi trường theo `SETUP.md` (WSL2, Miniconda, môi trường `fl` Python 3.10, cài `requirements.txt`).
2. Tải `creditcard.csv`, đặt vào `data/raw/`, chạy `python smoke_test.py` đến khi thấy `Smoke test FedAvg: OK`. Nếu phần FedAvg lỗi do khác API, sửa luôn tại đây để người 3 dùng lại.
3. Tạo repo Git private, mời 3 bạn; thêm `.gitignore` (`data/`, `models/`, `results/*.npz`, `__pycache__/`, `.ipynb_checkpoints/`).
4. Viết `README.md`: cách cài, cách chạy kiểm tra, quy ước đặt tên notebook, quy ước nhánh Git (`data`, `model`, `eval`), nơi chia sẻ dữ liệu và trọng số.
5. Yêu cầu mỗi người tự chạy `smoke_test.py` thành công và báo lại. Máy không chạy được TFF (Mac Apple M, không cài được WSL) thì phân sang việc không cần TFF.
6. Chốt quy ước chung bằng một tin nhắn: dùng `config.py`; định dạng bàn giao; thời hạn từng phần.

### Giai đoạn B: Viết lý thuyết (trong lúc chờ các bạn khác)

- Định nghĩa học liên kết, so sánh với huấn luyện tập trung.
- Thuật toán FedAvg: server gửi mô hình, client huấn luyện cục bộ, server lấy trung bình có trọng số theo số mẫu, lặp theo round.
- Ưu điểm (riêng tư, giảm băng thông) và hạn chế (non-IID, chi phí giao tiếp, mất cân bằng lớp).
- Sơ đồ luồng FedAvg vẽ bằng draw.io.
- Đoạn mô tả quy trình tiền xử lý và lý do bỏ các bước cũ (có sẵn ở mục 3).

### Giai đoạn C: Dựng khung và ghép báo cáo

1. Mở đầu và lý thuyết (Người 1)
2. Dữ liệu và tiền xử lý (Người 2)
3. Mô hình và thiết lập liên kết (Người 3)
4. Kết quả và biểu đồ (Người 4)
5. Kết luận (cả nhóm, Người 1 tổng hợp)

Gửi dàn ý kèm hạn nộp từng phần. Cuối dự án: ghép, thống nhất văn phong, đánh số hình và bảng, tạo mục lục, kiểm tra notebook chạy được từ đầu đến cuối trên môi trường sạch.

---

## 8. Điểm giao tiếp cần thống nhất

1. **Định dạng dữ liệu:** tên file, tên cột, thứ tự đặc trưng (Người 2 → Người 3, 4).
2. **Kiến trúc mô hình:** chốt sớm (Input 30 → 64 → 32 → 1).
3. **Định dạng lưu kết quả:** tên file trọng số và history (Người 3 → Người 4). Lưu trọng số dạng `.npz` hoặc `.h5` để người khác đọc được mà không cần TFF.
4. **Random seed và cách chia dữ liệu dùng chung** (`SEED = 42` trong `config.py`).

---

## 9. Đầu ra của dự án

### 9.1 Mã nguồn và cấu hình

| File | Mô tả | Người phụ trách |
|---|---|---|
| `config.py` | Cấu hình chung: đường dẫn, seed, cột, kiến trúc, số client, số round | 1 |
| `requirements.txt` | Phiên bản thư viện đã ghim | 1 |
| `SETUP.md`, `README.md` | Hướng dẫn cài đặt và quy ước | 1 |
| `smoke_test.py` | Kiểm tra môi trường, dữ liệu, FedAvg | 1 |
| `notebooks/01_data.ipynb` | EDA, tiền xử lý, chia client | 2 |
| `data_utils.py` | `load_data()`, `make_client_datasets(scenario)` | 2 |
| `notebooks/02_model_fl.ipynb` | Mô hình, baseline, huấn luyện TFF (IID và non-IID) | 3 |
| `notebooks/03_eval.ipynb` | Đánh giá, biểu đồ, phân tích | 4 |

### 9.2 Dữ liệu và kết quả trung gian

| File | Nội dung |
|---|---|
| `data/processed/train.npz`, `test.npz` | `X` (30 cột, đã chuẩn hóa), `y` |
| `data/processed/clients_iid.npz`, `clients_noniid.npz` | `X_i`, `y_i` của 5 client |
| `data/processed/scaler.pkl` | Scaler đã fit trên train |
| Bảng thống kê client | Số mẫu và số mẫu gian lận của từng client, mỗi kịch bản |
| `models/` | Trọng số: baseline, FL-IID, FL-non-IID |
| `results/` | History loss/metric theo round (`.json`/`.pkl`), biểu đồ (`.png`) |

### 9.3 Biểu đồ và bảng trong báo cáo

- Phân bố lớp, phân bố `Amount` và `Time`.
- Bảng phân phối mẫu theo client (IID và non-IID).
- Đường loss/metric theo round (FL-IID so với FL-non-IID).
- Confusion matrix của 3 mô hình.
- Đường ROC và Precision-Recall.
- Bảng và biểu đồ cột so sánh Accuracy, Precision, Recall, F1, PR-AUC của Centralized, FL-IID, FL-non-IID.

### 9.4 Báo cáo cuối

Cấu trúc: Mở đầu và lý thuyết → Dữ liệu và tiền xử lý → Mô hình và thiết lập liên kết → Kết quả và biểu đồ → Kết luận. Kèm mã nguồn chạy được từ đầu đến cuối.

Phần phân tích (Việc 13) cần trả lời:

- Mô hình liên kết kém hoặc bằng baseline bao nhiêu?
- IID và non-IID ảnh hưởng thế nào đến độ hội tụ?
- Hạn chế (mất cân bằng lớp, số round, mô hình đơn giản) và hướng cải thiện (SMOTE cục bộ, FedProx, differential privacy).

---

## 10. Lịch gợi ý

| Ngày | Người 1 | Người 2 | Người 3 | Người 4 |
|---|---|---|---|---|
| 0,5 | Dựng môi trường, tải data, `smoke_test`, repo, README | Chờ data, đọc tài liệu | Cài môi trường, thử TFF với data giả | Viết hàm đánh giá |
| 1-1,5 | Viết lý thuyết | Việc 4-7 | Xây mô hình, baseline | Thử hàm trên mô hình giả (Logistic Regression) |
| 2-3 | Khung báo cáo | Hỗ trợ debug chia client | Chạy TFF IID và non-IID | Chuẩn bị biểu đồ |
| 3-4 | Ghép báo cáo | Viết mô tả dữ liệu | Viết mô tả mô hình | Chạy đánh giá, phân tích |

---

## 11. Rủi ro chính

| Rủi ro | Cách xử lý |
|---|---|
| TFF khó cài (chỉ Linux, Python 3.9-3.11; Colab mặc định Python 3.13 không dùng được) | Dùng WSL2 + conda Python 3.10, ghim `tensorflow-federated==0.87.0`; phân việc theo máy từng người |
| Cài nhầm `tensorflow-federated 0.1.0` (bản 2019) | Luôn cài từ `requirements.txt` đã ghim phiên bản |
| Mỗi người một môi trường, kết quả không tái lập | Cùng `requirements.txt`, cùng `config.py`, cùng seed |
| Non-IID làm một client không có mẫu gian lận | Đảm bảo mỗi client có ít nhất vài chục mẫu gian lận |
| Rò rỉ dữ liệu | Chia train/test trước, fit scaler chỉ trên train, oversample sau khi chia client |
| Nhầm Accuracy là chỉ số chính | Báo cáo nhấn mạnh Recall, Precision, F1, PR-AUC |

**Lưu ý đơn giản hóa cần ghi vào báo cáo:** scaler fit trên toàn bộ train trước khi chia client nghĩa là server đã thấy thống kê của mọi client; đây là đơn giản hóa của mô phỏng, cách đúng hơn là mỗi client tự fit scaler cục bộ.
