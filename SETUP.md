# Hướng dẫn cài đặt môi trường

Dự án: **Học liên kết (Federated Learning) cho phát hiện gian lận thẻ tín dụng**

Mục tiêu: sau khi làm theo file này, bạn chạy được `python smoke_test.py` và thấy dòng `Smoke test FedAvg: OK`.

---

## 0. Yêu cầu hệ thống

TensorFlow Federated (TFF) bản **0.87.0** chỉ chạy được với:

| Yêu cầu | Giá trị |
|---|---|
| Hệ điều hành | **Linux x86_64** (Windows dùng WSL2) |
| Python | **3.10** (TFF hỗ trợ 3.9 đến 3.11) |
| TensorFlow | 2.14.x (tự động cài theo TFF) |

> **Không dùng được:** Windows cài trực tiếp (không qua WSL), macOS (đặc biệt chip Apple M), Python 3.12 trở lên, Google Colab mặc định (Python 3.13).
> Nếu máy bạn thuộc nhóm không dùng được, báo ngay cho Người 1 để được phân công việc không cần TFF.

---

## 1. Cài WSL2 (chỉ dành cho Windows 10/11)

Mở **PowerShell** bằng quyền Administrator:

```powershell
wsl --install -d Ubuntu
```

Khởi động lại máy khi được yêu cầu. Sau khi khởi động lại, cửa sổ Ubuntu sẽ tự mở và yêu cầu tạo tên người dùng và mật khẩu (mật khẩu không hiện ký tự khi gõ, đó là bình thường).

Kiểm tra:

```powershell
wsl -l -v
```

Cột `VERSION` của Ubuntu phải là `2`.

> Người dùng Linux bỏ qua bước này.

---

## 2. Cài Miniconda

Mở terminal Ubuntu (WSL) và chạy:

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

- Nhấn Enter để đọc điều khoản, gõ `yes` để đồng ý.
- Khi hỏi có chạy `conda init` không, chọn `yes`.

**Đóng terminal và mở lại**, sau đó kiểm tra:

```bash
conda --version
```

Nếu `conda` báo chấp nhận điều khoản dịch vụ (Terms of Service) khi tạo môi trường, chạy các lệnh mà thông báo đó gợi ý (thường là `conda tos accept ...`), rồi chạy lại lệnh tạo môi trường.

---

## 3. Tạo môi trường Python 3.10

```bash
conda create -n fl python=3.10 -y
conda activate fl
python --version
```

Kết quả phải là `Python 3.10.x`. Mỗi lần mở terminal mới, chạy lại `conda activate fl` trước khi làm việc.

---

## 4. Lấy mã nguồn dự án

Đặt dự án trong ổ Linux (`~/FL_CreditCard`), **không** đặt trong `/mnt/c/...` vì ổ Windows truy cập từ WSL rất chậm.

```bash
cd ~
git clone <LINK_REPO_CUA_NHOM> FL_CreditCard
cd FL_CreditCard
```

Nếu chưa có repo, tạo thủ công:

```bash
mkdir -p ~/FL_CreditCard/data/raw ~/FL_CreditCard/data/processed ~/FL_CreditCard/models ~/FL_CreditCard/results ~/FL_CreditCard/notebooks ~/FL_CreditCard/report
cd ~/FL_CreditCard
```

rồi copy `config.py`, `requirements.txt`, `smoke_test.py` vào thư mục này.

---

## 5. Cài thư viện

```bash
conda activate fl
pip install --upgrade pip
pip install -r requirements.txt
```

Quá trình cài có thể mất vài phút. Kiểm tra phiên bản:

```bash
python -c "import tensorflow as tf, tensorflow_federated as tff; print(tf.__version__, tff.__version__)"
```

Kết quả đúng: TensorFlow `2.14.x` và TFF `0.87.0`.

---

## 6. Đưa dữ liệu vào

1. Tải bộ dữ liệu **Credit Card Fraud Detection** trên Kaggle (dataset `mlg-ulb/creditcardfraud`), giải nén ra file `creditcard.csv`.
2. Đặt file vào `data/raw/creditcard.csv`.

Cách chép file từ Windows vào WSL: mở File Explorer, gõ vào thanh địa chỉ:

```
\\wsl$\Ubuntu\home\<tên_người_dùng>\FL_CreditCard\data\raw
```

rồi kéo thả `creditcard.csv` vào.

---

## 7. Chạy kiểm tra

```bash
cd ~/FL_CreditCard
conda activate fl
python smoke_test.py
```

**Kết quả đúng** cần có đủ các dòng:

```
TensorFlow: 2.14.x | TFF: 0.87.0
Hello FL
Du lieu OK: (284807, 31)
Round 1: ...
Round 2: ...
Round 3: ...
Smoke test FedAvg: OK
```

Các dòng cảnh báo về CUDA / GPU / `cudart` là **bình thường** (máy chạy bằng CPU), có thể bỏ qua.

---

## 8. Làm việc với notebook (VS Code)

1. Cài **VS Code** trên Windows, thêm extension **WSL** và **Jupyter**.
2. Trong terminal WSL, vào thư mục dự án và gõ `code .`
3. Mở file `.ipynb`, bấm **Select Kernel** ở góc phải trên, chọn môi trường `fl` (Python 3.10).

Hoặc chạy Jupyter trực tiếp:

```bash
conda activate fl
jupyter notebook --no-browser
```

rồi mở đường dẫn có `token` hiện ra trong trình duyệt trên Windows.

---

## 9. Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân và cách xử lý |
|---|---|
| `No matching distribution found for tensorflow-federated==0.87.0` | Sai Python hoặc sai hệ điều hành. Kiểm tra `python --version` phải là 3.10 và đang chạy trong WSL/Linux, không phải Windows. |
| `pip` cài ra `tensorflow-federated 0.1.0` | Bản cũ từ 2019, không dùng được. Gỡ bằng `pip uninstall -y tensorflow-federated`, rồi cài lại đúng `pip install -r requirements.txt` (đã ghim 0.87.0). |
| `ModuleNotFoundError: No module named 'config'` | Chưa chạy trong thư mục dự án. Chạy `cd ~/FL_CreditCard` trước, vì `config.py` nằm ở thư mục gốc. |
| `FileNotFoundError ... creditcard.csv` | Chưa đặt file vào `data/raw/`. Kiểm tra lại Bước 6. |
| `ValueError: Key backend ... is not a valid value` | Biến môi trường `MPLBACKEND` bị đặt sai. Chạy `export MPLBACKEND=Agg` rồi chạy lại. |
| `conda: command not found` | Chưa đóng/mở lại terminal sau khi cài Miniconda, hoặc chưa `conda init`. |
| Lệnh chạy rất chậm khi đọc file | Dự án đang nằm trong `/mnt/c/...`. Chuyển sang `~/FL_CreditCard`. |
| Cài thư viện bị đứng hoặc `Killed` | WSL hết RAM. Tạo file `C:\Users\<tên>\.wslconfig` với nội dung `[wsl2]` và `memory=6GB`, rồi chạy `wsl --shutdown` và mở lại. |

---

## 10. Quy ước làm việc chung

- Dùng `config.py` làm nguồn duy nhất cho seed, thứ tự cột, kiến trúc mô hình, số client. Không tự đặt lại các giá trị này trong notebook.
- Mỗi người chỉ sửa notebook của mình (`01_data`, `02_model_fl`, `03_eval`).
- Không đưa `data/` và `models/` lên Git (đã có trong `.gitignore`).
- Sau khi cài xong, báo vào nhóm: *"Đã chạy `smoke_test.py` thành công"* kèm phiên bản TFF của bạn.
