# Phần cloud (Azure)

Tài nguyên nằm trong resource group `rg-fl-fraud` (East Asia), subscription Azure for Students.

| Tài nguyên | Tên | Loại | Vai trò |
|---|---|---|---|
| Storage Account | `stis402flfraud` | PaaS | Blob: `raw`, `processed`, `models` (LRS, Hot) |
| Function App | `func-is402-etl` | FaaS | ETL tự động khi có file CSV mới (Linux Consumption, Python 3.12) |
| Application Insights | `func-is402-etl` | PaaS / SaaS | Log và số lần chạy của Function |

Quyền: 4 thành viên có role **Contributor** trên resource group và **Storage Blob Data Contributor** trên Storage Account. Các script dùng đăng nhập Microsoft Entra ID, không lưu access key trong code.

## 1. Cài công cụ (Windows, không cần TFF)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r cloud/requirements-cloud.txt azure-cli
.venv/Scripts/python -m azure.cli login --tenant hcmuit.edu.vn
```

## 2. Mở rộng dữ liệu lên GB (rubric 1.3)

```bash
.venv/Scripts/python cloud/expand_dataset.py --target-gb 2
```

Tạo `data/expanded/creditcard_x15.csv` (2,27 GB, 4,27 triệu dòng) và bản Parquet (1,03 GB). Dữ liệu mở rộng chỉ dùng cho thí nghiệm lưu trữ và tốc độ; huấn luyện vẫn dùng dữ liệu gốc.

## 3. Đo tốc độ đọc ghi Blob (rubric 3.1, 3.4)

```bash
.venv/Scripts/python cloud/benchmark_blob.py --account stis402flfraud --files data/raw/creditcard.csv --concurrency 1 4 8 --label laptop
.venv/Scripts/python cloud/benchmark_blob.py --account stis402flfraud --files data/expanded/creditcard_x15.csv data/expanded/creditcard_x15.parquet --concurrency 4 --label laptop_gb
```

Kết quả: `results/blob_benchmark_<label>.json/.csv` và `results/fig_blob_benchmark_<label>.png`.

## 4. ETL tự động bằng Azure Function (rubric 3.2)

Luồng: upload CSV vào `raw/incoming/` → Blob trigger chạy `etl_creditcard` → ghi kết quả vào `processed/runs/<run_id>/` và `processed/latest/`.

```
raw/incoming/creditcard.csv
        │  Blob trigger
        ▼
func-is402-etl / etl_creditcard
  Extract : đọc CSV từ Blob
  Transform: kiểm tra schema, bỏ dòng trùng, tách 80/20, StandardScaler, chia 5 client IID và non-IID
  Load    : ghi .npz, .parquet, scaler.pkl, manifest.json
        ▼
processed/latest/   (VM huấn luyện đọc từ đây)
processed/runs/<run_id>/
```

Chỉ thư mục `raw/incoming/` kích hoạt Function; các file benchmark trong `raw/benchmark/` không kích hoạt.

Kiểm tra cục bộ trước khi deploy (so khớp từng mảng với `prepare_data.py`):

```bash
.venv/Scripts/python cloud/etl_function/test_etl_local.py --reference data/processed
```

Deploy (remote build trên Azure):

```bash
cd cloud/etl_function && python -c "import zipfile;z=zipfile.ZipFile('../../etl.zip','w',zipfile.ZIP_DEFLATED);[z.write(f) for f in ['function_app.py','etl_core.py','host.json','requirements.txt']];z.close()" && cd ../..
.venv/Scripts/python -m azure.cli functionapp deployment source config-zip -g rg-fl-fraud -n func-is402-etl --src etl.zip --build-remote true
```

Chạy ETL:

```bash
.venv/Scripts/python -m azure.cli storage blob upload --account-name stis402flfraud --auth-mode login -c raw -n incoming/creditcard.csv -f data/raw/creditcard.csv --overwrite
```

Kết quả chạy thật (17/11/2026, file 144 MB):

| Lần | File | Từ lúc upload xong tới khi Function chạy | Extract | Transform | Load | Tổng |
|---|---|---|---|---|---|---|
| 1 | `creditcard.csv` | ~9 giây (host vừa khởi động) | 0,55 s | 1,40 s | 11,27 s | 13,2 s |
| 2 | `creditcard_run2.csv` | ~10 phút (host đang ngủ) | 0,52 s | 1,16 s | 9,45 s | 11,1 s |

Manifest: `results/etl_manifest_latest.json`, `results/etl_manifest_run2.json`. Log lần 2 xem trong Application Insights (bảng `traces`), ảnh `report/images/m3-8_function-etl-logs.png`.

**Nhận xét cho mục 3.4:** Blob trigger kiểu polling trên gói Consumption có thể trễ tới khoảng 10 phút khi app đang ngủ. Hướng tối ưu: dùng Blob trigger nguồn Event Grid (`source="EventGrid"`) để kích hoạt gần như tức thì, hoặc gói Flex Consumption / Premium có instance luôn sẵn sàng. Bước Load chiếm phần lớn thời gian vì ghi 16 file hai lần (`runs/` và `latest/`); có thể chỉ ghi `runs/` rồi dùng lệnh copy phía server để tạo `latest/`.

Portal tab **Invocations** của Function trống vì nó đọc bảng `requests`, trong khi Function Python dùng Blob trigger chỉ ghi vào bảng `traces`. Xem log bằng truy vấn KQL:

```
traces | where message startswith 'Executing' or message startswith 'Executed' or message startswith 'ETL'
| project timestamp, message | order by timestamp asc
```

### Lỗi đã gặp khi deploy (ghi vào báo cáo)

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `0 functions found`, log báo `pyarrow requires NumPy 2.0` | `requirements.txt` giới hạn NumPy dưới 2 trong khi pyarrow mới cần NumPy 2 | Ghim phiên bản đúng bộ đã kiểm tra cục bộ |
| Remote build thất bại sau 13 giây | NumPy 2.5 yêu cầu Python 3.12, Function App đang dùng 3.11 | Đổi runtime sang `Python|3.12` |
| `'3.12' is not recognized` khi chạy `az.bat` | Ký tự `|` bị cmd hiểu là pipe | Gọi `python -m azure.cli` thay cho `az.bat` |
