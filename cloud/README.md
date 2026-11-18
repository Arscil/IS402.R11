# Phần cloud (Azure)

Tài nguyên nằm trong resource group `rg-fl-fraud` (East Asia), subscription Azure for Students.

| Tài nguyên | Tên | Loại | Vai trò |
|---|---|---|---|
| Storage Account | `stis402flfraud` | PaaS | Blob: `raw`, `processed`, `models` (LRS, Hot) |
| Function App | `func-is402-etl` | FaaS | ETL tự động khi có file CSV mới (Linux Consumption, Python 3.12) |
| Application Insights | `func-is402-etl` | PaaS / SaaS | Log và số lần chạy của Function |
| Virtual Machine | `vm-is402-train` | IaaS | Huấn luyện TFF (Ubuntu 22.04, D4s_v4 hoặc D2s_v4), deallocate khi không dùng |

Quyền: 4 thành viên có role **Contributor** trên resource group và **Storage Blob Data Contributor** trên Storage Account; VM dùng **managed identity** có cùng quyền Blob. Các script dùng đăng nhập Microsoft Entra ID, không lưu access key trong code.

## 0. Huấn luyện trên Azure VM (rubric 4.1, 1.3)

Tạo VM (chỉ mở SSH cho IP của mình, managed identity có quyền Blob):

```bash
AZ=".venv/Scripts/python -m azure.cli"
$AZ vm create -g rg-fl-fraud -n vm-is402-train --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
  --size Standard_D4s_v4 --admin-username azureuser --ssh-key-values ~/.ssh/id_rsa.pub \
  --public-ip-sku Standard --nsg-rule NONE --assign-identity "[system]"
$AZ role assignment create --assignee-object-id <principalId của VM> --assignee-principal-type ServicePrincipal \
  --role "Storage Blob Data Contributor" --scope <id của stis402flfraud>
$AZ network nsg rule create -g rg-fl-fraud --nsg-name vm-is402-trainNSG -n allow-ssh-my-ip --priority 1000 \
  --access Allow --protocol Tcp --destination-port-ranges 22 --source-address-prefixes <IP của bạn>/32
```

Trên VM (sau khi chép mã nguồn lên `~/is402`):

```bash
bash cloud/vm/setup_vm.sh                                   # Miniconda, Python 3.10, TF 2.14, TFF 0.87 (~3 phút)
nohup bash cloud/vm/run_training.sh > vm_run.log 2>&1 &     # tải processed/latest -> huấn luyện -> đánh giá -> đẩy lên models/
```

**Chế độ tiết kiệm đang bật (18/11/2026):** VM tự deallocate lúc 23:00 giờ Việt Nam (16:00 UTC); ổ đĩa OS chuyển sang HDD `Standard_LRS` khi không dùng (đổi lại `Premium_LRS`/`StandardSSD_LRS` bằng `az disk update --sku` nếu cần chạy nhanh); các file benchmark trong `raw/benchmark/` chuyển sang tầng **Cool**.

`run_training.sh` lấy dữ liệu từ `processed/latest` (đầu ra của ETL), nên chuỗi xử lý trên cloud là liên tục: **upload CSV → Function ETL → Blob → VM huấn luyện → Blob `models`**. Xong thì deallocate:

```bash
$AZ vm deallocate -g rg-fl-fraud -n vm-is402-train
```

### Kết quả 17/11/2026 (trọng số lớp `sqrt`, 10 epoch / 30 round, cùng dữ liệu)

| Thời gian huấn luyện | Laptop WSL (12 CPU) | VM D2s_v4 (2 vCPU) | VM D4s_v4 (4 vCPU) |
|---|---|---|---|
| Tập trung | 13,6 s | 15,9 s | 14,3 s |
| FL-IID | 53,9 s | 163,8 s | 89,2 s |
| FL-non-IID | 83,6 s | 150,0 s | 89,8 s |

| Tốc độ Blob (4 kết nối song song) | Laptop (mạng gia đình) | VM cùng region |
|---|---|---|
| Upload CSV 144 MB | 31 MB/s | 151 MB/s |
| Download CSV 144 MB | 26 MB/s | 221 MB/s |
| Upload / download Parquet 1 GB | 12 / 29 MB/s | 197 / 169 MB/s |
| Tải dữ liệu huấn luyện 82,6 MB từ `processed/latest` | — | 1,6–2,0 s |

Nhận xét: FL trên TFF chạy song song theo CPU nên tăng từ 2 lên 4 vCPU giảm gần một nửa thời gian FL-IID; laptop 12 CPU vẫn nhanh nhất về tính toán. Lợi thế rõ nhất của cloud là **đường truyền dữ liệu**: VM cùng region đọc ghi Blob nhanh hơn laptop 5–8 lần, và cả chuỗi chạy không cần máy cá nhân. Đọc 2,3 GB CSV bằng pandas trên VM mất 30,4 s, Parquet cùng dữ liệu chỉ 1,25 s (nhanh hơn 24 lần), xác nhận lợi ích của Parquet. Biểu đồ: `results/fig_compare_training_time.png`, `results/fig_compare_blob_speed.png` (`python cloud/compare_laptop_cloud.py`).

Kết quả mô hình (VM 4 vCPU, `results/vm_4vcpu/metrics_summary.csv`):

| Mô hình | F1 (ngưỡng 0,5) | PR-AUC | F1 (ngưỡng tối ưu) | Precision / Recall (ngưỡng tối ưu) |
|---|---|---|---|---|
| Tập trung | 0,437 | 0,744 | 0,805 | 0,857 / 0,758 |
| FL-IID | 0,707 | 0,736 | 0,811 | 0,888 / 0,747 |
| FL-non-IID | 0,733 | 0,715 | 0,809 | 0,897 / 0,737 |
| FL-IID, 3 epoch cục bộ | 0,746 | 0,791 | 0,825 | 0,890 / 0,768 |
| FL-IID, 5 epoch cục bộ | 0,773 | 0,799 | 0,832 | 0,923 / 0,758 |

## API dự đoán trên Azure Container Apps (rubric 4.2, 3.3, 3.4)

| Tài nguyên | Tên | Loại |
|---|---|---|
| Container Registry (Basic) | `acris402fl.azurecr.io`, image `fraud-api:v2` | PaaS |
| Container Apps (Consumption, min 0 / max 2 replica, 0,25 vCPU, 0,5 GiB) | `ca-fraud-eastasia`, `ca-fraud-centralindia` | PaaS (serverless container) |
| User-assigned identity | `id-is402-acrpull` (AcrPull, Container Apps kéo image không cần mật khẩu) | |

URL:
- https://ca-fraud-eastasia.wittysand-fc563556.eastasia.azurecontainerapps.io/docs
- https://ca-fraud-centralindia.nicemoss-c5d7190c.centralindia.azurecontainerapps.io/docs

Mô hình phục vụ: FL-IID 5 epoch cục bộ (`export_model.py` → `api/model/model.json`), ngưỡng tối ưu F1 0,9429; API tính bằng NumPy (khớp Keras trên 56.746 mẫu test, sai lệch tối đa 5×10⁻⁷), image không cần TensorFlow. Endpoint: `GET /health`, `GET /model-info`, `POST /predict` (1–1000 giao dịch thô `Time, V1..V28, Amount`). Kiểm tra: `python api/test_api.py --url <URL>`.

Build và deploy (ACR Tasks bị chặn trên Azure for Students, nên build trên VM bằng Podman định dạng OCI, đăng nhập ACR bằng managed identity của VM):

```bash
# trên VM: bash cloud/vm/build_push_image.sh acris402fl fraud-api v2
$AZ containerapp env create -g rg-fl-fraud -n cae-is402-eastasia -l eastasia --logs-destination none
$AZ containerapp create -n ca-fraud-eastasia -g rg-fl-fraud --environment cae-is402-eastasia \
  --image acris402fl.azurecr.io/fraud-api:v2 --registry-server acris402fl.azurecr.io \
  --registry-identity <id của id-is402-acrpull> --user-assigned <id của id-is402-acrpull> \
  --target-port 8000 --ingress external --min-replicas 0 --max-replicas 2 --cpu 0.25 --memory 0.5Gi --env-vars REGION=eastasia
```

### Độ trễ đo từ Việt Nam (18/11/2026, `python cloud/latency_test.py`)

| Tình huống (p50) | East Asia | Central India |
|---|---|---|
| Cold start: request đầu khi app đã tắt về 0 replica | 4.485 ms | 4.333 ms |
| Request đầu sau 7 phút không dùng, **min-replicas = 1** | 194 ms | — |
| App đang chạy, kết nối HTTPS mới mỗi request | 144 ms | 302 ms |
| App đang chạy, giữ kết nối (keep-alive) | **44 ms** (p95 49) | 92 ms (p95 96) |
| Gửi lô 100 giao dịch (mỗi giao dịch) | **0,55 ms** | 1,0 ms |
| Thời gian xử lý trên server | ~0,1 ms | ~0,16 ms |

Kết luận cho báo cáo: thời gian tính toán gần như bằng 0, độ trễ đến từ **mạng** (region xa gấp đôi), **bắt tay TLS** (giữ kết nối nhanh hơn ~3 lần) và **cold start** (giữ 1 replica nhanh hơn ~23 lần ở request đầu, đổi lại tốn phí replica thường trực; demo để min 0 cho tiết kiệm). Gửi theo lô giảm chi phí mạng trên mỗi giao dịch ~80 lần. Biểu đồ: `results/latency/fig_latency_summary.png`, `fig_latency_warm.png`.

Lỗi đã gặp khi deploy:

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `TasksOperationsNotAllowed` khi `az acr build` | Azure for Students không cho dùng ACR Tasks | Build trên VM |
| Docker Desktop trên laptop không khởi động (lỗi 1920 với file socket) | Lỗi Windows với AF_UNIX socket của Docker | Không dùng Docker cục bộ; build trên VM |
| `ExpressRegistrySystemIdentityNotSupported` | Môi trường Container Apps kiểu express không hỗ trợ system-assigned identity cho registry | Dùng user-assigned identity `id-is402-acrpull` |
| Revision `Failed`: `unsupported MIME type ... docker.image.rootfs.diff.tar.gzip` | Môi trường express chỉ nhận image OCI; Docker legacy builder tạo manifest Docker v2 | Build bằng Podman `--format oci` (image `v2`) |

## Trang web gọi API (rubric 4.4)

URL công khai: **https://stis402flfraud.z7.web.core.windows.net/**

Trang tĩnh trong `web/` (HTML, CSS, JavaScript thuần), host bằng **Static Website của Storage Account** `stis402flfraud` (container `$web`, PaaS, gần như không tốn phí). Trình duyệt gọi thẳng API Container Apps qua HTTPS (API bật CORS), đọc JSON trả về và hiển thị:

1. **Trạng thái API**: `GET /health`, `GET /model-info` (mô hình, ngưỡng, F1/Precision/Recall), chọn region East Asia hoặc Central India.
2. **Dự đoán một giao dịch**: chọn 1 trong 10 giao dịch thật của tập test (hoặc tự nhập), `POST /predict`, hiển thị nhãn, thanh xác suất so với ngưỡng, thời gian khứ hồi và thời gian xử lý trên server.
3. **Dự đoán hàng loạt từ CSV**: đọc file trên trình duyệt, chia lô 1.000 dòng, nếu có cột `Class` thì tự tính Accuracy, Precision, Recall; dòng dự đoán sai được tô màu.
4. **Đo độ trễ từ trình duyệt**: 20 request mỗi region, thấy rõ cold start ở request đầu.

Sửa URL API trong `web/config.js`. Deploy:

```bash
$AZ storage blob service-properties update --account-name stis402flfraud --auth-mode login \
  --static-website --index-document index.html --404-document index.html
$AZ storage blob upload-batch --account-name stis402flfraud --auth-mode login -d '$web' -s web --overwrite
```

Kiểm tra trên URL công khai (18/11/2026): mẫu #1 dự đoán GIAN LẬN (xác suất 0,988); file CSV 10 giao dịch: Accuracy 90%, Precision 100%, Recall 80%, 86 ms; độ trễ East Asia p50 63 ms, Central India request đầu 4,65 s (cold start) rồi p50 95 ms. Giao diện co giãn được trên điện thoại.

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
