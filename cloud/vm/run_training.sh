#!/bin/bash
# Full training run on the Azure VM:
#   1. pull the ETL output (processed/latest) from Blob with the VM's managed identity
#   2. train centralized, FL-IID, FL-non-IID (+ local-epoch experiments)
#   3. evaluate and draw the charts
#   4. push models and results to the `models` container
# Run on the VM from the project root:
#   nohup bash cloud/vm/run_training.sh > vm_run.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fl
export MPLBACKEND=Agg TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1

RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
LABEL=${LABEL:-vm_$(nproc)vcpu}
EXPERIMENTS=${EXPERIMENTS:-full}   # full | core
echo "RUN_ID=$RUN_ID LABEL=$LABEL EXPERIMENTS=$EXPERIMENTS | $(nproc) vCPU | $(lscpu | grep 'Model name' | sed 's/Model name:\s*//')"

step() { echo; echo "=== $1 ($(date -u +%H:%M:%S))"; }

step "Tải dữ liệu đã xử lý từ Blob (processed/latest)"
python cloud/vm/blob_sync.py download --container processed --prefix latest/ \
  --dest data/processed --suffix .npz .pkl || exit 1

step "Huấn luyện tập trung"
python train_centralized.py
step "FL-IID"
python train_federated.py --scenario iid
step "FL-non-IID"
python train_federated.py --scenario noniid

if [ "$EXPERIMENTS" = "full" ]; then
  for e in 3 5; do
    step "FL-IID, $e epoch cục bộ mỗi round"
    python train_federated.py --scenario iid --local-epochs "$e" --name "fl_iid_e$e"
  done
fi

step "Đánh giá"
python evaluate.py

step "Đẩy mô hình và kết quả lên Blob (models/runs/$RUN_ID/)"
python cloud/vm/blob_sync.py upload --container models --src models --prefix "runs/$RUN_ID/models/" --suffix .keras
python cloud/vm/blob_sync.py upload --container models --src results --prefix "runs/$RUN_ID/results/" \
  --suffix .json .csv .png
echo; echo "XONG $RUN_ID"
