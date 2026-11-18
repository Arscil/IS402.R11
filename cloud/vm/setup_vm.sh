#!/bin/bash
# One-time setup of the training VM (Ubuntu 22.04): Miniconda + Python 3.10 + TF 2.14 + TFF 0.87.
# Run on the VM from the project root:  bash cloud/vm/setup_vm.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

T0=$(date +%s)
sudo apt-get update -qq
sudo apt-get install -y -qq unzip >/dev/null

if [ ! -d "$HOME/miniconda3" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
conda env list | grep -q "^fl " || conda create -y -q -n fl python=3.10
conda activate fl

pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q "azure-identity>=1.15" "azure-storage-blob>=12.19,<13" "pyarrow>=14"

python -c "import tensorflow as tf, tensorflow_federated as tff; print('TF', tf.__version__, '| TFF', tff.__version__)" 2>&1 | grep '^TF'
echo "CPU: $(nproc) vCPU | $(lscpu | grep 'Model name' | sed 's/Model name:\s*//')"
echo "Setup xong sau $(( $(date +%s) - T0 )) giây"
