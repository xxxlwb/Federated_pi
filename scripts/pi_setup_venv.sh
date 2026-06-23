#!/bin/bash
# Pi 上一键重建 venv + 装所有依赖.
# 在 Pi 上跑: bash setup_venv.sh
set -e
cd ~/fed_pi

echo "[setup] removing old .venv..."
rm -rf .venv

echo "[setup] creating new venv..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -1

echo "[setup] installing small deps via Tsinghua mirror..."
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir \
    numpy PyYAML psutil matplotlib pytest tqdm 2>&1 | tail -2

echo "[setup] installing torch CPU via official PyTorch index..."
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision 2>&1 | tail -3

echo "[setup] installing flwr via Tsinghua mirror..."
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir flwr 2>&1 | tail -2

echo "[setup] verifying..."
python -c 'import torch, torchvision, flwr; print("OK", torch.__version__, flwr.__version__)'

echo "[setup] DONE."
