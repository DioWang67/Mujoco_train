#!/bin/bash
# 在遠端機器上執行這個腳本，把輸出回傳給我
# Run this on the REMOTE machine first, then tell me the output.

echo "=== Remote Machine Info ==="
echo "OS:     $(uname -a)"
echo "Python: $(python3 --version 2>&1 || python --version 2>&1)"
echo "CUDA:   $(nvidia-smi 2>/dev/null | grep 'CUDA Version' || echo 'nvidia-smi not found')"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "glibc:  $(ldd --version | head -1)"
echo "pip:    $(pip3 --version 2>&1 || pip --version 2>&1)"
