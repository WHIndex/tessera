#!/usr/bin/env bash
set -euo pipefail

echo "[check_env] 开始环境检查"

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "[WARN] 当前未检测到 conda 环境激活"
else
  echo "[OK] CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"
fi

if command -v python >/dev/null 2>&1; then
  echo "[OK] python=$(python --version 2>&1)"
else
  echo "[ERR] python 不可用"
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[OK] nvidia-smi 可用"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  echo "[WARN] nvidia-smi 不可用，跳过 GPU 检查"
fi

if command -v ollama >/dev/null 2>&1; then
  echo "[OK] ollama 可用"
else
  echo "[WARN] ollama 不在 PATH 中"
fi

echo "[check_env] 完成"
