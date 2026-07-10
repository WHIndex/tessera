#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_EXAMPLE="${ROOT_DIR}/configs/paths.example.env"
ENV_FILE="${ROOT_DIR}/configs/paths.env"

if [[ ! -f "${ENV_EXAMPLE}" ]]; then
  echo "[ERR] 未找到 ${ENV_EXAMPLE}"
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  cp "${ENV_EXAMPLE}" "${ENV_FILE}"
  echo "[init] 已复制 configs/paths.example.env -> configs/paths.env"
else
  echo "[init] 已存在 configs/paths.env"
fi

mkdir -p \
  "${ROOT_DIR}/artifacts" \
  "${ROOT_DIR}/artifacts/models" \
  "${ROOT_DIR}/artifacts/results" \
  "${ROOT_DIR}/artifacts/retrieval" \
  "${ROOT_DIR}/logs" \
  "${ROOT_DIR}/runs"

echo "[init] 已创建基础目录"
bash "${SCRIPT_DIR}/run_preflight.sh"
