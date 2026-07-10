#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENV_FILE="${ROOT_DIR}/configs/paths.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[ERR] 未找到 ${ENV_FILE}"
  echo "请先执行: cp configs/paths.example.env configs/paths.env"
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

echo "[preflight] ROOT_DIR=${ROOT_DIR}"

echo "[preflight] 1) 环境检查"
bash "${SCRIPT_DIR}/check_env.sh"

echo "[preflight] 2) 资源检查"
python "${SCRIPT_DIR}/check_resources.py" --strict

echo "[preflight] 完成，满足进入编码阶段的基础条件。"
