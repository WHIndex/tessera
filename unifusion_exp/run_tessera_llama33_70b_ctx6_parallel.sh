#!/usr/bin/env bash
# Paper-facing Llama-3.3-70B ctx6 evaluation entry point.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${ROOT_DIR}/run_llama33_70b_ctx6_parallel_fixed.sh"
