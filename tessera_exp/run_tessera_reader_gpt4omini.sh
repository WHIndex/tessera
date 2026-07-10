#!/usr/bin/env bash
# Paper-facing GPT-4o-mini reader entry point for TESSERA rankings.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export METHOD_LABEL="${METHOD_LABEL:-tessera}"
export RANKING_METHOD="${RANKING_METHOD:-tessera}"
exec bash "${ROOT_DIR}/run_reader_from_rankings_gpt4omini.sh"
