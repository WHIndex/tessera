#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/services/docker-compose.yml"

echo "[services] stopping neo4j + milvus"
docker compose -f "${COMPOSE_FILE}" down
