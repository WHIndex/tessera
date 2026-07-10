#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/services/docker-compose.yml"

echo "[services] starting neo4j + milvus"
docker compose -f "${COMPOSE_FILE}" up -d

echo "[services] waiting for ports"
for i in $(seq 1 60); do
  MILVUS_OK=0
  NEO4J_OK=0
  if timeout 1 bash -lc "</dev/tcp/127.0.0.1/19530" 2>/dev/null; then MILVUS_OK=1; fi
  if timeout 1 bash -lc "</dev/tcp/127.0.0.1/7687" 2>/dev/null; then NEO4J_OK=1; fi

  if [[ "${MILVUS_OK}" -eq 1 && "${NEO4J_OK}" -eq 1 ]]; then
    echo "[OK] services are reachable"
    exit 0
  fi
  echo "[wait] ${i}/60 milvus=${MILVUS_OK} neo4j=${NEO4J_OK}"
  sleep 2
done

echo "[ERR] services not ready in time"
exit 1
