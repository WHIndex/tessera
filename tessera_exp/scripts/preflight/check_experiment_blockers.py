#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path
import importlib


def check_path(path: str):
    p = Path(path)
    return p.exists(), str(p)


def check_port(host: str, port: int, timeout: float = 1.0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        sock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Check blockers for full experiments")
    parser.add_argument("--milvus-host", default=os.getenv("MILVUS_HOST", "127.0.0.1"))
    parser.add_argument("--milvus-port", type=int, default=int(os.getenv("MILVUS_PORT", "19530")))
    parser.add_argument("--neo4j-host", default=os.getenv("NEO4J_HOST", "127.0.0.1"))
    parser.add_argument("--neo4j-port", type=int, default=int(os.getenv("NEO4J_PORT", "7687")))
    parser.add_argument("--ollama-host", default="127.0.0.1")
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--require-online-services", action="store_true", help="milvus/neo4j 端口不可达时判定为硬阻塞")
    args = parser.parse_args()

    required_env_paths = {
        "MMRAG_DATA_ROOT": os.getenv("MMRAG_DATA_ROOT", ""),
        "E5_MODEL_DIR": os.getenv("E5_MODEL_DIR", ""),
        "DEBERTA_MODEL_DIR": os.getenv("DEBERTA_MODEL_DIR", ""),
        "SIMKGC_ROOT": os.getenv("SIMKGC_ROOT", ""),
    }

    blockers = []

    print("[python dependencies]")
    for mod in ["transformers", "torch", "pymilvus", "neo4j", "ijson"]:
        try:
            importlib.import_module(mod)
            print(f"[OK] {mod}")
        except Exception:
            print(f"[ERR] {mod}: missing")
            blockers.append(f"python:{mod}")

    print("[paths]")
    for key, path in required_env_paths.items():
        if not path:
            print(f"[ERR] {key}: not set")
            blockers.append(f"env:{key}")
            continue
        ok, p = check_path(path)
        if ok:
            print(f"[OK] {key}: {p}")
        else:
            print(f"[ERR] {key}: missing {p}")
            blockers.append(f"path:{key}")

    print("\n[services]")
    service_checks = [
        ("milvus", args.milvus_host, args.milvus_port),
        ("neo4j", args.neo4j_host, args.neo4j_port),
        ("ollama", args.ollama_host, args.ollama_port),
    ]
    for name, host, port in service_checks:
        ok = check_port(host, port)
        if ok:
            print(f"[OK] {name}: {host}:{port}")
        else:
            print(f"[WARN] {name}: {host}:{port} not reachable")
            if args.require_online_services and name in {"milvus", "neo4j"}:
                blockers.append(f"service:{name}")

    if blockers:
        print("\n[summary] blockers found:")
        for b in blockers:
            print(f"- {b}")
        return 2

    print("\n[summary] no hard blockers for full experiments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
