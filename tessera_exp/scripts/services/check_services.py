#!/usr/bin/env python3
from __future__ import annotations

import socket


def check(host: str, port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        s.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        s.close()


if __name__ == "__main__":
    for name, port in [("milvus", 19530), ("neo4j", 7687), ("ollama", 11434)]:
        ok = check("127.0.0.1", port)
        print(f"{name}: {'OK' if ok else 'DOWN'}")
