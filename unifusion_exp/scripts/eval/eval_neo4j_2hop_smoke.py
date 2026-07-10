#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from neo4j import GraphDatabase


def main() -> int:
    parser = argparse.ArgumentParser(description="2-hop Neo4j smoke evaluation")
    parser.add_argument("--uri", default="bolt://127.0.0.1:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    parser.add_argument("--entity-file", type=Path, required=True)
    parser.add_argument("--max-entities", type=int, default=200)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--detail-file", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.entity_file.read_text(encoding="utf-8"))
    entities = data.get("entity_ids", [])[: args.max_entities]

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    latencies = []
    hit_flags = []
    neighbor_counts = []
    hit = 0
    with driver.session() as session:
        for e in entities:
            e2 = "/m/" + e[2:] if e.startswith("m.") else ("/g/" + e[2:] if e.startswith("g.") else e)
            t0 = time.time()
            res = session.run(
                """
                MATCH (s:Entity {id:$id})-[:REL*1..2]-(x:Entity)
                RETURN count(DISTINCT x) AS c
                """,
                id=e2,
            )
            c = res.single()["c"]
            dt = (time.time() - t0) * 1000
            latencies.append(dt)
            neighbor_counts.append(int(c))
            if c > 0:
                hit += 1
                hit_flags.append(1)
            else:
                hit_flags.append(0)

    driver.close()

    latencies.sort()
    p95 = latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0.0
    out = {
        "entities": len(entities),
        "hit_entities": hit,
        "hit_rate": hit / len(entities) if entities else 0.0,
        "latency_ms_avg": sum(latencies) / len(latencies) if latencies else 0.0,
        "latency_ms_p95": p95,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.detail_file is not None:
        detail = {
            "entities": len(entities),
            "entity_ids": entities,
            "hit": hit_flags,
            "neighbors_2hop": neighbor_counts,
            "latency_ms": latencies,
        }
        args.detail_file.parent.mkdir(parents=True, exist_ok=True)
        args.detail_file.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] detail -> {args.detail_file}")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[OK] saved -> {args.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
