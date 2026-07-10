#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from neo4j import GraphDatabase


def normalize_ent(x: str) -> str:
    x = x.strip()
    if x.startswith("m."):
        return "/m/" + x[2:]
    if x.startswith("g."):
        return "/g/" + x[2:]
    return x


def load_triples(path: Path, max_triples: int) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            out.append((normalize_ent(h), r.strip(), normalize_ent(t)))
            if max_triples > 0 and len(out) >= max_triples:
                break
    return out


def batched_rows(triples: list[tuple[str, str, str]], batch_size: int):
    for i in range(0, len(triples), batch_size):
        b = triples[i : i + batch_size]
        rows = [{"h": h, "r": r, "t": t} for h, r, t in b]
        yield rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Load subset triples into Neo4j")
    parser.add_argument("--uri", default="bolt://127.0.0.1:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    parser.add_argument("--triple-file", type=Path, required=True)
    parser.add_argument("--max-triples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--out-file", type=Path, required=True)
    args = parser.parse_args()

    triples = load_triples(args.triple_file, args.max_triples)
    if not triples:
        raise SystemExit(f"No triples loaded from {args.triple_file}")

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    t0 = time.time()
    with driver.session() as session:
        if args.recreate:
            session.run("MATCH (n) DETACH DELETE n")

        session.run("CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")

        total = 0
        for rows in batched_rows(triples, args.batch_size):
            session.run(
                """
                UNWIND $rows AS row
                MERGE (h:Entity {id: row.h})
                MERGE (t:Entity {id: row.t})
                MERGE (h)-[r:REL {type: row.r}]->(t)
                """,
                rows=rows,
            )
            total += len(rows)
            if total % (args.batch_size * 5) == 0:
                print(f"[load] {total}/{len(triples)}")

        n_nodes = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
        n_rels = session.run("MATCH ()-[r:REL]->() RETURN count(r) AS c").single()["c"]

    driver.close()

    out = {
        "triple_file": str(args.triple_file),
        "triples_loaded": len(triples),
        "nodes": int(n_nodes),
        "rels": int(n_rels),
        "elapsed_sec": time.time() - t0,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[OK] saved -> {args.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
