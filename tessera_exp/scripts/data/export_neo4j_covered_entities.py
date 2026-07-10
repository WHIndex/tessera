#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from neo4j import GraphDatabase


def main() -> int:
    parser = argparse.ArgumentParser(description="Export covered Neo4j entity IDs for fair 2-hop smoke evaluation")
    parser.add_argument("--uri", default="bolt://127.0.0.1:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    parser.add_argument("--max-entities", type=int, default=200)
    parser.add_argument("--min-degree", type=int, default=1)
    parser.add_argument("--sample", choices=["random", "top_degree"], default="random")
    parser.add_argument("--seed", type=int, default=20260326)
    parser.add_argument("--out-file", type=Path, required=True)
    args = parser.parse_args()

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    with driver.session() as session:
        rows = list(
            session.run(
                """
                MATCH (n:Entity)-[:REL]-()
                WITH n, count(*) AS deg
                WHERE deg >= $min_degree
                RETURN n.id AS id, deg
                """,
                min_degree=args.min_degree,
            )
        )
    driver.close()

    if not rows:
        raise SystemExit("No covered entities found in Neo4j")

    pairs = [(r["id"], int(r["deg"])) for r in rows if r.get("id")]
    if not pairs:
        raise SystemExit("No valid entity ids found in Neo4j")

    if args.sample == "top_degree":
        pairs.sort(key=lambda x: x[1], reverse=True)
        chosen = pairs[: args.max_entities]
    else:
        rng = random.Random(args.seed)
        rng.shuffle(pairs)
        chosen = pairs[: args.max_entities]

    out = {
        "entity_ids": [x[0] for x in chosen],
        "meta": {
            "count": len(chosen),
            "sample": args.sample,
            "seed": args.seed,
            "min_degree": args.min_degree,
            "source": "neo4j_covered_entities",
        },
    }

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["meta"], ensure_ascii=False, indent=2))
    print(f"[OK] saved -> {args.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
