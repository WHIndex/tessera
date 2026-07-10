#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path


def _load_entity_ids(path: Path) -> set[str]:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("entity_ids", []))


def _normalize_entity_id(entity_id: str) -> str:
    # mmRAG style: m.0abc / g.0xyz
    # SimKGC FB15k style: /m/0abc / /g/0xyz
    if entity_id.startswith("m."):
        return "/m/" + entity_id[2:]
    if entity_id.startswith("g."):
        return "/g/" + entity_id[2:]
    if entity_id.startswith("/m/"):
        return "m." + entity_id[3:]
    if entity_id.startswith("/g/"):
        return "g." + entity_id[3:]
    return entity_id


def _iter_triples(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            yield parts[0], parts[1], parts[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train TransE on mmRAG-induced graph subset")
    parser.add_argument("--entity-id-file", type=Path, required=True)
    parser.add_argument("--triples-tsv", type=Path, required=True, help="TSV: head\trelation\ttail")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-triples", type=int, default=None, help="Small-sample test mode")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=str, default="gpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    entity_ids = _load_entity_ids(args.entity_id_file)
    normalized_entity_ids = set(entity_ids)
    normalized_entity_ids.update({_normalize_entity_id(eid) for eid in entity_ids})
    print(f"[stage] loaded entity ids: {len(entity_ids)}")
    print(f"[stage] expanded normalized ids: {len(normalized_entity_ids)}")

    subset_path = args.out_dir / "mmrag_subset_triples.tsv"
    start = time.time()
    kept = 0
    scanned = 0

    with subset_path.open("w", encoding="utf-8") as out:
        for h, r, t in _iter_triples(args.triples_tsv):
            scanned += 1
            if h in normalized_entity_ids or t in normalized_entity_ids:
                out.write(f"{h}\t{r}\t{t}\n")
                kept += 1
            if args.max_triples is not None and kept >= args.max_triples:
                break
            if scanned % 1000000 == 0:
                elapsed = time.time() - start
                print(f"[progress] scanned={scanned} kept={kept} elapsed_s={elapsed:.1f}")

    elapsed = time.time() - start
    print(f"[stage] subset triples built: scanned={scanned} kept={kept} time_s={elapsed:.1f}")

    if args.dry_run:
        print("[dry-run] stop before pykeen training")
        return 0

    try:
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory
    except Exception as e:
        print("[ERR] pykeen is required for training. Install with: pip install pykeen")
        print(f"[ERR] import failure: {e}")
        return 2

    tf = TriplesFactory.from_path(str(subset_path), create_inverse_triples=True)
    print("[stage] start pykeen pipeline")
    result = pipeline(
        training=tf,
        testing=tf,
        model="TransE",
        model_kwargs={"embedding_dim": args.embedding_dim},
        training_loop="sLCWA",
        negative_sampler="basic",
        optimizer="Adam",
        training_kwargs={"num_epochs": args.epochs, "batch_size": args.batch_size, "use_tqdm": True},
        random_seed=42,
        device=args.device,
    )

    embeddings = result.model.entity_representations[0]().detach().cpu().numpy()
    import numpy as np

    emb_path = args.out_dir / "entity_embeddings.npy"
    np.save(emb_path, embeddings)

    meta = {
        "subset_triples": kept,
        "embedding_dim": args.embedding_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "embedding_file": str(emb_path),
    }
    import json

    (args.out_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] embeddings saved -> {emb_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
