#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from pymilvus import MilvusClient

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unifusion_exp.utils.e5_embed import encode_texts, load_e5


def main() -> int:
    parser = argparse.ArgumentParser(description="Index subset corpus into Milvus")
    parser.add_argument("--milvus-uri", type=str, default="http://127.0.0.1:19530")
    parser.add_argument("--collection", type=str, default="mmrag_subset_e5")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--max-corpus", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--insert-batch", type=int, default=1000)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--meta-file", type=Path, required=True)
    args = parser.parse_args()

    corpus = json.loads(args.corpus_file.read_text(encoding="utf-8"))
    if args.max_corpus is not None:
        corpus = corpus[: args.max_corpus]

    doc_ids = [d["id"] for d in corpus]
    texts = [d["text"] for d in corpus]

    tokenizer, model, device, resolved = load_e5(args.model_dir)
    print(f"[stage] encode corpus size={len(texts)} model={resolved} device={device}")
    vecs = encode_texts(texts, tokenizer, model, device, batch_size=args.batch_size)

    client = MilvusClient(uri=args.milvus_uri)

    if args.recreate and client.has_collection(args.collection):
        client.drop_collection(args.collection)

    if not client.has_collection(args.collection):
        client.create_collection(
            collection_name=args.collection,
            dimension=vecs.shape[1],
            metric_type="COSINE",
            auto_id=False,
            primary_field_name="id",
            id_type="int",
            vector_field_name="vector",
        )

    print(f"[stage] insert into {args.collection}")
    for i in range(0, len(corpus), args.insert_batch):
        batch = []
        for j, d in enumerate(corpus[i : i + args.insert_batch], start=i):
            batch.append({"id": j, "vector": vecs[j].tolist(), "doc_id": d["id"]})
        client.insert(collection_name=args.collection, data=batch)
        if (i // args.insert_batch) % 10 == 0:
            print(f"[insert] {min(i + args.insert_batch, len(corpus))}/{len(corpus)}")

    meta = {
        "collection": args.collection,
        "corpus_file": str(args.corpus_file),
        "indexed_docs": len(corpus),
        "dim": int(vecs.shape[1]),
        "doc_ids": doc_ids,
    }
    args.meta_file.parent.mkdir(parents=True, exist_ok=True)
    args.meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] meta saved -> {args.meta_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
