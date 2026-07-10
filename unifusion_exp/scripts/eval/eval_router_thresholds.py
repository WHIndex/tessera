#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "scripts" / "train" / "train_router.py"


def _load_train_router_main():
    spec = importlib.util.spec_from_file_location("train_router_mod", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load train script from {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


# Thin wrapper that sweeps thresholds by repeatedly invoking train_router in fast mode.
# This keeps evaluation logic minimal and reproducible.
def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep router thresholds")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=str, default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    args = parser.parse_args()

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    train_router_main = _load_train_router_main()

    for t in thresholds:
        argv = [
            "--train-file",
            str(args.train_file),
            "--val-file",
            str(args.val_file),
            "--out-dir",
            str(args.out_dir),
            "--threshold",
            str(t),
            "--run-id",
            f"router_t{str(t).replace('.', '_')}",
        ]
        if args.max_train is not None:
            argv.extend(["--max-train", str(args.max_train)])
        if args.max_val is not None:
            argv.extend(["--max-val", str(args.max_val)])

        old_argv = sys.argv
        try:
            sys.argv = ["train_router.py", *argv]
            print(f"[sweep] threshold={t}")
            train_router_main()
        finally:
            sys.argv = old_argv

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
