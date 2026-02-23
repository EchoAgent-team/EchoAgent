#!/usr/bin/env python3
"""
Simple runner for batch embedding generation from MXM + Last.fm.

Usage:
  python3 tests/test_embeddings.py --limit 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run MXM + Last.fm batch embedding generation")
    parser.add_argument(
        "--train",
        default=str(repo_root / "data" / "bow" / "mxm_dataset_train.txt"),
        help="Path to MXM train file",
    )
    parser.add_argument(
        "--test",
        default=str(repo_root / "data" / "bow" / "mxm_dataset_test.txt"),
        help="Path to MXM test file",
    )
    parser.add_argument(
        "--lastfm",
        default=str(repo_root / "data" / "msd_lastfm_map.cls"),
        help="Path to Last.fm map (.cls or .zip)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of embedded tracks (useful for a quick test)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print progress every N embedded tracks",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from backend.data.embeddings import upsert_all_from_mxm_and_lastfm
    except ModuleNotFoundError as exc:
        print(f"Import error: {exc}")
        print("Install required packages first (e.g. chromadb, sentence-transformers, numpy).")
        return 1

    try:
        stats = upsert_all_from_mxm_and_lastfm(
            mxm_train_path=args.train,
            mxm_test_path=args.test,
            lastfm_map_path=args.lastfm,
            progress_every=args.progress_every,
            limit=args.limit,
        )
    except Exception as exc:  # Keep this script easy to use/debug.
        print(f"Batch embedding run failed: {exc}")
        return 1

    print("Batch embedding run completed.")
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
