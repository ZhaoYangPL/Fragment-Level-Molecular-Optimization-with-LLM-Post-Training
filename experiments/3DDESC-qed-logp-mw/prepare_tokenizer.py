#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_utils import local_tokenizer_dir_for_model
from model_utils import prepare_local_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a model tokenizer once and save it under 2D-nonsubprocess/tokenizers."
    )
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--local-tokenizer-dir",
        help="Tokenizer output directory. Defaults to tokenizers/<base-model-with-__>.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Re-save tokenizer files even if the local tokenizer directory already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_dir = (
        Path(args.local_tokenizer_dir)
        if args.local_tokenizer_dir
        else local_tokenizer_dir_for_model(args.base_model)
    )
    artifacts = prepare_local_tokenizer(
        args.base_model,
        local_tokenizer_dir=local_dir,
        allow_overwrite=args.allow_overwrite,
    )
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
