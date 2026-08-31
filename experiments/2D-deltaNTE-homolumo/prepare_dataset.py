#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import dataset as dataset_mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the GRPO dataset once and save it under 2D-nonsubprocess/data."
    )
    parser.add_argument("--dataset-name", default=dataset_mod.DEFAULT_DATASET_NAME)
    parser.add_argument("--data-split", default=dataset_mod.DEFAULT_DATA_SPLIT)
    parser.add_argument(
        "--local-data-dir",
        default=str(dataset_mod.DEFAULT_LOCAL_DATA_DIR),
        help="Directory for HuggingFace save_to_disk output.",
    )
    parser.add_argument("--proxy-host", default=dataset_mod.DEFAULT_PROXY_HOST)
    parser.add_argument("--proxy-port", type=int, default=dataset_mod.DEFAULT_PROXY_PORT)
    parser.add_argument(
        "--no-proxy-probe",
        dest="use_proxy_probe",
        action="store_false",
        help="Do not probe/set HTTP proxy before loading from HuggingFace.",
    )
    parser.add_argument(
        "--no-jsonl",
        dest="export_jsonl",
        action="store_false",
        help="Skip exporting split JSONL files.",
    )
    parser.set_defaults(use_proxy_probe=True, export_jsonl=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_data_dir = Path(args.local_data_dir)
    dataset, artifacts = dataset_mod.download_and_prepare_dataset(
        dataset_name=args.dataset_name,
        data_split=args.data_split,
        local_data_dir=local_data_dir,
        use_proxy_probe=args.use_proxy_probe,
        proxy_host=args.proxy_host,
        proxy_port=args.proxy_port,
        export_jsonl=args.export_jsonl,
    )

    raw_data = dataset_mod.load_local_dataset(
        local_data_dir=local_data_dir,
        data_split=args.data_split,
    )
    dataset_mod.validate_columns(raw_data)

    manifest = {
        "dataset_name": args.dataset_name,
        "data_split": args.data_split,
        "local_data_dir": str(local_data_dir),
        "splits": {
            split_name: len(dataset[split_name])
            for split_name in dataset_mod._split_names(dataset)
        },
        "artifacts": artifacts,
    }
    manifest_path = local_data_dir / "local_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved local dataset manifest to: {manifest_path}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
