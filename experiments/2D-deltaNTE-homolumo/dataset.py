from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

DEFAULT_DATASET_NAME = "Alan123/homo-lumo-grpo"
DEFAULT_DATA_SPLIT = "train"
DEFAULT_PROMPT_COL = "input"
DEFAULT_ANSWER_COL = "output"
DEFAULT_PROXY_HOST = "172.18.48.1"
DEFAULT_PROXY_PORT = 7897
DEFAULT_LOCAL_DATA_DIR = Path(__file__).resolve().parent / "data" / "homo-lumo-grpo"


def test_port(ip: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        return sock.connect_ex((ip, port)) == 0


def configure_proxy(proxy_host: str = DEFAULT_PROXY_HOST, proxy_port: int = DEFAULT_PROXY_PORT) -> str | None:
    proxy_url = f"http://{proxy_host}:{proxy_port}"
    if test_port(proxy_host, proxy_port):
        print(f"Found proxy service: {proxy_url}")
        os.environ["http_proxy"] = proxy_url
        os.environ["https_proxy"] = proxy_url
        return proxy_url

    print(f"Proxy probe failed for {proxy_host}:{proxy_port}; continuing without changing proxy env.")
    return None


def has_local_dataset(local_data_dir: str | Path = DEFAULT_LOCAL_DATA_DIR) -> bool:
    local_data_dir = Path(local_data_dir)
    return (local_data_dir / "dataset_dict.json").exists() or (local_data_dir / "state.json").exists()


def _split_names(dataset) -> list[str]:
    if hasattr(dataset, "keys"):
        return list(dataset.keys())
    return [DEFAULT_DATA_SPLIT]


def _get_split(dataset, split_name: str):
    if hasattr(dataset, "keys") and split_name in dataset:
        return dataset[split_name]
    if split_name == DEFAULT_DATA_SPLIT:
        return dataset
    raise KeyError(f"Split {split_name!r} not found in dataset.")


def export_dataset_jsonl(dataset, *, local_data_dir: str | Path = DEFAULT_LOCAL_DATA_DIR) -> dict[str, str]:
    local_data_dir = Path(local_data_dir)
    jsonl_dir = local_data_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}
    for split_name in _split_names(dataset):
        split = _get_split(dataset, split_name)
        out_path = jsonl_dir / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for row in split:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        artifacts[split_name] = str(out_path)
        print(f"Exported {split_name} JSONL to: {out_path}")
    return artifacts


def save_dataset_locally(
    dataset,
    *,
    local_data_dir: str | Path = DEFAULT_LOCAL_DATA_DIR,
    export_jsonl: bool = True,
) -> dict[str, Any]:
    local_data_dir = Path(local_data_dir)
    if has_local_dataset(local_data_dir):
        print(f"Local dataset already exists, leaving it unchanged: {local_data_dir}")
        artifacts: dict[str, Any] = {"dataset_dir": str(local_data_dir), "saved": False}
        if export_jsonl:
            artifacts["jsonl"] = export_dataset_jsonl(dataset, local_data_dir=local_data_dir)
        return artifacts

    local_data_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(local_data_dir))
    print(f"Saved HuggingFace dataset to local disk: {local_data_dir}")

    artifacts = {"dataset_dir": str(local_data_dir), "saved": True}
    if export_jsonl:
        artifacts["jsonl"] = export_dataset_jsonl(dataset, local_data_dir=local_data_dir)
    return artifacts


def load_local_dataset(
    *,
    local_data_dir: str | Path = DEFAULT_LOCAL_DATA_DIR,
    data_split: str = DEFAULT_DATA_SPLIT,
):
    from datasets import load_from_disk

    local_data_dir = Path(local_data_dir)
    if not has_local_dataset(local_data_dir):
        raise FileNotFoundError(f"No saved HuggingFace dataset found at: {local_data_dir}")

    dataset = load_from_disk(str(local_data_dir))
    raw_data = _get_split(dataset, data_split)
    print(f"Loaded local dataset from: {local_data_dir}")
    print(raw_data)
    print(f"Loaded {len(raw_data)} local training examples")
    return raw_data


def download_and_prepare_dataset(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    data_split: str = DEFAULT_DATA_SPLIT,
    local_data_dir: str | Path = DEFAULT_LOCAL_DATA_DIR,
    use_proxy_probe: bool = True,
    proxy_host: str = DEFAULT_PROXY_HOST,
    proxy_port: int = DEFAULT_PROXY_PORT,
    export_jsonl: bool = True,
):
    from datasets import load_dataset

    if use_proxy_probe:
        configure_proxy(proxy_host=proxy_host, proxy_port=proxy_port)

    print("Downloading dataset from HuggingFace.")
    dataset = load_dataset(dataset_name)
    artifacts = save_dataset_locally(
        dataset,
        local_data_dir=local_data_dir,
        export_jsonl=export_jsonl,
    )

    raw_data = _get_split(dataset, data_split)
    print(f"Prepared split {data_split!r}: {len(raw_data)} examples")
    return dataset, artifacts


def load_raw_data(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    data_split: str = DEFAULT_DATA_SPLIT,
    local_data_dir: str | Path = DEFAULT_LOCAL_DATA_DIR,
    prefer_local: bool = True,
    cache_local: bool = True,
    export_jsonl: bool = True,
    use_proxy_probe: bool = True,
    proxy_host: str = DEFAULT_PROXY_HOST,
    proxy_port: int = DEFAULT_PROXY_PORT,
) -> Any:
    from datasets import load_dataset

    if prefer_local and has_local_dataset(local_data_dir):
        return load_local_dataset(local_data_dir=local_data_dir, data_split=data_split)

    if use_proxy_probe:
        configure_proxy(proxy_host=proxy_host, proxy_port=proxy_port)

    print("Using original prompt/ground-truth dataset directly (no four-step synthesis).")
    dataset = load_dataset(dataset_name)
    if cache_local:
        save_dataset_locally(dataset, local_data_dir=local_data_dir, export_jsonl=export_jsonl)

    if hasattr(dataset, "keys") and "train" in dataset:
        train_split = dataset["train"]
    else:
        train_split = dataset

    if hasattr(train_split, "select"):
        subset = train_split.select(range(min(10, len(train_split))))
        print(f"Debug subset size: {len(subset)}")

    print(f"Loaded train split size: {len(train_split)}")

    if hasattr(dataset, "keys") and data_split in dataset:
        raw_data = dataset[data_split]
    elif data_split == "train":
        raw_data = train_split
    else:
        raise KeyError(f"Split {data_split!r} not found in dataset {dataset_name!r}.")

    print(raw_data)
    print(f"Loaded {len(raw_data)} training examples")
    return raw_data


def validate_columns(raw_data: Any, prompt_col: str = DEFAULT_PROMPT_COL, answer_col: str = DEFAULT_ANSWER_COL) -> None:
    column_names = getattr(raw_data, "column_names", None)
    if column_names is None:
        sample = raw_data[0]
        column_names = list(sample.keys()) if hasattr(sample, "keys") else []

    missing = [name for name in (prompt_col, answer_col) if name not in column_names]
    if missing:
        raise KeyError(
            f"Missing required dataset columns {missing}. Available columns: {list(column_names)}"
        )
