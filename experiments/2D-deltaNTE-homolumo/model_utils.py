from __future__ import annotations

import json
import os
from getpass import getpass
from pathlib import Path


DEFAULT_LOCAL_TOKENIZER_ROOT = Path(__file__).resolve().parent / "tokenizers"


def local_tokenizer_dir_for_model(
    base_model: str,
    *,
    tokenizer_root: str | Path = DEFAULT_LOCAL_TOKENIZER_ROOT,
) -> Path:
    safe_name = base_model.split(":")[0].replace("/", "__")
    return Path(tokenizer_root) / safe_name


def has_local_tokenizer(local_tokenizer_dir: str | Path) -> bool:
    local_tokenizer_dir = Path(local_tokenizer_dir)
    marker_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
    ]
    return local_tokenizer_dir.is_dir() and any((local_tokenizer_dir / name).exists() for name in marker_files)


def prepare_local_tokenizer(
    base_model: str,
    *,
    local_tokenizer_dir: str | Path | None = None,
    allow_overwrite: bool = False,
) -> dict[str, str]:
    from tinker_cookbook import tokenizer_utils

    target_dir = Path(local_tokenizer_dir) if local_tokenizer_dir else local_tokenizer_dir_for_model(base_model)
    if has_local_tokenizer(target_dir) and not allow_overwrite:
        print(f"Local tokenizer already exists: {target_dir}")
        return {"base_model": base_model, "local_tokenizer_dir": str(target_dir), "saved": "false"}

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading/loading tokenizer for {base_model}...")
    tokenizer = tokenizer_utils.get_tokenizer(base_model)
    tokenizer.save_pretrained(str(target_dir))

    manifest = {
        "base_model": base_model,
        "local_tokenizer_dir": str(target_dir),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
    }
    manifest_path = target_dir / "local_tokenizer_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved local tokenizer to: {target_dir}")
    print(f"Saved tokenizer manifest to: {manifest_path}")
    return {"base_model": base_model, "local_tokenizer_dir": str(target_dir), "saved": "true"}


def ensure_tokens(prompt_for_missing: bool = True) -> None:
    if "TINKER_API_KEY" not in os.environ:
        if not prompt_for_missing:
            raise RuntimeError("TINKER_API_KEY is not set.")
        os.environ["TINKER_API_KEY"] = getpass("Enter your Tinker API key: ")

    if "HF_TOKEN" not in os.environ and prompt_for_missing:
        token = getpass("Enter your HuggingFace token (or press Enter to skip): ")
        if token:
            os.environ["HF_TOKEN"] = token


def create_service_client(prompt_for_missing_tokens: bool = True):
    import tinker

    ensure_tokens(prompt_for_missing=prompt_for_missing_tokens)
    service_client = tinker.ServiceClient()
    capabilities = service_client.get_server_capabilities()
    print("Connected to Tinker. Available Qwen3 models:")
    for model in capabilities.supported_models:
        if "Qwen3" in model.model_name:
            print(f"  - {model.model_name}")
    return service_client


def load_tokenizer_and_renderer(
    base_model: str,
    *,
    local_tokenizer_dir: str | Path | None = None,
    prefer_local_tokenizer: bool = True,
    cache_tokenizer: bool = True,
):
    from tinker_cookbook import renderers
    from tinker_cookbook import tokenizer_utils

    target_dir = Path(local_tokenizer_dir) if local_tokenizer_dir else local_tokenizer_dir_for_model(base_model)
    tokenizer_source = base_model

    if prefer_local_tokenizer and has_local_tokenizer(target_dir):
        tokenizer_source = str(target_dir)
        print(f"Loading tokenizer from local directory: {target_dir}")
    else:
        print(f"Loading tokenizer from model id: {base_model}")
        if cache_tokenizer:
            prepare_local_tokenizer(base_model, local_tokenizer_dir=target_dir, allow_overwrite=False)
            if has_local_tokenizer(target_dir):
                tokenizer_source = str(target_dir)
                print(f"Reloading tokenizer from local directory: {target_dir}")

    tokenizer = tokenizer_utils.get_tokenizer(tokenizer_source)
    renderer = renderers.get_renderer("qwen3", tokenizer)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Stop sequences: {renderer.get_stop_sequences()}")
    return tokenizer, renderer


def create_grpo_training_client(
    service_client,
    *,
    base_model: str,
    lora_rank: int,
    checkpoint: str,
):
    training_client = service_client.create_lora_training_client(
        base_model=base_model,
        rank=lora_rank,
    )
    training_client.load_state(checkpoint)
    print(f"Loaded GRPO initial checkpoint: {checkpoint}")
    return training_client
