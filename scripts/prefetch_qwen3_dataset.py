#!/usr/bin/env python
"""Download and materialize the exact Qwen3 training dataset for offline jobs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


DEFAULT_DATASET_NAME = "Salesforce/wikitext"
DEFAULT_DATASET_CONFIG_NAME = "wikitext-103-raw-v1"


def default_fscratch_base() -> Path:
    home = Path.home()
    fscratch = home / "fscratch"
    if fscratch.exists():
        return fscratch.resolve() / "qwen3"
    return Path(__file__).resolve().parents[1] / ".local_data"


def dataset_slug(dataset_name: str, dataset_config_name: str | None) -> str:
    parts = [dataset_name.replace("/", "_")]
    if dataset_config_name:
        parts.append(dataset_config_name)
    return "__".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config-name", default=DEFAULT_DATASET_CONFIG_NAME)
    parser.add_argument("--fscratch-base", type=Path, default=default_fscratch_base())
    parser.add_argument("--dataset-dir", type=Path, default=None)
    args = parser.parse_args()

    fscratch_base = args.fscratch_base.expanduser().resolve()
    hf_home = Path(os.environ.get("HF_HOME", fscratch_base / "hf_cache")).expanduser().resolve()
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))

    for key in ("HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_HUB_CACHE"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        dataset_dir = fscratch_base / "datasets" / dataset_slug(args.dataset_name, args.dataset_config_name)
    dataset_dir = dataset_dir.expanduser().resolve()
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset_name} / {args.dataset_config_name}")
    print(f"HF_HOME: {os.environ['HF_HOME']}")
    print(f"Saving DatasetDict to: {dataset_dir}")

    raw_datasets = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=os.environ["HF_HOME"],
    )
    raw_datasets.save_to_disk(str(dataset_dir))

    manifest = {
        "dataset_name": args.dataset_name,
        "dataset_config_name": args.dataset_config_name,
        "dataset_dir": str(dataset_dir),
        "splits": {
            split: {
                "num_rows": len(dataset),
                "columns": list(dataset.column_names),
            }
            for split, dataset in raw_datasets.items()
        },
    }
    manifest_path = dataset_dir / "qwen3_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
