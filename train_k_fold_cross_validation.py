#!/usr/bin/env python
"""Run deterministic k-fold Qwen3 wave experiments through run_clm.py."""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean


WAVES = ("sinusoid", "triangular", "square", "sawtooth")
SCRIPT_DIR = Path(__file__).resolve().parent
RUN_CLM = SCRIPT_DIR / "examples" / "pytorch" / "language-modeling" / "run_clm.py"
COMPLETION_ARTIFACTS = ("config.json", "trainer_state.json", "all_results.json")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run Qwen3 k-fold cross validation for one or more RoPE waveforms.",
        allow_abbrev=False,
    )
    parser.add_argument("--resume", action="store_true", help="Skip complete folds and rerun incomplete fold dirs.")
    parser.add_argument("--functions", default=",".join(WAVES), help="Comma-separated waveforms to train.")
    parser.add_argument("--folds", default=None, help="Comma-separated 1-based folds to run. Defaults to all folds.")
    parser.add_argument("--k-folds", type=int, default=10, help="Total number of deterministic folds.")
    parser.add_argument("--split-seed", type=int, default=_env_int("K_FOLD_SPLIT_SEED", 42))
    parser.add_argument("--seed", type=int, default=_env_int("GLOBAL_SEED", 89), help="Training seed for every run.")
    parser.add_argument("--run-id", default=os.environ.get("KFOLD_RUN_ID") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    parser.add_argument(
        "--output-base-dir",
        default=os.environ.get("QWEN3_KFOLD_OUTPUT_BASE_DIR") or str(SCRIPT_DIR / "outputs" / "k_fold"),
    )
    parser.add_argument(
        "--indices-dir",
        default=os.environ.get("QWEN3_KFOLD_INDICES_DIR") or None,
        help="Where fold index JSON files are written.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print fold commands without running training.")
    args, run_clm_args = parser.parse_known_args()
    if run_clm_args and run_clm_args[0] == "--":
        run_clm_args = run_clm_args[1:]
    return args, run_clm_args


def split_csv(value: str | None, default: tuple[str, ...] | None = None) -> list[str]:
    if value is None:
        return list(default or ())
    return [item.strip() for item in value.split(",") if item.strip()]


def get_arg_value(args: list[str], *names: str) -> str | None:
    for index, arg in enumerate(args):
        for name in names:
            if arg == name and index + 1 < len(args):
                return args[index + 1]
            if arg.startswith(f"{name}="):
                return arg.split("=", 1)[1]
    return None


def has_flag(args: list[str], *names: str) -> bool:
    return any(arg in names for arg in args)


def strip_args(args: list[str], value_names: set[str], flag_names: set[str]) -> list[str]:
    cleaned = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        name = arg.split("=", 1)[0]
        if name in value_names:
            if "=" not in arg:
                skip_next = True
            continue
        if name in flag_names:
            continue
        cleaned.append(arg)
    return cleaned


def slugify(value: str | None) -> str:
    if not value:
        return "unspecified-dataset"
    value = Path(value).name
    if "." in value:
        value = Path(value).stem
    safe = "".join(char if char.isalnum() or char in "._-" else "-" for char in value).strip("-")
    return safe or "unknown"


def dataset_slug(run_clm_args: list[str]) -> str:
    dataset_name = get_arg_value(run_clm_args, "--dataset_name", "--dataset-name")
    dataset_config = get_arg_value(run_clm_args, "--dataset_config_name", "--dataset-config-name")
    dataset_dir = get_arg_value(run_clm_args, "--dataset_dir", "--dataset-dir")
    train_file = get_arg_value(run_clm_args, "--train_file", "--train-file")
    validation_file = get_arg_value(run_clm_args, "--validation_file", "--validation-file")
    if dataset_name and dataset_config:
        return slugify(f"{dataset_name}-{dataset_config}")
    if dataset_name:
        return slugify(dataset_name)
    if dataset_dir:
        return slugify(dataset_dir)
    if train_file:
        return slugify(train_file)
    if validation_file:
        return slugify(validation_file)
    return "unspecified-dataset"


def load_base_train_dataset(run_clm_args: list[str]):
    from datasets import DatasetDict, IterableDataset, IterableDatasetDict, load_dataset, load_from_disk

    if has_flag(run_clm_args, "--streaming"):
        raise ValueError("K-fold training needs random access; remove --streaming.")

    cache_dir = get_arg_value(run_clm_args, "--cache_dir", "--cache-dir")
    token = get_arg_value(run_clm_args, "--token")
    dataset_dir = get_arg_value(run_clm_args, "--dataset_dir", "--dataset-dir")
    dataset_name = get_arg_value(run_clm_args, "--dataset_name", "--dataset-name")
    dataset_config = get_arg_value(run_clm_args, "--dataset_config_name", "--dataset-config-name")
    train_file = get_arg_value(run_clm_args, "--train_file", "--train-file")
    validation_file = get_arg_value(run_clm_args, "--validation_file", "--validation-file")

    if dataset_dir:
        raw = load_from_disk(dataset_dir)
    elif dataset_name:
        raw = load_dataset(dataset_name, dataset_config, cache_dir=cache_dir, token=token)
    elif train_file or validation_file:
        data_files = {}
        if train_file:
            data_files["train"] = train_file
        if validation_file:
            data_files["validation"] = validation_file
        source_file = train_file or validation_file
        extension = Path(source_file).suffix.lstrip(".")
        dataset_args = {}
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not has_flag(run_clm_args, "--no_keep_linebreaks")
        raw = load_dataset(extension, data_files=data_files, cache_dir=cache_dir, token=token, **dataset_args)
    else:
        raise ValueError("Pass --dataset_dir, --dataset_name, or --train_file for k-fold training.")

    if isinstance(raw, (IterableDataset, IterableDatasetDict)):
        raise ValueError("K-fold training needs a non-streaming dataset.")
    if isinstance(raw, DatasetDict):
        if "train" not in raw:
            raise ValueError("The base dataset must contain a train split.")
        return raw["train"]
    return raw


def fold_specs(num_examples: int, k_folds: int, split_seed: int) -> list[dict[str, object]]:
    if k_folds < 2:
        raise ValueError("--k-folds must be at least 2.")
    if num_examples < k_folds:
        raise ValueError(f"Cannot create {k_folds} folds from {num_examples} examples.")
    indices = list(range(num_examples))
    random.Random(split_seed).shuffle(indices)
    fold_size = num_examples // k_folds
    specs = []
    for fold_index in range(k_folds):
        start = fold_index * fold_size
        end = num_examples if fold_index == k_folds - 1 else (fold_index + 1) * fold_size
        validation_indices = indices[start:end]
        train_indices = indices[:start] + indices[end:]
        specs.append(
            {
                "fold": fold_index + 1,
                "k_folds": k_folds,
                "split_seed": split_seed,
                "num_examples": num_examples,
                "train_indices": train_indices,
                "validation_indices": validation_indices,
            }
        )
    return specs


def write_fold_indices(specs: list[dict[str, object]], indices_dir: Path) -> dict[int, Path]:
    indices_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for spec in specs:
        fold = int(spec["fold"])
        path = indices_dir / f"fold_{fold:02d}.json"
        path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        paths[fold] = path
    return paths


def is_complete(output_dir: Path) -> bool:
    return all((output_dir / artifact).exists() for artifact in COMPLETION_ARTIFACTS)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def extract_histories(output_dir: Path) -> dict[str, object]:
    trainer_state = read_json(output_dir / "trainer_state.json")
    log_history = trainer_state.get("log_history", [])
    train_loss_history = [
        {"step": row.get("step"), "loss": row["loss"]} for row in log_history if "loss" in row and "eval_loss" not in row
    ]
    eval_loss_history = [{"step": row.get("step"), "eval_loss": row["eval_loss"]} for row in log_history if "eval_loss" in row]
    accuracy_history = [
        {"step": row.get("step"), "eval_accuracy": row["eval_accuracy"]} for row in log_history if "eval_accuracy" in row
    ]
    train_results = read_json(output_dir / "train_results.json")
    eval_results = read_json(output_dir / "eval_results.json")
    all_results = read_json(output_dir / "all_results.json")
    return {
        "train_loss_history": train_loss_history,
        "validation_loss_history": eval_loss_history,
        "accuracy_history": accuracy_history,
        "train_results": train_results,
        "eval_results": eval_results,
        "all_results": all_results,
    }


def write_fold_summary(output_dir: Path, wave: str, fold: int, k_folds: int, split_seed: int, seed: int) -> dict[str, object]:
    histories = extract_histories(output_dir)
    all_results = histories["all_results"]
    summary = {
        "wave": wave,
        "fold": fold,
        "k_folds": k_folds,
        "split_seed": split_seed,
        "seed": seed,
        "output_dir": str(output_dir),
        "complete": is_complete(output_dir),
        "train_loss": all_results.get("train_loss"),
        "eval_loss": all_results.get("eval_loss"),
        "eval_accuracy": all_results.get("eval_accuracy"),
        "perplexity": all_results.get("perplexity"),
        **histories,
    }
    (output_dir / "k_fold_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def write_summaries(output_base: Path, run_id: str, summaries: list[dict[str, object]]) -> None:
    by_wave: dict[str, list[dict[str, object]]] = {}
    for summary in summaries:
        by_wave.setdefault(str(summary["wave"]), []).append(summary)

    global_rows = []
    for wave, rows in by_wave.items():
        eval_losses = [row["eval_loss"] for row in rows if isinstance(row.get("eval_loss"), (int, float))]
        perplexities = [row["perplexity"] for row in rows if isinstance(row.get("perplexity"), (int, float))]
        accuracies = [row["eval_accuracy"] for row in rows if isinstance(row.get("eval_accuracy"), (int, float))]
        wave_summary = {
            "wave": wave,
            "run_id": run_id,
            "folds": rows,
            "mean_eval_loss": mean(eval_losses) if eval_losses else None,
            "mean_perplexity": mean(perplexities) if perplexities else None,
            "mean_eval_accuracy": mean(accuracies) if accuracies else None,
        }
        wave_dir = output_base / f"qwen3_{wave}"
        wave_dir.mkdir(parents=True, exist_ok=True)
        (wave_dir / "k_fold_summary.json").write_text(json.dumps(wave_summary, indent=2) + "\n", encoding="utf-8")
        global_rows.append({key: value for key, value in wave_summary.items() if key != "folds"})

    global_summary = {"run_id": run_id, "waves": global_rows}
    (output_base / "k_fold_global_summary.json").write_text(json.dumps(global_summary, indent=2) + "\n", encoding="utf-8")


def command_for(run_clm_args: list[str], wave: str, fold: int, seed: int) -> list[str]:
    value_names = {
        "--rope_waveform",
        "--rope-waveform",
        "--seed",
        "--data_seed",
        "--data-seed",
        "--output_dir",
        "--output-dir",
        "--run_name",
        "--run-name",
    }
    cleaned = strip_args(run_clm_args, value_names=value_names, flag_names=set())
    return [
        sys.executable,
        str(RUN_CLM),
        "--rope_waveform",
        wave,
        "--seed",
        str(seed),
        "--data_seed",
        str(seed),
        "--run_name",
        f"qwen3_{wave}_fold{fold:02d}",
        *cleaned,
    ]


def main() -> int:
    args, run_clm_args = parse_args()
    waves = split_csv(args.functions, WAVES)
    unknown_waves = sorted(set(waves) - set(WAVES))
    if unknown_waves:
        raise ValueError(f"Unknown waveforms: {', '.join(unknown_waves)}")
    folds = [int(item) for item in split_csv(args.folds)] if args.folds else list(range(1, args.k_folds + 1))
    bad_folds = [fold for fold in folds if fold < 1 or fold > args.k_folds]
    if bad_folds:
        raise ValueError(f"Fold numbers must be between 1 and {args.k_folds}: {bad_folds}")

    output_base = Path(args.output_base_dir).resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    slug = dataset_slug(run_clm_args)
    indices_dir = Path(args.indices_dir).resolve() if args.indices_dir else output_base / "_fold_indices" / f"{args.run_id}_{slug}_k{args.k_folds}_seed{args.split_seed}"

    fold_index_paths = {}
    if not args.dry_run:
        base_train = load_base_train_dataset(run_clm_args)
        specs = fold_specs(len(base_train), args.k_folds, args.split_seed)
        fold_index_paths = write_fold_indices(specs, indices_dir)
        print(f"Wrote deterministic fold indices to {indices_dir}", flush=True)
    else:
        indices_dir.mkdir(parents=True, exist_ok=True)
        for fold in folds:
            fold_index_paths[fold] = indices_dir / f"fold_{fold:02d}.json"

    completed_summaries = []
    for wave in waves:
        for fold in folds:
            output_dir = output_base / f"qwen3_{wave}" / f"fold_{fold:02d}"
            cmd = command_for(run_clm_args, wave, fold, args.seed)
            env = os.environ.copy()
            env["QWEN3_KFOLD_INDICES_FILE"] = str(fold_index_paths[fold])
            env["QWEN3_OUTPUT_DIR"] = str(output_dir)
            env["QWEN3_KFOLD_RUN_ID"] = args.run_id
            env["QWEN3_KFOLD_FOLD"] = str(fold)
            env["QWEN3_KFOLD_WAVE"] = wave

            if args.resume and is_complete(output_dir):
                print(f"Skipping complete fold: wave={wave} fold={fold} output={output_dir}", flush=True)
                completed_summaries.append(
                    write_fold_summary(output_dir, wave, fold, args.k_folds, args.split_seed, args.seed)
                )
                continue
            if output_dir.exists() and not is_complete(output_dir):
                print(f"Removing incomplete fold output: {output_dir}", flush=True)
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            printable = " ".join(cmd)
            print(f"Running wave={wave} fold={fold}/{args.k_folds}: {printable}", flush=True)
            if args.dry_run:
                continue
            subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)
            completed_summaries.append(write_fold_summary(output_dir, wave, fold, args.k_folds, args.split_seed, args.seed))
            write_summaries(output_base, args.run_id, completed_summaries)

    if completed_summaries:
        write_summaries(output_base, args.run_id, completed_summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
