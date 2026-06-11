#!/usr/bin/env python3
"""Generate averaged k-fold graphs for Qwen3 positional-wave runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams["pdf.fonttype"] = "truetype"

REPO_ROOT = Path(__file__).resolve().parent
GRAPHS_DIR = REPO_ROOT / "graphs_k_fold"
WAVES = ("sinusoid", "square", "triangular", "sawtooth")
WAVE_COLORS = {
    "sinusoid": "r",
    "square": "b",
    "triangular": "g",
    "sawtooth": "orange",
}
METRIC_DESCRIPTIONS = {
    "loss": ("Training loss", "lower", "Cross-entropy loss on training batches."),
    "eval_loss": ("Eval loss", "lower", "Cross-entropy loss on held-out validation samples."),
    "eval_accuracy": ("Eval accuracy", "higher", "Next-token accuracy on validation samples."),
    "eval_perplexity": ("Eval perplexity", "lower", "Exponentiated eval loss."),
    "perplexity": ("Final perplexity", "lower", "Final exponentiated eval loss."),
    "grad_norm": ("Gradient norm", "stable", "Gradient magnitude; spikes can indicate instability."),
    "learning_rate": ("Learning rate", "schedule", "Optimizer step size from the cosine schedule."),
    "eval_runtime": ("Eval runtime", "lower", "Seconds spent in evaluation."),
    "eval_samples_per_second": ("Eval samples per second", "higher", "Validation throughput."),
    "eval_steps_per_second": ("Eval steps per second", "higher", "Validation step throughput."),
    "train_loss": ("Final train loss", "lower", "Overall final training loss reported by Trainer."),
    "train_runtime": ("Train runtime", "lower", "Seconds spent in training."),
    "train_samples_per_second": ("Train samples per second", "higher", "Training throughput."),
    "train_steps_per_second": ("Train steps per second", "higher", "Training step throughput."),
}
FOLD_RE = re.compile(r"fold_(\d+)$")


@dataclass(frozen=True)
class FoldRun:
    wave: str
    fold: int
    path: Path
    complete: bool
    series: dict[str, dict[float, float]]
    final_metrics: dict[str, float]


@dataclass(frozen=True)
class MetricAggregate:
    metric: str
    steps: list[float]
    mean: list[float]
    std: list[float]
    counts: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot averaged Qwen3 k-fold metrics by RoPE waveform.")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="Directory containing qwen3_<wave>/fold_XX outputs. If omitted, the newest complete k-fold run is used.",
    )
    parser.add_argument("--results-dir", type=Path, default=None, help="Alias for --outputs-dir.")
    parser.add_argument("--run-id", default=None, help="K-fold run id, for example 2026-05-10_21-18-08.")
    parser.add_argument("--graphs-dir", type=Path, default=GRAPHS_DIR, help="Base directory for timestamped graph outputs.")
    parser.add_argument("--waves", default=",".join(WAVES), help="Comma-separated waves to include.")
    parser.add_argument("--k-folds", type=int, default=10, help="Expected number of folds per wave.")
    parser.add_argument("--allow-partial", action="store_true", help="Plot incomplete wave/fold sets instead of failing.")
    return parser.parse_args()


def numeric(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric_label(metric: str) -> str:
    return METRIC_DESCRIPTIONS.get(metric, (metric.replace("_", " ").title(), "", ""))[0]


def metric_direction(metric: str) -> str:
    return METRIC_DESCRIPTIONS.get(metric, ("", "higher", ""))[1]


def metric_description(metric: str) -> str:
    return METRIC_DESCRIPTIONS.get(metric, ("", "", "Numeric metric saved during training."))[2]


def parse_waves(raw: str) -> tuple[str, ...]:
    waves = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not waves:
        raise ValueError("At least one wave must be selected.")
    unknown = sorted(set(waves) - set(WAVES))
    if unknown:
        raise ValueError(f"Unknown waves: {', '.join(unknown)}")
    return waves


def candidate_roots() -> list[Path]:
    roots = [REPO_ROOT / "outputs" / "k_fold"]
    home = Path(os.environ.get("HOME", "")).expanduser()
    if str(home) != ".":
        roots.append(home / "fscratch" / "qwen3" / "outputs" / "k_fold")
    roots.append(Path("/mnt2/fscratch/users/tic_163_uma/macorisd/qwen3/outputs/k_fold"))
    roots.append(Path("/mnt/home/users/tic_163_uma/macorisd/fscratch/qwen3/outputs/k_fold"))
    unique: list[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved not in unique:
            unique.append(resolved)
    return unique


def is_run_dir(path: Path, waves: Iterable[str], k_folds: int) -> bool:
    for wave in waves:
        for fold in range(1, k_folds + 1):
            fold_dir = path / f"qwen3_{wave}" / f"fold_{fold:02d}"
            if not ((fold_dir / "trainer_state.json").is_file() and (fold_dir / "all_results.json").is_file()):
                return False
    return True


def discover_outputs_dir(run_id: str | None, waves: tuple[str, ...], k_folds: int) -> Path:
    candidates: list[Path] = []
    for root in candidate_roots():
        if not root.is_dir():
            continue
        if run_id:
            path = root / run_id
            if path.is_dir():
                candidates.append(path)
            continue
        candidates.extend(path for path in root.iterdir() if path.is_dir())

    complete = [path for path in candidates if is_run_dir(path, waves, k_folds)]
    if complete:
        return max(complete, key=lambda path: path.stat().st_mtime)
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    searched = ", ".join(str(root) for root in candidate_roots())
    raise FileNotFoundError(f"No Qwen3 k-fold output run found. Searched: {searched}")


def collect_series_from_history(log_history: list[dict[str, object]]) -> dict[str, dict[float, float]]:
    series: dict[str, dict[float, float]] = {}
    for row in log_history:
        step = numeric(row.get("step"))
        if step is None:
            continue
        for key, raw_value in row.items():
            if key in {"step", "epoch"}:
                continue
            value = numeric(raw_value)
            if value is None:
                continue
            series.setdefault(key, {})[step] = value
            if key == "eval_loss":
                series.setdefault("eval_perplexity", {})[step] = math.exp(value)
    return series


def collect_series_from_kfold(metrics: dict) -> dict[str, dict[float, float]]:
    series: dict[str, dict[float, float]] = {}
    history_specs = (
        ("train_loss_history", "loss"),
        ("validation_loss_history", "eval_loss"),
        ("accuracy_history", "eval_accuracy"),
    )
    for history_key, value_key in history_specs:
        rows = metrics.get(history_key, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            step = numeric(row.get("step"))
            value = numeric(row.get(value_key))
            if step is None or value is None:
                continue
            series.setdefault(value_key, {})[step] = value
            if value_key == "eval_loss":
                series.setdefault("eval_perplexity", {})[step] = math.exp(value)
    return series


def load_fold(fold_dir: Path, wave: str) -> FoldRun:
    match = FOLD_RE.match(fold_dir.name)
    if match is None:
        raise ValueError(f"Unexpected fold directory name: {fold_dir}")
    fold = int(match.group(1))
    kfold_metrics = read_json(fold_dir / "k_fold_metrics.json")
    trainer_state = read_json(fold_dir / "trainer_state.json")
    all_results = read_json(fold_dir / "all_results.json")
    train_results = read_json(fold_dir / "train_results.json")
    eval_results = read_json(fold_dir / "eval_results.json")

    series = collect_series_from_kfold(kfold_metrics)
    history = trainer_state.get("log_history", [])
    if isinstance(history, list):
        trainer_series = collect_series_from_history([row for row in history if isinstance(row, dict)])
        for key, values in trainer_series.items():
            series.setdefault(key, {}).update(values)

    final_metrics: dict[str, float] = {}
    for source in (train_results, eval_results, all_results, kfold_metrics):
        for key, value in source.items():
            parsed = numeric(value)
            if parsed is not None:
                final_metrics[key] = parsed
    complete = bool(kfold_metrics.get("complete")) if kfold_metrics else False
    complete = complete or all((fold_dir / name).is_file() for name in ("trainer_state.json", "all_results.json", "config.json"))
    return FoldRun(wave=wave, fold=fold, path=fold_dir, complete=complete, series=series, final_metrics=final_metrics)


def load_runs(outputs_dir: Path, waves: tuple[str, ...], k_folds: int, allow_partial: bool) -> dict[str, list[FoldRun]]:
    runs: dict[str, list[FoldRun]] = {}
    missing: list[str] = []
    incomplete: list[str] = []
    for wave in waves:
        wave_runs = []
        for fold in range(1, k_folds + 1):
            fold_dir = outputs_dir / f"qwen3_{wave}" / f"fold_{fold:02d}"
            if not fold_dir.is_dir():
                missing.append(f"{wave}/fold_{fold:02d}")
                continue
            run = load_fold(fold_dir, wave)
            if not run.complete:
                incomplete.append(f"{wave}/fold_{fold:02d}")
            wave_runs.append(run)
        runs[wave] = wave_runs

    if (missing or incomplete) and not allow_partial:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if incomplete:
            details.append("incomplete: " + ", ".join(incomplete))
        raise RuntimeError("Refusing to average a partial k-fold set (" + "; ".join(details) + ").")
    return runs


def aggregate_metric(folds: list[FoldRun], metric: str) -> MetricAggregate | None:
    step_sets = [set(run.series.get(metric, {})) for run in folds if run.series.get(metric)]
    if not step_sets:
        return None
    common_steps = sorted(set.intersection(*step_sets))
    if not common_steps:
        return None
    means: list[float] = []
    stds: list[float] = []
    counts: list[int] = []
    for step in common_steps:
        values = [run.series[metric][step] for run in folds if step in run.series.get(metric, {})]
        means.append(mean(values))
        stds.append(stdev(values) if len(values) > 1 else 0.0)
        counts.append(len(values))
    return MetricAggregate(metric=metric, steps=common_steps, mean=means, std=stds, counts=counts)


def aggregate_all(runs: dict[str, list[FoldRun]]) -> dict[str, dict[str, MetricAggregate]]:
    metrics = sorted({metric for folds in runs.values() for run in folds for metric in run.series})
    aggregated: dict[str, dict[str, MetricAggregate]] = {}
    for metric in metrics:
        per_wave: dict[str, MetricAggregate] = {}
        for wave, folds in runs.items():
            aggregate = aggregate_metric(folds, metric)
            if aggregate is not None:
                per_wave[wave] = aggregate
        if per_wave:
            aggregated[metric] = per_wave
    return aggregated


def final_scalar_stats(runs: dict[str, list[FoldRun]]) -> dict[str, dict[str, tuple[float, float, int]]]:
    metrics = sorted({metric for folds in runs.values() for run in folds for metric in run.final_metrics})
    result: dict[str, dict[str, tuple[float, float, int]]] = {}
    for metric in metrics:
        per_wave = {}
        for wave, folds in runs.items():
            values = [run.final_metrics[metric] for run in folds if metric in run.final_metrics]
            if values:
                per_wave[wave] = (mean(values), stdev(values) if len(values) > 1 else 0.0, len(values))
        if per_wave:
            result[metric] = per_wave
    return result


def create_graph_dir(graphs_dir: Path) -> Path:
    path = graphs_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_time_series_plot(metric: str, per_wave: dict[str, MetricAggregate], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    for wave in WAVES:
        aggregate = per_wave.get(wave)
        if aggregate is None:
            continue
        color = WAVE_COLORS[wave]
        ax.plot(aggregate.steps, aggregate.mean, label=wave, color=color, linewidth=2)
        if len(set(aggregate.counts)) == 1 and aggregate.counts[0] > 1:
            lower = [value - spread for value, spread in zip(aggregate.mean, aggregate.std)]
            upper = [value + spread for value, spread in zip(aggregate.mean, aggregate.std)]
            ax.fill_between(aggregate.steps, lower, upper, color=color, alpha=0.14, linewidth=0)
    ax.set_title(f"{metric_label(metric)} averaged across folds")
    ax.set_xlabel("Step")
    ax.set_ylabel(metric_label(metric))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / f"{metric}_mean_std_by_wave.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_loss_combo(aggregated: dict[str, dict[str, MetricAggregate]], output_dir: Path) -> Path | None:
    if "loss" not in aggregated and "eval_loss" not in aggregated:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False)
    for ax, metric in zip(axes, ("loss", "eval_loss")):
        per_wave = aggregated.get(metric, {})
        for wave in WAVES:
            aggregate = per_wave.get(wave)
            if aggregate is None:
                continue
            color = WAVE_COLORS[wave]
            ax.plot(aggregate.steps, aggregate.mean, label=wave, color=color, linewidth=2)
            lower = [value - spread for value, spread in zip(aggregate.mean, aggregate.std)]
            upper = [value + spread for value, spread in zip(aggregate.mean, aggregate.std)]
            ax.fill_between(aggregate.steps, lower, upper, color=color, alpha=0.14, linewidth=0)
        ax.set_title(metric_label(metric))
        ax.set_xlabel("Step")
        ax.set_ylabel(metric_label(metric))
        ax.grid(True, alpha=0.25)
    axes[0].legend()
    fig.suptitle("Qwen3 k-fold loss comparison")
    fig.tight_layout()
    output_path = output_dir / "loss_eval_loss_mean_std_by_wave.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_scalar_bar(metric: str, per_wave: dict[str, tuple[float, float, int]], output_dir: Path) -> Path:
    waves = [wave for wave in WAVES if wave in per_wave]
    means = [per_wave[wave][0] for wave in waves]
    stds = [per_wave[wave][1] for wave in waves]
    colors = [WAVE_COLORS[wave] for wave in waves]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(waves, means, yerr=stds, color=colors, alpha=0.82, capsize=5)
    ax.set_title(f"{metric_label(metric)} final mean across folds")
    ax.set_ylabel(metric_label(metric))
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    output_path = output_dir / f"final_{metric}_mean_std_by_wave.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_summary_csv(runs: dict[str, list[FoldRun]], scalar_stats: dict[str, dict[str, tuple[float, float, int]]], output_dir: Path) -> Path:
    metrics = sorted(scalar_stats)
    path = output_dir / "summary.csv"
    fieldnames = ["wave", "folds_loaded"] + [f"{metric}_mean" for metric in metrics] + [f"{metric}_std" for metric in metrics]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for wave in WAVES:
            if wave not in runs:
                continue
            row: dict[str, object] = {"wave": wave, "folds_loaded": len(runs[wave])}
            for metric in metrics:
                values = scalar_stats[metric].get(wave)
                if values is None:
                    row[f"{metric}_mean"] = ""
                    row[f"{metric}_std"] = ""
                else:
                    row[f"{metric}_mean"] = f"{values[0]:.10g}"
                    row[f"{metric}_std"] = f"{values[1]:.10g}"
            writer.writerow(row)
    return path


def rank_values(values: list[tuple[str, float]], direction: str) -> list[tuple[str, float]]:
    if direction == "lower":
        return sorted(values, key=lambda item: item[1])
    if direction == "higher":
        return sorted(values, key=lambda item: item[1], reverse=True)
    return sorted(values, key=lambda item: item[0])


def fmt(value: float | None) -> str:
    if value is None:
        return "missing"
    if abs(value) >= 10000 or (0 < abs(value) < 0.001):
        return f"{value:.4e}"
    return f"{value:.6g}"


def segment_rows(aggregate: MetricAggregate, segments: int = 5) -> list[tuple[int, float, float]]:
    if not aggregate.steps:
        return []
    rows = []
    total = len(aggregate.steps)
    effective_segments = min(segments, total)
    for segment in range(effective_segments):
        start = round(total * segment / segments)
        end = round(total * (segment + 1) / segments)
        if end <= start:
            end = min(total, start + 1)
        values = aggregate.mean[start:end]
        spreads = aggregate.std[start:end]
        if values:
            rows.append((segment + 1, mean(values), mean(spreads) if spreads else 0.0))
    return rows


def write_results_explanation(
    output_dir: Path,
    outputs_dir: Path,
    runs: dict[str, list[FoldRun]],
    aggregated: dict[str, dict[str, MetricAggregate]],
    scalar_stats: dict[str, dict[str, tuple[float, float, int]]],
    plots: list[Path],
) -> Path:
    lines = [
        "# Qwen3 K-Fold Results Explanation",
        "",
        f"Source run: `{outputs_dir}`",
        f"Loaded folds: " + ", ".join(f"{wave}={len(folds)}" for wave, folds in runs.items()),
        "",
        "The script reads `k_fold_metrics.json` where available, merges the same numeric time series from `trainer_state.json`, aligns folds on common step values, and plots the per-wave mean with standard deviation bands. Final scalar summaries are averaged across folds from `all_results.json`, `train_results.json`, `eval_results.json`, and `k_fold_metrics.json`.",
        "",
        "Generated plots:",
        "",
    ]
    for plot in plots:
        lines.append(f"- `{plot.name}`")
    lines.append("")

    important_scalars = [metric for metric in ("train_loss", "eval_loss", "eval_accuracy", "perplexity") if metric in scalar_stats]
    if important_scalars:
        lines.extend(["## Final Scalar Ranking", ""])
    for metric in important_scalars:
        direction = metric_direction(metric)
        lines.extend(
            [
                f"### {metric_label(metric)}",
                "",
                f"Direction: `{direction}`. {metric_description(metric)}",
                "",
                "| rank | wave | mean | std | folds |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        values = [(wave, stats[0]) for wave, stats in scalar_stats[metric].items()]
        for index, (wave, value) in enumerate(rank_values(values, direction), start=1):
            mean_value, std_value, count = scalar_stats[metric][wave]
            lines.append(f"| {index} | {wave} | {fmt(mean_value)} | {fmt(std_value)} | {count} |")
        lines.append("")

    for metric in sorted(aggregated):
        per_wave = aggregated[metric]
        direction = metric_direction(metric)
        lines.extend(
            [
                f"## {metric_label(metric)}",
                "",
                f"Direction: `{direction}`. {metric_description(metric)}",
                "",
                "Final aligned point ranking:",
                "",
                "| rank | wave | final mean | final std | folds at final point |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        finals = [(wave, aggregate.mean[-1]) for wave, aggregate in per_wave.items()]
        for index, (wave, _) in enumerate(rank_values(finals, direction), start=1):
            aggregate = per_wave[wave]
            lines.append(f"| {index} | {wave} | {fmt(aggregate.mean[-1])} | {fmt(aggregate.std[-1])} | {aggregate.counts[-1]} |")
        lines.extend(["", "Five progress segments:", "", "| segment | wave | mean | std |", "|---:|---|---:|---:|"])
        for wave in WAVES:
            aggregate = per_wave.get(wave)
            if aggregate is None:
                continue
            for segment, segment_mean, segment_std in segment_rows(aggregate):
                lines.append(f"| {segment} | {wave} | {fmt(segment_mean)} | {fmt(segment_std)} |")
        lines.append("")

    path = output_dir / "results_explanation.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    waves = parse_waves(args.waves)
    outputs_dir = (args.outputs_dir or args.results_dir)
    if outputs_dir is None:
        outputs_dir = discover_outputs_dir(args.run_id, waves, args.k_folds)
    else:
        outputs_dir = outputs_dir.resolve()
        if args.run_id and not outputs_dir.name == args.run_id and (outputs_dir / args.run_id).is_dir():
            outputs_dir = outputs_dir / args.run_id

    runs = load_runs(outputs_dir, waves, args.k_folds, args.allow_partial)
    aggregated = aggregate_all(runs)
    scalar_stats = final_scalar_stats(runs)
    graph_dir = create_graph_dir(args.graphs_dir)

    plots: list[Path] = []
    loss_plot = save_loss_combo(aggregated, graph_dir)
    if loss_plot is not None:
        plots.append(loss_plot)
    for metric, per_wave in sorted(aggregated.items()):
        plots.append(save_time_series_plot(metric, per_wave, graph_dir))
    for metric, per_wave in sorted(scalar_stats.items()):
        if metric in {"total_flos", "train_samples", "eval_samples", "epoch", "seed", "split_seed", "fold", "k_folds"}:
            continue
        plots.append(save_scalar_bar(metric, per_wave, graph_dir))

    summary_csv = write_summary_csv(runs, scalar_stats, graph_dir)
    explanation = write_results_explanation(graph_dir, outputs_dir, runs, aggregated, scalar_stats, plots)

    print(f"Source outputs: {outputs_dir}")
    print(f"Graph directory: {graph_dir}")
    print(f"Plots: {len(plots)}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Results explanation: {explanation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
