#!/usr/bin/env python3
"""Generate graphs for simple Qwen3 positional-wave training runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams["pdf.fonttype"] = "truetype"

REPO_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
GRAPHS_DIR = REPO_ROOT / "graphs"
JOBS_DIR = REPO_ROOT.parent / "jobs"
WAVES = ("sinusoid", "square", "triangular", "sawtooth")
WAVE_COLORS = {
    "sinusoid": "r",
    "square": "b",
    "triangular": "g",
    "sawtooth": "orange",
}
METRIC_DESCRIPTIONS = {
    "loss": ("Training loss", "lower", "Cross-entropy loss on training batches; lower usually means better fit."),
    "eval_loss": ("Eval loss", "lower", "Cross-entropy loss on validation samples; lower usually means better generalization."),
    "eval_accuracy": ("Eval accuracy", "higher", "Next-token accuracy on validation samples; higher is better."),
    "eval_perplexity": ("Eval perplexity", "lower", "Exponentiated eval loss; lower means the model is less surprised by validation text."),
    "grad_norm": ("Gradient norm", "stable", "Magnitude of gradients during training; lower is not always better, but spikes can signal instability."),
    "learning_rate": ("Learning rate", "schedule", "Optimizer step size from the scheduler; this is a controlled schedule, not a quality metric."),
    "eval_runtime": ("Eval runtime", "lower", "Seconds spent on evaluation; lower is faster if evaluation setup is comparable."),
    "eval_samples_per_second": ("Eval samples per second", "higher", "Validation throughput; higher is faster."),
    "eval_steps_per_second": ("Eval steps per second", "higher", "Validation step throughput; higher is faster."),
    "train_loss": ("Final train loss", "lower", "Overall final training loss reported by Trainer; lower usually means better fit."),
    "perplexity": ("Final perplexity", "lower", "Final perplexity from evaluation results; lower is better."),
}
RUN_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"
JOB_RE = re.compile(r"job_qwen3\.(?P<wave>[^.]+)\.(?P<jobid>\d+)\.(?P<kind>out|err)$")


@dataclass(frozen=True)
class Series:
    steps: list[float]
    values: list[float]

    def from_step(self, threshold: int | None) -> "Series":
        if threshold is None:
            return self
        pairs = [(step, value) for step, value in zip(self.steps, self.values) if step >= threshold]
        if not pairs:
            return Series([], [])
        steps, values = zip(*pairs)
        return Series(list(steps), list(values))


@dataclass(frozen=True)
class WaveRun:
    wave: str
    run_dir: Path
    job_id: str | None
    series: dict[str, Series]
    final_metrics: dict[str, float | str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Qwen3 simple-training metrics for positional waves.")
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR, help="Directory containing qwen3_<wave> outputs.")
    parser.add_argument("--graphs-dir", type=Path, default=GRAPHS_DIR, help="Directory where timestamped graph folders are created.")
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR, help="Directory containing job_qwen3.<wave>.<jobid> logs.")
    parser.add_argument("--waves", default=",".join(WAVES), help="Comma-separated waves to plot.")
    parser.add_argument("--run-id", default=None, help="Use runs whose directory name starts with this timestamp/run id.")
    parser.add_argument("--individual", action="store_true", help="Also save one graph per wave for each time-series metric.")
    parser.add_argument(
        "--final-step-threshold",
        type=int,
        default=None,
        help="If set, also save final-segment versions for step-based time-series plots.",
    )
    return parser.parse_args()


def parse_waves(raw_waves: str) -> tuple[str, ...]:
    waves = tuple(wave.strip() for wave in raw_waves.split(",") if wave.strip())
    if not waves:
        raise ValueError("At least one wave must be selected.")
    return waves


def parse_run_timestamp(run_dir: Path) -> datetime:
    try:
        return datetime.strptime(run_dir.name[:19], RUN_TIMESTAMP_FORMAT)
    except ValueError:
        return datetime.fromtimestamp(run_dir.stat().st_mtime)


def latest_run_dir(outputs_dir: Path, wave: str, run_id: str | None) -> Path:
    wave_dir = outputs_dir / f"qwen3_{wave}"
    if not wave_dir.is_dir():
        raise FileNotFoundError(f"Missing output directory for wave '{wave}': {wave_dir}")

    candidates = [path for path in wave_dir.iterdir() if path.is_dir() and (path / "trainer_state.json").is_file()]
    if run_id is not None:
        candidates = [path for path in candidates if path.name.startswith(run_id)]
    if not candidates:
        suffix = f" starting with {run_id!r}" if run_id else ""
        raise FileNotFoundError(f"No Qwen3 trainer_state.json found for wave '{wave}'{suffix}.")
    return max(candidates, key=parse_run_timestamp)


def latest_job_id(jobs_dir: Path, wave: str) -> str | None:
    if not jobs_dir.is_dir():
        return None
    matches = []
    for path in jobs_dir.glob(f"job_qwen3.{wave}.*.out"):
        match = JOB_RE.match(path.name)
        if match:
            matches.append((path.stat().st_mtime, match.group("jobid")))
    if not matches:
        return None
    return max(matches)[1]


def numeric(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(value):
            return float(value)
        return None
    return None


def collect_series(log_history: list[dict[str, object]]) -> dict[str, Series]:
    collected: dict[str, list[tuple[float, float]]] = {}
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
            collected.setdefault(key, []).append((step, value))
            if key == "eval_loss":
                collected.setdefault("eval_perplexity", []).append((step, math.exp(value)))

    return {
        key: Series([step for step, _ in values], [value for _, value in values])
        for key, values in collected.items()
        if len(values) >= 2
    }


def load_final_metrics(run_dir: Path, trainer_state: dict[str, object]) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {"run_dir": str(run_dir)}
    for filename in ("train_results.json", "eval_results.json", "all_results.json"):
        path = run_dir / filename
        if not path.is_file():
            continue
        with path.open() as handle:
            data = json.load(handle)
        for key, value in data.items():
            parsed = numeric(value)
            metrics[key] = parsed if parsed is not None else value

    for row in trainer_state.get("log_history", []):
        if not isinstance(row, dict):
            continue
        for key in ("train_loss", "eval_loss", "eval_accuracy", "total_flos"):
            parsed = numeric(row.get(key))
            if parsed is not None:
                metrics.setdefault(key, parsed)
    return metrics


def load_wave_run(outputs_dir: Path, jobs_dir: Path, wave: str, run_id: str | None) -> WaveRun:
    run_dir = latest_run_dir(outputs_dir, wave, run_id)
    with (run_dir / "trainer_state.json").open() as handle:
        trainer_state = json.load(handle)
    log_history = trainer_state.get("log_history", [])
    if not isinstance(log_history, list):
        raise ValueError(f"Invalid log_history in {run_dir / 'trainer_state.json'}")
    return WaveRun(
        wave=wave,
        run_dir=run_dir,
        job_id=latest_job_id(jobs_dir, wave),
        series=collect_series(log_history),
        final_metrics=load_final_metrics(run_dir, trainer_state),
    )


def create_run_graph_dir(graphs_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_graph_dir = graphs_dir / timestamp
    run_graph_dir.mkdir(parents=True, exist_ok=False)
    return run_graph_dir


def metric_label(metric: str) -> str:
    return METRIC_DESCRIPTIONS.get(metric, (metric.replace("_", " ").title(), "", ""))[0]


def metric_direction(metric: str) -> str:
    return METRIC_DESCRIPTIONS.get(metric, ("", "higher", ""))[1]


def metric_description(metric: str) -> str:
    return METRIC_DESCRIPTIONS.get(metric, ("", "higher", "Numeric training metric."))[2]


def format_value(value: float | str | None) -> str:
    if value is None:
        return "missing"
    if isinstance(value, str):
        return value
    if abs(value) >= 10000 or (abs(value) < 0.001 and value != 0):
        return f"{value:.4e}"
    return f"{value:.6g}"


def ranking_for(values: list[tuple[str, float]], direction: str) -> list[tuple[str, float]]:
    if direction == "lower":
        return sorted(values, key=lambda item: item[1])
    if direction == "higher":
        return sorted(values, key=lambda item: item[1], reverse=True)
    return sorted(values, key=lambda item: item[0])


def segment_summary(series: Series, segments: int = 5) -> list[float | None]:
    if not series.values:
        return [None] * segments

    start = min(series.steps)
    end = max(series.steps)
    if start == end:
        values = [None] * (segments - 1)
        values.append(series.values[-1])
        return values

    result: list[float | None] = []
    span = end - start
    for index in range(segments):
        lower = start + span * index / segments
        upper = start + span * (index + 1) / segments
        bucket = [
            value
            for step, value in zip(series.steps, series.values)
            if (step >= lower and (step < upper or index == segments - 1))
        ]
        result.append(bucket[-1] if bucket else None)
    return result


def trend_text(values: list[float | None], direction: str) -> str:
    present = [value for value in values if value is not None]
    if len(present) < 2:
        return "insufficient data"
    delta = present[-1] - present[0]
    tolerance = max(abs(present[0]) * 0.01, 1e-9)
    if abs(delta) <= tolerance:
        return "mostly flat"
    if direction == "lower":
        return "improves" if delta < 0 else "worsens"
    if direction == "higher":
        return "improves" if delta > 0 else "worsens"
    return "decreases" if delta < 0 else "increases"


def write_metric_section(lines: list[str], runs: list[WaveRun], metric: str) -> None:
    direction = metric_direction(metric)
    direction_note = {
        "lower": "Lower values are better.",
        "higher": "Higher values are better.",
        "stable": "No simple higher/lower ranking: stability and absence of spikes matter most.",
        "schedule": "This is a configured scheduler signal, not a performance ranking.",
    }.get(direction, "Higher values are treated as better.")

    lines.append(f"## {metric_label(metric)}")
    lines.append(metric_description(metric))
    lines.append(direction_note)
    lines.append("")

    rows = []
    for run in runs:
        series = run.series.get(metric)
        if series is None or not series.values:
            rows.append((run.wave, None, [None] * 5, "missing"))
            continue
        segments = segment_summary(series)
        rows.append((run.wave, series.values[-1], segments, trend_text(segments, direction)))

    values_for_ranking = [(wave, final) for wave, final, _, _ in rows if final is not None]
    if values_for_ranking and direction in {"lower", "higher"}:
        ranked = ranking_for(values_for_ranking, direction)
        lines.append("Final ranking: " + " > ".join(f"{wave} ({format_value(value)})" for wave, value in ranked))
    elif values_for_ranking:
        lines.append("Final ordering shown for reference; interpret this metric by trajectory and stability.")
    else:
        lines.append("No values found for this metric.")

    lines.append("")
    lines.append("| wave | 0-20% | 20-40% | 40-60% | 60-80% | 80-100% | final | trend |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for wave, final, segments, trend in rows:
        segment_cells = " | ".join(format_value(value) for value in segments)
        lines.append(f"| {wave} | {segment_cells} | {format_value(final)} | {trend} |")
    lines.append("")

    if values_for_ranking and direction in {"lower", "higher"}:
        winner, winner_value = ranking_for(values_for_ranking, direction)[0]
        lines.append(f"Interpretation: {winner} finishes best on this metric at {format_value(winner_value)}.")
        for wave, final, segments, trend in rows:
            if final is None:
                lines.append(f"- {wave}: missing data.")
            else:
                lines.append(f"- {wave}: final {format_value(final)}; trajectory {trend}.")
    else:
        for wave, final, segments, trend in rows:
            lines.append(f"- {wave}: final {format_value(final)}; trajectory {trend}.")
    lines.append("")


def write_final_metric_section(lines: list[str], runs: list[WaveRun], metric: str) -> None:
    direction = metric_direction(metric)
    lines.append(f"## {metric_label(metric)} Bar Summary")
    lines.append(metric_description(metric))
    lines.append("")
    values = []
    for run in runs:
        value = numeric(run.final_metrics.get(metric))
        if value is not None:
            values.append((run.wave, value))

    if not values:
        lines.append("No final values found.")
        lines.append("")
        return

    if direction in {"lower", "higher"}:
        ranked = ranking_for(values, direction)
        lines.append("Final ranking: " + " > ".join(f"{wave} ({format_value(value)})" for wave, value in ranked))
    else:
        ranked = ranking_for(values, direction)
        lines.append("Final ordering shown for reference: " + " > ".join(f"{wave} ({format_value(value)})" for wave, value in ranked))
    lines.append("")


def write_results_explanation(runs: list[WaveRun], save_path: Path, plotted_metrics: Iterable[str]) -> None:
    lines = [
        "Qwen3 Graph Results Explanation",
        "================================",
        "",
        "This file is generated from the same Trainer JSON data used by graph.py:",
        "- trainer_state.json log_history provides step-wise curves.",
        "- train_results.json, eval_results.json, and all_results.json provide final scalar summaries.",
        "",
        "Runs used:",
    ]
    for run in runs:
        job = run.job_id or "unknown"
        lines.append(f"- {run.wave}: job {job}, {run.run_dir}")
    lines.append("")

    lines.append("## Loss Comparison")
    lines.append("Training loss and eval loss are cross-entropy losses; lower values are better.")
    lines.append("")
    write_metric_section(lines, runs, "loss")
    write_metric_section(lines, runs, "eval_loss")

    for metric in plotted_metrics:
        if metric in {"loss", "eval_loss"}:
            continue
        write_metric_section(lines, runs, metric)

    for metric in ("train_loss", "eval_loss", "eval_accuracy", "perplexity"):
        write_final_metric_section(lines, runs, metric)

    save_path.write_text("\n".join(lines).rstrip() + "\n")
    print(f"Explanation saved: {save_path}")


def draw_metric_comparison(
    runs: Iterable[WaveRun],
    metric: str,
    save_path: Path,
    title: str | None = None,
    ylabel: str | None = None,
    threshold: int | None = None,
    log_scale: bool = False,
) -> bool:
    plt.figure(figsize=(10, 6))
    has_data = False
    for run in runs:
        series = run.series.get(metric)
        if series is None:
            continue
        series = series.from_step(threshold)
        if not series.values:
            continue
        has_data = True
        plt.plot(
            series.steps,
            series.values,
            color=WAVE_COLORS.get(run.wave),
            linewidth=2,
            label=run.wave.capitalize(),
        )

    if not has_data:
        plt.close()
        return False

    plt.xlabel("Step", fontsize=12)
    plt.ylabel(ylabel or metric_label(metric), fontsize=12)
    plt.title(title or f"{metric_label(metric)} vs Step", fontsize=14)
    if log_scale:
        plt.yscale("log")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, which="both", axis="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graph saved: {save_path}")
    return True


def draw_loss_comparison(runs: Iterable[WaveRun], save_path: Path, threshold: int | None = None, log_scale: bool = False) -> bool:
    plt.figure(figsize=(10, 6))
    has_data = False
    for run in runs:
        color = WAVE_COLORS.get(run.wave)
        train_series = run.series.get("loss")
        if train_series is not None:
            train_series = train_series.from_step(threshold)
            if train_series.values:
                has_data = True
                plt.plot(
                    train_series.steps,
                    train_series.values,
                    color=color,
                    linestyle="-",
                    linewidth=2,
                    label=f"{run.wave.capitalize()} Train Loss",
                )

        eval_series = run.series.get("eval_loss")
        if eval_series is not None:
            eval_series = eval_series.from_step(threshold)
            if eval_series.values:
                has_data = True
                plt.plot(
                    eval_series.steps,
                    eval_series.values,
                    color=color,
                    linestyle="--",
                    linewidth=2,
                    label=f"{run.wave.capitalize()} Eval Loss",
                )

    if not has_data:
        plt.close()
        return False

    title = "Training and Eval Loss vs Step"
    if threshold is not None:
        title += f" from Step {threshold}"
    if log_scale:
        title += " (Log Scale)"
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss (log scale)" if log_scale else "Loss", fontsize=12)
    plt.title(title, fontsize=14)
    if log_scale:
        plt.yscale("log")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, which="both", axis="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graph saved: {save_path}")
    return True


def draw_individual(runs: Iterable[WaveRun], metric: str, save_dir: Path) -> None:
    for run in runs:
        series = run.series.get(metric)
        if series is None or not series.values:
            continue
        draw_metric_comparison(
            [run],
            metric,
            save_dir / f"{metric}_vs_step_{run.wave}.pdf",
            title=f"{run.wave.capitalize()} {metric_label(metric)} vs Step",
        )


def write_summary_csv(runs: list[WaveRun], save_path: Path) -> None:
    preferred = [
        "wave",
        "job_id",
        "run_dir",
        "train_loss",
        "eval_loss",
        "eval_accuracy",
        "perplexity",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "eval_runtime",
        "eval_samples_per_second",
        "eval_steps_per_second",
        "total_flos",
    ]
    extra_keys = sorted({key for run in runs for key in run.final_metrics if key not in preferred})
    fieldnames = preferred + extra_keys
    with save_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            row = {"wave": run.wave, "job_id": run.job_id or "", **run.final_metrics}
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"Summary saved: {save_path}")


def draw_final_metric_bars(runs: list[WaveRun], save_dir: Path) -> None:
    for metric in ("train_loss", "eval_loss", "eval_accuracy", "perplexity"):
        values = []
        labels = []
        for run in runs:
            value = numeric(run.final_metrics.get(metric))
            if value is None:
                continue
            labels.append(run.wave.capitalize())
            values.append(value)
        if not values:
            continue

        plt.figure(figsize=(8, 5))
        colors = [WAVE_COLORS.get(label.lower(), "#333333") for label in labels]
        plt.bar(labels, values, color=colors)
        plt.ylabel(metric_label(metric), fontsize=12)
        plt.title(f"Final {metric_label(metric)}", fontsize=14)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        save_path = save_dir / f"final_{metric}.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Graph saved: {save_path}")


def main() -> None:
    args = parse_args()
    waves = parse_waves(args.waves)
    runs = [load_wave_run(args.outputs_dir, args.jobs_dir, wave, args.run_id) for wave in waves]
    run_graph_dir = create_run_graph_dir(args.graphs_dir)

    for run in runs:
        job = f" job {run.job_id}" if run.job_id else ""
        print(f"Using {run.wave}{job}: {run.run_dir}")

    generated = 0
    generated += draw_loss_comparison(runs, run_graph_dir / "loss_vs_step_comparison.pdf")
    generated += draw_loss_comparison(runs, run_graph_dir / "loss_vs_step_comparison_log.pdf", log_scale=True)

    additional_metrics = (
        "eval_accuracy",
        "eval_perplexity",
        "grad_norm",
        "learning_rate",
        "eval_runtime",
        "eval_samples_per_second",
        "eval_steps_per_second",
    )
    for metric in additional_metrics:
        generated += draw_metric_comparison(runs, metric, run_graph_dir / f"{metric}_vs_step_comparison.pdf")
        if args.individual:
            draw_individual(runs, metric, run_graph_dir)

    if args.individual:
        draw_individual(runs, "loss", run_graph_dir)
        draw_individual(runs, "eval_loss", run_graph_dir)

    if args.final_step_threshold is not None:
        suffix = f"from_step_{args.final_step_threshold}"
        generated += draw_loss_comparison(
            runs,
            run_graph_dir / f"loss_vs_step_comparison_{suffix}.pdf",
            threshold=args.final_step_threshold,
        )
        generated += draw_loss_comparison(
            runs,
            run_graph_dir / f"loss_vs_step_comparison_log_{suffix}.pdf",
            threshold=args.final_step_threshold,
            log_scale=True,
        )

    write_summary_csv(runs, run_graph_dir / "final_metrics_summary.csv")
    draw_final_metric_bars(runs, run_graph_dir)
    write_results_explanation(
        runs,
        run_graph_dir / "results_explanation.md",
        ("loss", "eval_loss", *additional_metrics),
    )

    if generated == 0:
        raise SystemExit("No time-series graphs were generated. Check trainer_state.json log_history.")
    print(f"Graph generation completed: {run_graph_dir}")


if __name__ == "__main__":
    main()
