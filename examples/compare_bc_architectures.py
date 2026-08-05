from __future__ import annotations

# ruff: noqa: INP001
import argparse
import json
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_ACTIVE_ACCURACY_NAMES = (
    "worker_active_accuracy",
    "cart_active_accuracy",
    "city_active_accuracy",
)
_MINIMUM_RUN_COUNT = 2
_ARCHITECTURE_VARIANT_FIELDS = frozenset(("encoder_type", "output_dir"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare behavior-cloning encoder runs.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=METRICS_JSON",
        help="Named metrics.json input; provide at least two.",
    )
    parser.add_argument("--output-dir", default="models/bc_architecture_comparison")
    return parser.parse_args()


def parse_run(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        message = f"Expected NAME=METRICS_JSON, got {value!r}"
        raise ValueError(message)
    return name, Path(raw_path)


def load_runs(values: Sequence[str]) -> dict[str, Mapping[str, object]]:
    if len(values) < _MINIMUM_RUN_COUNT:
        raise ValueError("At least two --run values are required")
    runs = {}
    for value in values:
        name, path = parse_run(value)
        if name in runs:
            message = f"Duplicate run name: {name}"
            raise ValueError(message)
        runs[name] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def _comparable_training_config(metrics: Mapping[str, object]) -> dict[str, object]:
    config = metrics.get("training_config")
    if not isinstance(config, dict):
        return {}
    return {name: value for name, value in config.items() if name not in _ARCHITECTURE_VARIANT_FIELDS}


def validate_comparable_runs(runs: Mapping[str, Mapping[str, object]]) -> None:
    first_name, first = next(iter(runs.items()))
    required = ("data_split_signature", "class_statistics_signature")
    for field in required:
        if field not in first:
            message = f"Run {first_name} does not contain {field}"
            raise ValueError(message)
    if not isinstance(first.get("training_config"), dict):
        message = f"Run {first_name} does not contain training_config"
        raise TypeError(message)
    for name, metrics in list(runs.items())[1:]:
        for field in required:
            if metrics.get(field) != first[field]:
                message = f"Run {name} has a different {field}"
                raise ValueError(message)
        if _comparable_training_config(metrics) != _comparable_training_config(first):
            message = f"Run {name} has a different training_config"
            raise ValueError(message)


def summarize_run(name: str, metrics: Mapping[str, object]) -> dict[str, object]:
    history = metrics.get("history") or []
    if not history:
        message = f"Run {name} has no epoch history"
        raise ValueError(message)
    best = min(history, key=lambda item: float(item["validation"]["loss"]))
    validation = best["validation"]
    train_throughputs = [float(item["train"]["samples_per_second"]) for item in history]
    memory_values = [
        int(item["train"]["peak_cuda_memory_allocated_bytes"])
        for item in history
        if item["train"].get("peak_cuda_memory_allocated_bytes") is not None
    ]
    model_config = metrics["model_config"]
    return {
        "name": name,
        "encoder_type": model_config["encoder_type"],
        "best_epoch": int(best["epoch"]),
        "validation_loss": float(validation["loss"]),
        **{metric: float(validation.get(metric, 0.0)) for metric in _ACTIVE_ACCURACY_NAMES},
        "test_loss": float(metrics["test"]["loss"]),
        "model_parameter_count": int(metrics["model_parameter_count"]),
        "encoder_parameter_count": int(metrics["encoder_parameter_count"]),
        "train_samples_per_second": median(train_throughputs),
        "peak_cuda_memory_allocated_bytes": max(memory_values) if memory_values else None,
    }


def build_comparison(runs: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    validate_comparable_runs(runs)
    rows = sorted(
        (summarize_run(name, metrics) for name, metrics in runs.items()),
        key=lambda row: row["validation_loss"],
    )
    first = next(iter(runs.values()))
    return {
        "primary_metric": "validation_loss",
        "winner": rows[0]["name"],
        "data_split_signature": first["data_split_signature"],
        "class_statistics_signature": first["class_statistics_signature"],
        "training_config": _comparable_training_config(first),
        "runs": rows,
    }


def plot_loss_curves(runs: Mapping[str, Mapping[str, object]], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for name, metrics in runs.items():
        history = metrics["history"]
        epochs = [int(item["epoch"]) + 1 for item in history]
        losses = [float(item["validation"]["loss"]) for item in history]
        axis.plot(epochs, losses, marker="o", markersize=3, label=name)
    axis.set(title="Validation loss by encoder", xlabel="Epoch", ylabel="Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_resources(comparison: Mapping[str, object], output_path: Path) -> None:
    rows = comparison["runs"]
    names = [row["name"] for row in rows]
    throughput = [row["train_samples_per_second"] for row in rows]
    memory_gib = [
        0.0 if row["peak_cuda_memory_allocated_bytes"] is None else row["peak_cuda_memory_allocated_bytes"] / 2**30
        for row in rows
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(names, throughput)
    axes[0].set(title="Training throughput", ylabel="Samples / second")
    axes[1].bar(names, memory_gib)
    axes[1].set(title="Peak CUDA memory", ylabel="GiB")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def print_table(comparison: Mapping[str, object]) -> None:
    print("name\tvalidation_loss\tworker_active\tcart_active\tcity_active\tsamples/s\tpeak_GiB\tparams")
    for row in comparison["runs"]:
        memory = row["peak_cuda_memory_allocated_bytes"]
        memory_gib = "-" if memory is None else f"{memory / 2**30:.2f}"
        print(
            f"{row['name']}\t{row['validation_loss']:.6f}\t"
            f"{row['worker_active_accuracy']:.4f}\t{row['cart_active_accuracy']:.4f}\t"
            f"{row['city_active_accuracy']:.4f}\t{row['train_samples_per_second']:.1f}\t"
            f"{memory_gib}\t{row['model_parameter_count']:,}"
        )


def main() -> None:
    args = parse_args()
    runs = load_runs(args.run)
    comparison = build_comparison(runs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "architecture_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    plot_loss_curves(runs, output_dir / "validation_loss_comparison.png")
    plot_resources(comparison, output_dir / "resource_comparison.png")
    print_table(comparison)
    print(f"Saved {comparison_path}")


if __name__ == "__main__":
    main()
