from __future__ import annotations

# ruff: noqa: INP001
import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

LIGHT_TEXT_THRESHOLD = 0.55

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize behavior-cloning metrics.")
    parser.add_argument("--metrics", default="models/bc_v2/metrics.json")
    parser.add_argument("--output-dir", help="Defaults to <metrics directory>/plots.")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--epoch",
        type=int,
        help="Zero-based epoch for validation confusion matrices; defaults to the last recorded epoch.",
    )
    return parser.parse_args()


def plot_loss(history: Sequence[Mapping[str, object]], output_path: Path) -> None:
    epochs = np.asarray([int(item["epoch"]) + 1 for item in history])
    train = np.asarray([float(item["train"]["loss"]) for item in history])
    validation = np.asarray([float(item["validation"]["loss"]) for item in history])

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train, marker="o", markersize=3, label="Train")
    axis.plot(epochs, validation, marker="o", markersize=3, label="Validation")
    best_index = int(validation.argmin())
    axis.scatter(epochs[best_index], validation[best_index], color="tab:red", zorder=3)
    axis.annotate(
        f"best={validation[best_index]:.4f}",
        (epochs[best_index], validation[best_index]),
        xytext=(8, 8),
        textcoords="offset points",
    )
    axis.set(title="Behavior cloning loss", xlabel="Epoch", ylabel="Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _select_metrics(
    metrics: Mapping[str, object],
    split: str,
    epoch: int | None,
) -> tuple[Mapping[str, object], str]:
    if split == "test":
        return metrics["test"], "test"
    history = metrics["history"]
    if not history:
        raise ValueError("metrics.json has no epoch history")
    if epoch is None:
        selected = history[-1]
    else:
        selected = next((item for item in history if int(item["epoch"]) == epoch), None)
        if selected is None:
            message = f"Epoch {epoch} is not present in metrics.json"
            raise ValueError(message)
    return selected["validation"], f"validation epoch {int(selected['epoch']) + 1}"


def plot_confusion_matrices(
    matrices: Mapping[str, Sequence[Sequence[int]]],
    action_schema: Mapping[str, Sequence[str]],
    title_suffix: str,
    output_path: Path,
) -> None:
    names = list(action_schema)
    columns = 3
    rows = (len(names) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(15, 4.6 * rows), squeeze=False)

    for axis, name in zip(axes.flat, names):
        counts = np.asarray(matrices[name], dtype=np.int64)
        row_totals = counts.sum(axis=1, keepdims=True)
        normalized = np.divide(
            counts,
            row_totals,
            out=np.zeros_like(counts, dtype=np.float64),
            where=row_totals != 0,
        )
        image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
        labels = action_schema[name]
        axis.set(
            title=name,
            xlabel="Predicted",
            ylabel="True",
            xticks=np.arange(len(labels)),
            yticks=np.arange(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
        )
        axis.tick_params(axis="x", rotation=35)
        for row in range(counts.shape[0]):
            for column in range(counts.shape[1]):
                color = "white" if normalized[row, column] >= LIGHT_TEXT_THRESHOLD else "black"
                axis.text(
                    column,
                    row,
                    f"{normalized[row, column]:.2f}\n({counts[row, column]:,})",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes.flat[len(names) :]:
        axis.axis("off")
    figure.suptitle(f"Normalized confusion matrices ({title_suffix})", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir) if args.output_dir else metrics_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    history = metrics.get("history", [])
    if history:
        loss_path = output_dir / "loss_curve.png"
        plot_loss(history, loss_path)
        print(f"Saved {loss_path}")

    selected, title_suffix = _select_metrics(metrics, args.split, args.epoch)
    matrices = selected.get("confusion_matrices")
    action_schema = metrics.get("action_schema")
    if matrices is None or action_schema is None:
        print(
            "Confusion matrices are not present in this metrics.json. "
            "They are recorded by the updated trainer on the next training run."
        )
        return
    confusion_path = output_dir / f"confusion_matrices_{args.split}.png"
    plot_confusion_matrices(matrices, action_schema, title_suffix, confusion_path)
    print(f"Saved {confusion_path}")


if __name__ == "__main__":
    main()
