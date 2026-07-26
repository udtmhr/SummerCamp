from __future__ import annotations

# ruff: noqa: INP001
import argparse
import json
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from luxai2021.imitation.actions import ACTION_SCHEMA
from luxai2021.imitation.class_stats import (
    checkpoint_class_statistics,
    class_statistics_signature,
    load_class_statistics,
    save_class_statistics,
)
from luxai2021.imitation.data import (
    LuxReplayDataset,
    ReplayBatchSampler,
    class_counts,
    discover_replays,
    discover_sources,
    limit_replays_per_source,
    split_replays,
)
from luxai2021.imitation.model import (
    LuxBehaviorCloningModel,
    ModelConfig,
    behavior_cloning_loss,
    compute_confusion_matrices,
    compute_metrics,
    load_bc_checkpoint,
    make_class_weights,
    save_bc_checkpoint,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Lux AI 2021 behavior-cloning policy.")
    parser.add_argument("--replay-dir", required=True, help="Directory containing Kaggle replay JSON files.")
    parser.add_argument("--output-dir", default="models/bc", help="Checkpoint and metrics output directory.")
    parser.add_argument("--resume", help="Checkpoint to resume from.")
    parser.add_argument("--training-profile", choices=("baseline", "durrett"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--winner-weight", type=float, default=1.5)
    parser.add_argument(
        "--team-selection",
        choices=("winner", "all", "source"),
        help="Select winner, both players, or the source submission represented by each replay directory.",
    )
    parser.add_argument("--class-weight-exponent", type=float)
    parser.add_argument("--stay-weight", type=float)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or an explicit torch device.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="-1 selects up to 4 workers automatically; use 0 for in-process loading.",
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--recompute-class-statistics",
        action="store_true",
        help="Ignore cached class counts and scan the training split again.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=0, help="Limit turns per replay; useful for smoke tests.")
    parser.add_argument(
        "--max-replays-per-source",
        type=int,
        default=0,
        help="Limit each source submission to this many replays; zero keeps all replays.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def apply_profile_defaults(args: argparse.Namespace, checkpoint: Mapping[str, object] | None) -> None:
    checkpoint_profile = None if checkpoint is None else str(checkpoint.get("training_profile", "baseline"))
    if args.training_profile is None:
        args.training_profile = checkpoint_profile or "baseline"
    elif checkpoint_profile is not None and args.training_profile != checkpoint_profile:
        msg = f"Checkpoint profile is {checkpoint_profile}, not {args.training_profile}"
        raise ValueError(msg)

    if args.training_profile == "durrett":
        defaults = {
            "epochs": 100,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "team_selection": "source",
            "class_weight_exponent": 0.0,
            "stay_weight": 0.3,
            "gradient_accumulation_steps": 4,
        }
        if args.team_selection not in {None, "source"}:
            raise ValueError("The durrett profile requires --team-selection source")
    else:
        defaults = {
            "epochs": 20,
            "batch_size": 32,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "team_selection": "winner",
            "class_weight_exponent": 0.5,
            "stay_weight": None,
            "gradient_accumulation_steps": 1,
        }
        if args.team_selection == "source":
            raise ValueError("--team-selection source requires --training-profile durrett")
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_num_workers(num_workers: int) -> int:
    if num_workers >= 0:
        return num_workers
    return min(4, max(1, (os.cpu_count() or 2) // 4))


def configure_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    for name, value in batch.items():
        if name == "observation" and device.type == "cuda":
            moved[name] = value.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            moved[name] = value.to(device, non_blocking=True)
    return moved


def merge_metrics(
    totals: dict[str, tuple[torch.Tensor, torch.Tensor]],
    metrics: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    for name, (correct, count) in metrics.items():
        if name not in totals:
            totals[name] = correct.detach(), count.detach()
        else:
            old_correct, old_count = totals[name]
            totals[name] = old_correct + correct.detach(), old_count + count.detach()


def finalized_metrics(
    loss_total: float,
    batches: int,
    metrics: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    result = {"loss": loss_total / max(batches, 1)}
    result.update(
        {
            name: float((correct.float() / count.clamp_min(1)).cpu()) if int(count.cpu()) else 0.0
            for name, (correct, count) in metrics.items()
        }
    )
    return result


def merge_confusion_matrices(totals: dict[str, torch.Tensor], matrices: Mapping[str, torch.Tensor]) -> None:
    for name, matrix in matrices.items():
        detached = matrix.detach()
        totals[name] = detached if name not in totals else totals[name] + detached


def run_epoch(  # noqa: C901, PLR0913
    model: LuxBehaviorCloningModel,
    loader: DataLoader,
    device: torch.device,
    class_weights: Mapping[str, torch.Tensor],
    *,
    optimizer: torch.optim.Optimizer = None,
    scaler: torch.amp.GradScaler = None,
    gradient_clip: float = 1.0,
    description: str = "",
    show_progress: bool = True,
    collect_confusion: bool = False,
    gradient_accumulation_steps: int = 1,
    source_ids: tuple[int, ...] = (),
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    loss_total = torch.zeros((), device=device)
    metric_totals: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    confusion_totals: dict[str, torch.Tensor] = {}
    source_metric_totals: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    source_loss_totals: dict[int, float] = {}
    source_batch_counts: dict[int, int] = {}
    amp_enabled = device.type == "cuda"
    progress = tqdm(
        loader,
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, raw_batch in enumerate(progress, start=1):
        batch = move_batch(raw_batch, device)
        source_index = batch["source_index"] if source_ids else None
        with torch.set_grad_enabled(training), torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(batch["observation"], source_index=source_index)
            losses = behavior_cloning_loss(output, batch, class_weights)
        if training:
            group_start = ((batch_index - 1) // gradient_accumulation_steps) * gradient_accumulation_steps
            group_size = min(gradient_accumulation_steps, len(loader) - group_start)
            scaler.scale(losses["loss"] / group_size).backward()
            should_step = batch_index % gradient_accumulation_steps == 0 or batch_index == len(loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        loss_total += losses["loss"].detach()
        merge_metrics(metric_totals, compute_metrics(output, batch))
        if source_ids:
            with torch.no_grad():
                for source_index_value in torch.unique(batch["source_index"]).tolist():
                    selection = batch["source_index"] == source_index_value
                    source_output = {name: value[selection].detach().float() for name, value in output.items()}
                    source_batch = {name: value[selection] for name, value in batch.items()}
                    source_losses = behavior_cloning_loss(source_output, source_batch, class_weights)
                    source_loss_totals[source_index_value] = source_loss_totals.get(source_index_value, 0.0) + float(
                        source_losses["loss"]
                    )
                    source_batch_counts[source_index_value] = source_batch_counts.get(source_index_value, 0) + 1
                    source_totals = source_metric_totals.setdefault(source_index_value, {})
                    merge_metrics(source_totals, compute_metrics(source_output, source_batch))
        if collect_confusion:
            merge_confusion_matrices(confusion_totals, compute_confusion_matrices(output, batch))
        if batch_index % 20 == 0 or batch_index == len(loader):
            progress.set_postfix(loss=f"{float(loss_total.cpu()) / batch_index:.4f}")
    result = finalized_metrics(float(loss_total.cpu()), len(loader), metric_totals)
    if source_ids:
        result["sources"] = {
            str(source_ids[source_index_value]): finalized_metrics(
                source_loss_totals[source_index_value],
                source_batch_counts[source_index_value],
                source_metric_totals[source_index_value],
            )
            for source_index_value in sorted(source_metric_totals)
        }
    if collect_confusion:
        result["confusion_matrices"] = {name: matrix.cpu().tolist() for name, matrix in confusion_totals.items()}
    return result


def make_loader(  # noqa: PLR0913
    dataset: LuxReplayDataset,
    *,
    batch_size: int | None,
    batch_sampler: ReplayBatchSampler | None,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    options = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    if num_workers > 0:
        options["prefetch_factor"] = prefetch_factor
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, **options)
    return DataLoader(dataset, batch_size=batch_size, **options)


def make_datasets(  # noqa: PLR0913
    split: Mapping[str, Iterable[str]],
    team_selection: str,
    winner_weight: float,
    seed: int,
    max_turns: int,
    source_ids: tuple[int, ...],
) -> dict[str, LuxReplayDataset]:
    return {
        "train": LuxReplayDataset(
            [Path(path) for path in split["train"]],
            augment=True,
            team_selection=team_selection,
            winner_weight=winner_weight,
            seed=seed,
            max_turns=max_turns,
            source_ids=source_ids,
        ),
        "train_counting": LuxReplayDataset(
            [Path(path) for path in split["train"]],
            augment=False,
            team_selection=team_selection,
            winner_weight=winner_weight,
            seed=seed,
            max_turns=max_turns,
            source_ids=source_ids,
        ),
        "validation": LuxReplayDataset(
            [Path(path) for path in split["validation"]],
            augment=False,
            team_selection=team_selection,
            winner_weight=winner_weight,
            seed=seed,
            max_turns=max_turns,
            source_ids=source_ids,
        ),
        "test": LuxReplayDataset(
            [Path(path) for path in split["test"]],
            augment=False,
            team_selection=team_selection,
            winner_weight=winner_weight,
            seed=seed,
            max_turns=max_turns,
            source_ids=source_ids,
        ),
    }


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    configure_device(device)
    num_workers = resolve_num_workers(args.num_workers)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    source_catalog: tuple[Mapping[str, object], ...] = ()
    default_source_id = None

    if args.resume:
        model, checkpoint = load_bc_checkpoint(args.resume, str(device))
        apply_profile_defaults(args, checkpoint)
        split = checkpoint["split"]
        source_catalog = tuple(checkpoint.get("source_catalog") or ())
        default_source_id = checkpoint.get("default_source_id")
        start_epoch = int(checkpoint["epoch"]) + 1
    else:
        apply_profile_defaults(args, None)
        replay_paths = discover_replays(args.replay_dir)
        if args.training_profile == "durrett":
            replay_paths = limit_replays_per_source(replay_paths, args.max_replays_per_source)
            sources = discover_sources(replay_paths)
            if not sources:
                raise ValueError("No source submissions were discovered")
            source_catalog = tuple(
                {
                    "source_id": source.source_id,
                    "lb": source.lb,
                    "index": index,
                }
                for index, source in enumerate(sources)
            )
            default_source_id = max(sources, key=lambda source: (source.lb, -source.source_id)).source_id
            source_ids = tuple(source.source_id for source in sources)
            model_config = ModelConfig(
                base_channels=384,
                feature_channels=384,
                encoder_type="durrett",
                source_ids=source_ids,
            )
        else:
            if args.max_replays_per_source:
                raise ValueError("--max-replays-per-source requires --training-profile durrett")
            model_config = ModelConfig()
        path_split = split_replays(replay_paths, seed=args.seed)
        split = {name: [str(path) for path in paths] for name, paths in path_split.items()}
        model = LuxBehaviorCloningModel(model_config).to(device)
        start_epoch = 0
    source_ids = tuple(model.config.source_ids)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)

    datasets = make_datasets(
        split,
        args.team_selection,
        args.winner_weight,
        args.seed,
        args.max_turns,
        source_ids,
    )
    show_progress = not args.no_progress
    tqdm.write(
        f"profile={args.training_profile} device={device} workers={num_workers} batch_size={args.batch_size} "
        f"effective_batch={args.batch_size * args.gradient_accumulation_steps} train={len(datasets['train']):,} "
        f"validation={len(datasets['validation']):,} test={len(datasets['test']):,} sources={len(source_ids)}"
    )
    train_sampler = ReplayBatchSampler(
        datasets["train"],
        args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    loaders = {
        "train": make_loader(
            datasets["train"],
            batch_size=None,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=device.type == "cuda",
            persistent_workers=True,
        ),
        "validation": make_loader(
            datasets["validation"],
            batch_size=args.batch_size,
            batch_sampler=None,
            num_workers=num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
        ),
        "test": make_loader(
            datasets["test"],
            batch_size=args.batch_size,
            batch_sampler=None,
            num_workers=num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
        ),
    }
    statistics_signature = class_statistics_signature(
        split["train"],
        team_selection=args.team_selection,
        max_turns=args.max_turns,
        source_ids=source_ids,
    )
    statistics_path = output_dir / "class_statistics.pt"
    counts = None
    statistics_source = None
    if not args.recompute_class_statistics:
        counts = checkpoint_class_statistics(checkpoint, statistics_signature)
        statistics_source = "checkpoint" if counts is not None else None
        if counts is None:
            counts = load_class_statistics(statistics_path, statistics_signature)
            statistics_source = str(statistics_path) if counts is not None else None
    if counts is None:
        counts = class_counts(
            datasets["train_counting"],
            show_progress=show_progress,
            num_workers=num_workers,
            prefetch_factor=args.prefetch_factor,
        )
        save_class_statistics(statistics_path, statistics_signature, counts)
        statistics_source = "computed"
    tqdm.write(f"Class statistics: {statistics_source}")
    weights = make_class_weights(
        counts,
        device,
        exponent=args.class_weight_exponent,
        stay_weight=args.stay_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    if args.resume and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler = None
    if args.training_profile == "durrett":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(50, 80), gamma=0.1)
        if args.resume and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best_loss = float(checkpoint["metrics"]["validation"]["loss"]) if args.resume else float("inf")
    history = []
    epoch_progress = tqdm(
        range(start_epoch, args.epochs),
        desc="Epochs",
        unit="epoch",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for epoch in epoch_progress:
        train_metrics = run_epoch(
            model,
            loaders["train"],
            device,
            weights,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip=args.gradient_clip,
            description=f"Train {epoch + 1}/{args.epochs}",
            show_progress=show_progress,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            source_ids=source_ids,
        )
        with torch.inference_mode():
            validation_metrics = run_epoch(
                model,
                loaders["validation"],
                device,
                weights,
                description=f"Validation {epoch + 1}/{args.epochs}",
                show_progress=show_progress,
                collect_confusion=True,
                source_ids=source_ids,
            )
        if scheduler is not None:
            scheduler.step()
        epoch_metrics = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(epoch_metrics)
        tqdm.write(json.dumps(epoch_metrics, sort_keys=True))
        save_bc_checkpoint(
            output_dir / "latest.pt",
            model,
            optimizer,
            epoch,
            epoch_metrics,
            split,
            class_counts=counts,
            class_statistics_signature=statistics_signature,
            scheduler=scheduler,
            training_profile=args.training_profile,
            source_catalog=source_catalog,
            default_source_id=default_source_id,
        )
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            save_bc_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                epoch_metrics,
                split,
                class_counts=counts,
                class_statistics_signature=statistics_signature,
                scheduler=scheduler,
                training_profile=args.training_profile,
                source_catalog=source_catalog,
                default_source_id=default_source_id,
            )
        epoch_progress.set_postfix(
            train=f"{train_metrics['loss']:.4f}",
            validation=f"{validation_metrics['loss']:.4f}",
            best=f"{best_loss:.4f}",
        )

    with torch.inference_mode():
        test_metrics = run_epoch(
            model,
            loaders["test"],
            device,
            weights,
            description="Test",
            show_progress=show_progress,
            collect_confusion=True,
            source_ids=source_ids,
        )
    summary = {
        "device": str(device),
        "training_profile": args.training_profile,
        "source_catalog": source_catalog,
        "default_source_id": default_source_id,
        "action_schema": {name: list(actions) for name, actions in ACTION_SCHEMA.items()},
        "test": test_metrics,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    tqdm.write(json.dumps({"test": test_metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
