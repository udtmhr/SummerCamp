from __future__ import annotations

# ruff: noqa: INP001
import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import asdict
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
    parser.add_argument("--encoder-type", choices=("unet", "transformer16", "axial32"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--winner-weight", type=float, default=1.5)
    parser.add_argument(
        "--team-selection",
        choices=("winner", "all"),
        default="winner",
        help="Train on the winner only (default), or retain both players.",
    )
    parser.add_argument("--class-weight-exponent", type=float, default=0.5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compile the model; enabled by default on CUDA.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="max-autotune",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="CUDA compute dtype; bfloat16 is the stable default for Transformer training.",
    )
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
    parser.add_argument(
        "--class-statistics-path",
        help="Shared class-statistics cache; defaults to <output-dir>/class_statistics.pt.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=0, help="Limit turns per replay; useful for smoke tests.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_args(args: argparse.Namespace) -> None:
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1")


def resolve_compile(*, enabled: bool | None, device: torch.device) -> bool:
    return device.type == "cuda" if enabled is None else enabled


def resolve_amp_dtype(name: str, device: torch.device) -> torch.dtype:
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]
    if device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bfloat16; use --amp-dtype float16")
    return dtype


def data_split_signature(split: Mapping[str, Iterable[str]]) -> str:
    serialized = json.dumps(
        {name: list(paths) for name, paths in sorted(split.items())},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


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


def run_epoch(  # noqa: C901, PLR0913, PLR0915
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
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started_at = time.perf_counter()
    loss_total = torch.zeros((), device=device)
    sample_count = 0
    optimizer_steps = 0
    metric_totals: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    confusion_totals: dict[str, torch.Tensor] = {}
    amp_enabled = device.type == "cuda" and amp_dtype != torch.float32
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
        sample_count += int(batch["observation"].shape[0])
        with (
            torch.set_grad_enabled(training),
            torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
                dtype=amp_dtype,
            ),
        ):
            output = model(batch["observation"])
            losses = behavior_cloning_loss(output, batch, class_weights)
        if not bool(torch.isfinite(losses["loss"]).detach()):
            phase = description or ("training" if training else "evaluation")
            nonfinite_outputs = [name for name, value in output.items() if not bool(torch.isfinite(value).all())]
            nonfinite_losses = [name for name, value in losses.items() if not bool(torch.isfinite(value).all())]
            message = (
                f"Non-finite loss in {phase} at batch {batch_index}; "
                f"outputs={nonfinite_outputs}, losses={nonfinite_losses}"
            )
            raise FloatingPointError(message)
        if training:
            group_start = ((batch_index - 1) // gradient_accumulation_steps) * gradient_accumulation_steps
            group_size = min(gradient_accumulation_steps, len(loader) - group_start)
            scaler.scale(losses["loss"] / group_size).backward()
            should_step = batch_index % gradient_accumulation_steps == 0 or batch_index == len(loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip,
                    error_if_nonfinite=True,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        loss_total += losses["loss"].detach()
        merge_metrics(metric_totals, compute_metrics(output, batch))
        if collect_confusion:
            merge_confusion_matrices(confusion_totals, compute_confusion_matrices(output, batch))
        if batch_index % 20 == 0 or batch_index == len(loader):
            progress.set_postfix(loss=f"{float(loss_total.cpu()) / batch_index:.4f}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration_seconds = time.perf_counter() - started_at
    result = finalized_metrics(float(loss_total.cpu()), len(loader), metric_totals)
    result.update(
        {
            "duration_seconds": duration_seconds,
            "samples": sample_count,
            "samples_per_second": sample_count / max(duration_seconds, 1e-9),
            "optimizer_steps": optimizer_steps,
            "peak_cuda_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
        }
    )
    if collect_confusion:
        result["confusion_matrices"] = {name: matrix.cpu().tolist() for name, matrix in confusion_totals.items()}
    return result


def warm_up_compiled_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    amp_dtype: torch.dtype,
) -> float:
    model.train()
    model.zero_grad(set_to_none=True)
    batch = move_batch(next(iter(loader)), device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    with torch.amp.autocast(
        device_type=device.type,
        enabled=device.type == "cuda" and amp_dtype != torch.float32,
        dtype=amp_dtype,
    ):
        output = model(batch["observation"])
        warmup_loss = torch.stack(tuple(value.float().mean() for value in output.values())).sum()
    scaler.scale(warmup_loss).backward()
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - started_at


def warm_up_compiled_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> float:
    model.eval()
    batch = move_batch(next(iter(loader)), device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    with (
        torch.inference_mode(),
        torch.amp.autocast(
            device_type=device.type,
            enabled=device.type == "cuda" and amp_dtype != torch.float32,
            dtype=amp_dtype,
        ),
    ):
        model(batch["observation"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - started_at


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


def make_datasets(
    split: Mapping[str, Iterable[str]],
    team_selection: str,
    winner_weight: float,
    seed: int,
    max_turns: int,
) -> dict[str, LuxReplayDataset]:
    return {
        "train": LuxReplayDataset(
            [Path(path) for path in split["train"]],
            augment=True,
            team_selection=team_selection,
            winner_weight=winner_weight,
            seed=seed,
            max_turns=max_turns,
        ),
        "train_counting": LuxReplayDataset(
            [Path(path) for path in split["train"]],
            augment=False,
            team_selection=team_selection,
            winner_weight=winner_weight,
            seed=seed,
            max_turns=max_turns,
        ),
        "validation": LuxReplayDataset(
            [Path(path) for path in split["validation"]],
            augment=False,
            team_selection=team_selection,
            winner_weight=winner_weight,
            seed=seed,
            max_turns=max_turns,
        ),
        "test": LuxReplayDataset(
            [Path(path) for path in split["test"]],
            augment=False,
            team_selection=team_selection,
            winner_weight=winner_weight,
            seed=seed,
            max_turns=max_turns,
        ),
    }


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(args.amp_dtype, device)
    compile_enabled = resolve_compile(enabled=args.compile, device=device)
    configure_device(device)
    num_workers = resolve_num_workers(args.num_workers)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = None

    if args.resume:
        model, checkpoint = load_bc_checkpoint(args.resume, str(device))
        checkpoint_encoder_type = model.config.encoder_type
        if args.encoder_type is not None and args.encoder_type != checkpoint_encoder_type:
            msg = f"Checkpoint encoder is {checkpoint_encoder_type}, not {args.encoder_type}"
            raise ValueError(msg)
        args.encoder_type = checkpoint_encoder_type
        split = checkpoint["split"]
        start_epoch = int(checkpoint["epoch"]) + 1
    else:
        args.encoder_type = args.encoder_type or "unet"
        replay_paths = discover_replays(args.replay_dir)
        path_split = split_replays(replay_paths, seed=args.seed)
        split = {name: [str(path) for path in paths] for name, paths in path_split.items()}
        model = LuxBehaviorCloningModel(ModelConfig(encoder_type=args.encoder_type)).to(device)
        start_epoch = 0
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)

    datasets = make_datasets(
        split,
        args.team_selection,
        args.winner_weight,
        args.seed,
        args.max_turns,
    )
    show_progress = not args.no_progress
    tqdm.write(
        f"encoder={args.encoder_type} device={device} workers={num_workers} batch_size={args.batch_size} "
        f"effective_batch={args.batch_size * args.gradient_accumulation_steps} "
        f"compile={compile_enabled} compile_mode={args.compile_mode} amp_dtype={args.amp_dtype} "
        f"train={len(datasets['train']):,} validation={len(datasets['validation']):,} "
        f"test={len(datasets['test']):,}"
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
    )
    statistics_path = (
        Path(args.class_statistics_path) if args.class_statistics_path else output_dir / "class_statistics.pt"
    )
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
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    if args.resume and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda" and amp_dtype == torch.float16,
    )
    if args.resume and checkpoint.get("scaler") is not None and scaler.is_enabled():
        scaler.load_state_dict(checkpoint["scaler"])
    execution_model = torch.compile(model, mode=args.compile_mode) if compile_enabled else model
    compile_warmup_seconds = 0.0
    if compile_enabled:
        compile_warmup_seconds = warm_up_compiled_model(
            execution_model,
            loaders["validation"],
            device,
            scaler,
            amp_dtype,
        )
        seed_everything(args.seed)
        tqdm.write(f"torch.compile warmup: {compile_warmup_seconds:.2f}s")

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
            execution_model,
            loaders["train"],
            device,
            weights,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip=args.gradient_clip,
            description=f"Train {epoch + 1}/{args.epochs}",
            show_progress=show_progress,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            amp_dtype=amp_dtype,
        )
        if compile_enabled and epoch == start_epoch:
            inference_warmup_seconds = warm_up_compiled_inference(
                execution_model,
                loaders["validation"],
                device,
                amp_dtype,
            )
            compile_warmup_seconds += inference_warmup_seconds
            tqdm.write(f"torch.compile inference warmup: {inference_warmup_seconds:.2f}s")
        with torch.inference_mode():
            validation_metrics = run_epoch(
                execution_model,
                loaders["validation"],
                device,
                weights,
                description=f"Validation {epoch + 1}/{args.epochs}",
                show_progress=show_progress,
                collect_confusion=True,
                amp_dtype=amp_dtype,
            )
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
            scaler=scaler,
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
                scaler=scaler,
            )
        epoch_progress.set_postfix(
            train=f"{train_metrics['loss']:.4f}",
            validation=f"{validation_metrics['loss']:.4f}",
            best=f"{best_loss:.4f}",
        )

    with torch.inference_mode():
        test_metrics = run_epoch(
            execution_model,
            loaders["test"],
            device,
            weights,
            description="Test",
            show_progress=show_progress,
            collect_confusion=True,
            amp_dtype=amp_dtype,
        )
    summary = {
        "device": str(device),
        "model_config": asdict(model.config),
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "encoder_parameter_count": sum(parameter.numel() for parameter in model.encoder.parameters()),
        "compile_enabled": compile_enabled,
        "compile_mode": args.compile_mode,
        "amp_dtype": args.amp_dtype,
        "compile_warmup_seconds": compile_warmup_seconds,
        "data_split_signature": data_split_signature(split),
        "class_statistics_signature": statistics_signature,
        "training_config": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "compile_enabled": compile_enabled,
            "compile_mode": args.compile_mode,
            "amp_dtype": args.amp_dtype,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "winner_weight": args.winner_weight,
            "team_selection": args.team_selection,
            "class_weight_exponent": args.class_weight_exponent,
            "gradient_clip": args.gradient_clip,
            "max_turns": args.max_turns,
        },
        "action_schema": {name: list(actions) for name, actions in ACTION_SCHEMA.items()},
        "test": test_metrics,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    tqdm.write(json.dumps({"test": test_metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
