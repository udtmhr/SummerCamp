from __future__ import annotations

# ruff: noqa: C901, INP001, PLR0912, PLR0913, PLR0915
import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from luxai2021.imitation.actions import FIRST_PLACE_ACTION_SCHEMA
from luxai2021.imitation.data import ReplayBatchSampler, discover_replays, prepare_replay_cache, split_replays
from luxai2021.imitation.distillation import (
    LuxDistillationDataset,
    augment_distillation_batch,
    compact_distillation_collate,
    distillation_loss,
    distillation_metrics,
)
from luxai2021.imitation.first_place import FIRST_PLACE_TEACHER_SHA256
from luxai2021.imitation.model import (
    POLICY_SCHEMA_FIRST_PLACE_FLAT,
    LuxBehaviorCloningModel,
    ModelConfig,
    load_bc_checkpoint,
    save_bc_checkpoint,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

DEFAULT_STUDENTS = {
    "unet": "models/bc_v2/best.pt",
    "resnet17x32": None,
    "resnet17x48": None,
    "resattn8": None,
    "transformer16": "models/bc_encoder_compare/transformer16/best.pt",
    "axial32": "models/bc_encoder_compare/axial32/best.pt",
    "axial32_4m5": None,
}
DEFAULT_LEARNING_RATE = 1e-4


def data_split_signature(split: Mapping[str, Iterable[str]]) -> str:
    serialized = json.dumps(
        {name: list(paths) for name, paths in sorted(split.items())},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill the Lux AI 2021 first-place policy.")
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--teacher-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder-type", choices=tuple(DEFAULT_STUDENTS), default="unet")
    parser.add_argument("--student-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--replay-cache-dir")
    parser.add_argument("--prepared-cache-dir")
    parser.add_argument("--prepared-observation-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--learning-rate",
        type=float,
        help=(
            "Initial learning rate. Fresh runs default to 1e-4; resumed runs preserve the checkpoint rate "
            "unless this option is specified."
        ),
    )
    parser.add_argument("--lr-scheduler", choices=("none", "plateau"), default="plateau")
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--lr-threshold", type=float, default=2e-3)
    parser.add_argument("--min-learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--distill-weight", type=float, default=0.75)
    parser.add_argument("--hard-label-weight", type=float, default=0.0)
    parser.add_argument("--illegal-weight", type=float, default=0.1)
    parser.add_argument("--winner-weight", type=float, default=1.5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--no-compile", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    *,
    factor: float,
    patience: int,
    threshold: float,
    min_learning_rate: float,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if scheduler_name == "none":
        return None
    if scheduler_name != "plateau":
        message = f"Unsupported learning-rate scheduler: {scheduler_name}"
        raise ValueError(message)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        threshold=threshold,
        threshold_mode="abs",
        cooldown=0,
        min_lr=min_learning_rate,
    )


def restore_lr_scheduler(
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    checkpoint: Mapping[str, object] | None,
    best_validation_loss: float | None = None,
) -> None:
    if scheduler is None or checkpoint is None:
        return
    state = checkpoint.get("lr_scheduler_state")
    if isinstance(state, dict):
        scheduler.load_state_dict(state)
        return
    if best_validation_loss is not None:
        scheduler.step(best_validation_loss)
        return
    validation = checkpoint.get("metrics", {}).get("validation", {})
    if "loss" in validation:
        scheduler.step(float(validation["loss"]))


def load_existing_history(path: Path, start_epoch: int, model_config: ModelConfig) -> list[dict[str, object]]:
    if not path.exists():
        return []
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("model_config") != asdict(model_config):
        return []
    return [item for item in summary.get("history", []) if int(item["epoch"]) < start_epoch]


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    for name, value in batch.items():
        if name == "observation" and device.type == "cuda":
            moved[name] = value.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
        elif name == "observation":
            moved[name] = value.to(device, dtype=torch.float32, non_blocking=True)
        else:
            moved[name] = value.to(device, non_blocking=True)
    return moved


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    temperature: float,
    distill_weight: float,
    hard_label_weight: float,
    illegal_weight: float,
    amp_dtype: torch.dtype,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip: float = 1.0,
    accumulation_steps: int = 1,
    description: str,
    show_progress: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    totals: dict[str, torch.Tensor] = {}
    metric_totals: dict[str, list[torch.Tensor]] = {}
    sample_count = 0
    started = time.perf_counter()
    progress = tqdm(loader, desc=description, unit="batch", disable=not show_progress, dynamic_ncols=True, leave=False)
    for batch_index, source_batch in enumerate(progress):
        batch = move_batch(source_batch, device)
        if training:
            batch = augment_distillation_batch(batch)
        autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
        with (
            torch.set_grad_enabled(training),
            torch.autocast(
                device.type,
                dtype=amp_dtype,
                enabled=autocast_enabled,
            ),
        ):
            output = model(batch["observation"])
            losses = distillation_loss(
                output,
                batch,
                temperature=temperature,
                distill_weight=distill_weight,
                hard_label_weight=hard_label_weight,
                illegal_weight=illegal_weight,
            )
            loss = losses["loss"] / accumulation_steps
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            should_step = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader)
            if should_step:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        real_samples = source_batch["observation"].shape[0]
        sample_count += real_samples
        for name, value in losses.items():
            weighted = value.detach().float() * real_samples
            totals[name] = totals.get(name, torch.zeros_like(weighted)) + weighted
        for name, (numerator, denominator) in distillation_metrics(output, batch).items():
            pair = metric_totals.setdefault(
                name,
                [torch.zeros_like(numerator), torch.zeros_like(denominator)],
            )
            pair[0] += numerator
            pair[1] += denominator
        if (batch_index + 1) % 50 == 0 or batch_index + 1 == len(loader):
            postfix = {"loss": f"{losses['loss'].detach().float().item():.4f}"}
            
            unit_ill_top1_num = metric_totals.get("worker_illegal_top1", [0, 1])[0] + metric_totals.get("cart_illegal_top1", [0, 1])[0]
            unit_ill_top1_den = metric_totals.get("worker_illegal_top1", [0, 1])[1] + metric_totals.get("cart_illegal_top1", [0, 1])[1]
            if unit_ill_top1_den > 0:
                postfix["u_top1"] = f"{float(unit_ill_top1_num) / float(unit_ill_top1_den):.1%}"
                
            city_ill_top1_num = metric_totals.get("city_illegal_top1", [0, 1])[0]
            city_ill_top1_den = metric_totals.get("city_illegal_top1", [0, 1])[1]
            if city_ill_top1_den > 0:
                postfix["c_top1"] = f"{float(city_ill_top1_num) / float(city_ill_top1_den):.1%}"
                
            unit_ill_prob_num = metric_totals.get("worker_illegal_prob_mass", [0, 1])[0] + metric_totals.get("cart_illegal_prob_mass", [0, 1])[0]
            if unit_ill_top1_den > 0:
                postfix["u_prob"] = f"{float(unit_ill_prob_num) / float(unit_ill_top1_den):.3f}"
                
            city_ill_prob_num = metric_totals.get("city_illegal_prob_mass", [0, 1])[0]
            if city_ill_top1_den > 0:
                postfix["c_prob"] = f"{float(city_ill_prob_num) / float(city_ill_top1_den):.3f}"
                
            progress.set_postfix(**postfix)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    result = {name: value.item() / max(sample_count, 1) for name, value in totals.items()}
    result.update(
        {
            name: numerator.item() / max(denominator.item(), 1)
            for name, (numerator, denominator) in metric_totals.items()
        }
    )
    result.update(
        {
            "sample_count": sample_count,
            "duration_seconds": duration,
            "samples_per_second": sample_count / max(duration, 1e-9),
        }
    )
    return result


def make_loader(
    dataset: LuxDistillationDataset,
    batch_size: int,
    *,
    training: bool,
    num_workers: int,
    prefetch_factor: int,
    seed: int,
) -> DataLoader:
    options = {
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": compact_distillation_collate,
    }
    if num_workers:
        options.update(
            prefetch_factor=prefetch_factor,
            persistent_workers=training,
        )
    if training:
        options["batch_sampler"] = ReplayBatchSampler(dataset, batch_size, shuffle=True, seed=seed)
    else:
        options.update(batch_size=batch_size, shuffle=False)
    return DataLoader(dataset, **options)


def main() -> None:
    args = parse_args()
    learning_rate_was_explicit = args.learning_rate is not None
    if args.learning_rate is None:
        args.learning_rate = DEFAULT_LEARNING_RATE
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("Batch size and gradient accumulation must be positive")
    if args.temperature <= 0:
        raise ValueError("Temperature must be positive")
    if args.distill_weight < 0 or args.hard_label_weight < 0 or args.illegal_weight < 0:
        raise ValueError("Loss weights must be non-negative")
    if args.distill_weight + args.hard_label_weight + args.illegal_weight <= 0:
        raise ValueError("At least one loss weight must be positive")
    if args.learning_rate <= 0 or args.min_learning_rate <= 0:
        raise ValueError("Learning rates must be positive")
    if args.min_learning_rate > args.learning_rate:
        raise ValueError("--min-learning-rate cannot exceed --learning-rate")
    if not 0 < args.lr_decay_factor < 1:
        raise ValueError("--lr-decay-factor must be between 0 and 1")
    if args.lr_patience < 0 or args.lr_threshold < 0:
        raise ValueError("LR patience and threshold must be non-negative")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if args.num_workers < 0:
        worker_limit = 8 if args.prepared_cache_dir else 4
        num_workers = min(worker_limit, max(1, (os.cpu_count() or 2) // 2))
    else:
        num_workers = args.num_workers
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.amp_dtype]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = None
    source_checkpoint = args.student_checkpoint or DEFAULT_STUDENTS[args.encoder_type]
    if args.resume:
        model, checkpoint = load_bc_checkpoint(args.resume, str(device))
        source_checkpoint = checkpoint.get("source_student_checkpoint", source_checkpoint)
        if model.config.policy_schema != POLICY_SCHEMA_FIRST_PLACE_FLAT:
            raise ValueError("--resume must be a distilled flat-policy checkpoint")
        if model.config.encoder_type != args.encoder_type:
            message = f"Resume encoder is {model.config.encoder_type}, not {args.encoder_type}"
            raise ValueError(message)
        split = checkpoint["split"]
        start_epoch = int(checkpoint["epoch"]) + 1
    else:
        if source_checkpoint is None:
            config = ModelConfig(
                encoder_type=args.encoder_type,
                policy_schema=POLICY_SCHEMA_FIRST_PLACE_FLAT,
            )
            model = LuxBehaviorCloningModel(config)
        else:
            source_model, _ = load_bc_checkpoint(source_checkpoint, "cpu")
            if source_model.config.encoder_type != args.encoder_type:
                message = f"Student checkpoint encoder is {source_model.config.encoder_type}, not {args.encoder_type}"
                raise ValueError(message)
            config = replace(source_model.config, policy_schema=POLICY_SCHEMA_FIRST_PLACE_FLAT)
            model = LuxBehaviorCloningModel(config)
            model.encoder.load_state_dict(source_model.encoder.state_dict(), strict=True)
        replay_paths = discover_replays(args.replay_dir)
        path_split = split_replays(replay_paths, seed=args.seed)
        split = {name: [str(path) for path in paths] for name, paths in path_split.items()}
        start_epoch = 0
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)

    replay_cache_dir = Path(args.replay_cache_dir) if args.replay_cache_dir else output_dir.parent / "replay_cache"
    if not args.prepared_cache_dir:
        prepare_replay_cache(
            [Path(path) for paths in split.values() for path in paths],
            replay_cache_dir,
            num_workers=max(1, num_workers),
            show_progress=not args.no_progress,
        )
    datasets = {
        name: LuxDistillationDataset(
            [Path(path) for path in paths],
            Path(args.teacher_cache_dir),
            winner_weight=args.winner_weight,
            seed=args.seed,
            max_turns=args.max_turns,
            replay_cache_dir=replay_cache_dir,
            prepared_cache_dir=Path(args.prepared_cache_dir) if args.prepared_cache_dir else None,
            prepared_observation_dtype=args.prepared_observation_dtype,
        )
        for name, paths in split.items()
    }
    loaders = {
        name: make_loader(
            dataset,
            args.batch_size,
            training=name == "train",
            num_workers=num_workers,
            prefetch_factor=args.prefetch_factor,
            seed=args.seed,
        )
        for name, dataset in datasets.items()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    if checkpoint is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if learning_rate_was_explicit:
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate
                group["initial_lr"] = args.learning_rate
        else:
            args.learning_rate = float(optimizer.param_groups[0]["lr"])
    scheduler = make_lr_scheduler(
        optimizer,
        args.lr_scheduler,
        factor=args.lr_decay_factor,
        patience=args.lr_patience,
        threshold=args.lr_threshold,
        min_learning_rate=args.min_learning_rate,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and amp_dtype == torch.float16)
    if checkpoint is not None and checkpoint.get("scaler") is not None and scaler.is_enabled():
        scaler.load_state_dict(checkpoint["scaler"])
    metrics_path = output_dir / "metrics.json"
    history = load_existing_history(metrics_path, start_epoch, model.config) if checkpoint else []
    validation_losses = [float(item["validation"]["loss"]) for item in history]
    if checkpoint:
        validation_losses.append(float(checkpoint["metrics"]["validation"]["loss"]))
    best_loss = min(validation_losses, default=float("inf"))
    restore_lr_scheduler(scheduler, checkpoint, None if not validation_losses else best_loss)
    scheduler_config = {
        "name": args.lr_scheduler,
        "factor": args.lr_decay_factor,
        "patience": args.lr_patience,
        "threshold": args.lr_threshold,
        "threshold_mode": "abs",
        "min_learning_rate": args.min_learning_rate,
    }
    extra_metadata = {
        "teacher_sha256": FIRST_PLACE_TEACHER_SHA256,
        "data_split_signature": data_split_signature(split),
        "class_statistics_signature": None,
        "source_student_checkpoint": source_checkpoint,
        "inference_augmentation": "rot180",
        "distillation_config": {
            "temperature": args.temperature,
            "distill_weight": args.distill_weight,
            "hard_label_weight": args.hard_label_weight,
            "illegal_weight": args.illegal_weight,
        },
        "lr_scheduler_config": scheduler_config,
    }
    epochs = tqdm(
        range(start_epoch, args.epochs),
        desc="Epochs",
        unit="epoch",
        disable=args.no_progress,
        dynamic_ncols=True,
    )
    for epoch in epochs:
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        train = run_epoch(
            model,
            loaders["train"],
            device,
            temperature=args.temperature,
            distill_weight=args.distill_weight,
            hard_label_weight=args.hard_label_weight,
            illegal_weight=args.illegal_weight,
            amp_dtype=amp_dtype,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip=args.gradient_clip,
            accumulation_steps=args.gradient_accumulation_steps,
            description=f"Train {epoch + 1}/{args.epochs}",
            show_progress=not args.no_progress,
        )
        with torch.inference_mode():
            validation = run_epoch(
                model,
                loaders["validation"],
                device,
                temperature=args.temperature,
                distill_weight=args.distill_weight,
                hard_label_weight=args.hard_label_weight,
                illegal_weight=args.illegal_weight,
                amp_dtype=amp_dtype,
                description=f"Validation {epoch + 1}/{args.epochs}",
                show_progress=not args.no_progress,
            )
        if scheduler is not None:
            scheduler.step(validation["loss"])
        next_learning_rate = float(optimizer.param_groups[0]["lr"])
        metrics = {
            "epoch": epoch,
            "learning_rate": epoch_learning_rate,
            "next_learning_rate": next_learning_rate,
            "train": train,
            "validation": validation,
        }
        history.append(metrics)
        checkpoint_metadata = {
            **extra_metadata,
            "lr_scheduler_state": None if scheduler is None else scheduler.state_dict(),
        }
        save_bc_checkpoint(
            output_dir / "latest.pt",
            model,
            optimizer,
            epoch,
            metrics,
            split,
            scaler=scaler,
            extra_metadata=checkpoint_metadata,
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            save_bc_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                metrics,
                split,
                scaler=scaler,
                extra_metadata=checkpoint_metadata,
            )
        print(json.dumps(metrics, sort_keys=True))
        epochs.set_postfix(train=f"{train['loss']:.4f}", validation=f"{validation['loss']:.4f}")
    with torch.inference_mode():
        test = run_epoch(
            model,
            loaders["test"],
            device,
            temperature=args.temperature,
            distill_weight=args.distill_weight,
            hard_label_weight=args.hard_label_weight,
            illegal_weight=args.illegal_weight,
            amp_dtype=amp_dtype,
            description="Test",
            show_progress=not args.no_progress,
        )
    summary = {
        "model_config": asdict(model.config),
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "encoder_parameter_count": sum(parameter.numel() for parameter in model.encoder.parameters()),
        "device": str(device),
        "amp_dtype": args.amp_dtype,
        "teacher_cache_dir": args.teacher_cache_dir,
        "action_schema": {name: list(actions) for name, actions in FIRST_PLACE_ACTION_SCHEMA.items()},
        "training_config": vars(args),
        "history": history,
        "test": test,
        **extra_metadata,
    }
    metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"test": test}, sort_keys=True))


if __name__ == "__main__":
    main()
