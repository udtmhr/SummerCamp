from __future__ import annotations

# ruff: noqa: C901, INP001, PLR0913, PLR0915
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from itertools import combinations
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

if __package__:
    from examples.train_bc import (
        configure_device,
        data_split_signature,
        resolve_amp_dtype,
        resolve_device,
        resolve_num_workers,
        run_epoch,
    )
else:
    from train_bc import (
        configure_device,
        data_split_signature,
        resolve_amp_dtype,
        resolve_device,
        resolve_num_workers,
        run_epoch,
    )

from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default
from luxai2021.imitation.agent import BehaviorCloningAgent
from luxai2021.imitation.data import LuxReplayDataset
from luxai2021.imitation.model import load_bc_checkpoint, make_class_weights

if TYPE_CHECKING:
    from luxai2021.game.game import Game

_MIN_MATCH_CHECKPOINTS = 2


def parse_checkpoint(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        message = f"Expected NAME=CHECKPOINT, got {value!r}"
        raise ValueError(message)
    return name, Path(raw_path)


def parse_match_workers(value: str) -> str | int:
    if value == "auto":
        return value
    try:
        workers = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("match workers must be 'auto' or a positive integer") from error
    if workers <= 0:
        raise argparse.ArgumentTypeError("match workers must be 'auto' or a positive integer")
    return workers


def resolve_match_workers(requested: str | int, device: torch.device, seed_count: int) -> int:
    if seed_count <= 0:
        return 0
    workers = (2 if device.type == "cuda" else min(4, os.cpu_count() or 1)) if requested == "auto" else int(requested)
    return min(workers, seed_count)


def shard_match_seeds(seed_start: int, seed_count: int, worker_count: int) -> list[list[int]]:
    if seed_count < 0:
        raise ValueError("match seed count must be zero or greater")
    if worker_count < 0 or (seed_count > 0 and worker_count == 0):
        raise ValueError("match worker count must be positive when seeds are requested")
    seeds = list(range(seed_start, seed_start + seed_count))
    return [seeds[index::worker_count] for index in range(worker_count)] if worker_count else []


def expected_match_game_count(model_count: int, seed_count: int) -> int:
    return model_count * (model_count - 1) // 2 * seed_count * 2


def match_pair_orientations(model_count: int) -> list[tuple[int, int, int, int]]:
    assignments = []
    for pair_index, pair in enumerate(combinations(range(model_count), 2)):
        assignments.extend((pair_index, pair[0], pair[1], orientation) for orientation in (0, 1))
    return assignments


def sort_match_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        games,
        key=lambda game: (
            int(game["pair_index"]),
            int(game["seed"]),
            int(game["orientation"]),
        ),
    )


def _game_team_statistics(game: Game) -> dict[str, dict[str, int | float]]:
    city_tiles = {0: 0, 1: 0}
    for city in game.cities.values():
        city_tiles[city.team] += len(city.city_cells)
    return {
        str(team): {
            "city_tiles": city_tiles[team],
            "units": len(game.get_teams_units(team)),
            "fuel_generated": game.stats["teamStats"][team]["fuelGenerated"],
        }
        for team in (0, 1)
    }


def _winning_team(team_statistics: dict[str, dict[str, int | float]]) -> int | None:
    scores = {
        team: (
            team_statistics[str(team)]["city_tiles"],
            team_statistics[str(team)]["units"],
            team_statistics[str(team)]["fuel_generated"],
        )
        for team in (0, 1)
    }
    if scores[0] == scores[1]:
        return None
    return 0 if scores[0] > scores[1] else 1


def _run_orientation(
    *,
    worker_index: int,
    pair_index: int,
    pair: tuple[dict[str, str], dict[str, str]],
    orientation: int,
    seeds: list[int],
    device: str,
) -> list[dict[str, Any]]:
    first, second = pair if orientation == 0 else (pair[1], pair[0])
    try:
        first_agent = BehaviorCloningAgent(first["checkpoint"], device=device)
        second_agent = BehaviorCloningAgent(second["checkpoint"], device=device)
        config = dict(LuxMatchConfigs_Default)
        env = LuxEnvironment(config, first_agent, second_agent)
    except Exception as error:
        message = (
            f"match worker={worker_index} pair={pair[0]['name']} vs {pair[1]['name']} "
            f"seed={seeds[0]} orientation={orientation} failed while loading models"
        )
        raise RuntimeError(message) from error

    games = []
    try:
        for seed in seeds:
            started_at = time.perf_counter()
            try:
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                with suppress(StopIteration):
                    env.reset(seed=seed)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                team_models = {
                    str(first_agent.team): first["name"],
                    str(second_agent.team): second["name"],
                }
                team_statistics = _game_team_statistics(env.game)
                winning_team = _winning_team(team_statistics)
                games.append(
                    {
                        "pair_index": pair_index,
                        "pair": [pair[0]["name"], pair[1]["name"]],
                        "seed": seed,
                        "orientation": orientation,
                        "first_model": first["name"],
                        "second_model": second["name"],
                        "team_models": team_models,
                        "winner": None if winning_team is None else team_models[str(winning_team)],
                        "winner_team": winning_team,
                        "draw": winning_team is None,
                        "end_turn": int(env.game.state["turn"]),
                        "team_statistics": team_statistics,
                        "duration_seconds": time.perf_counter() - started_at,
                    }
                )
            except Exception as error:
                message = (
                    f"match worker={worker_index} pair={pair[0]['name']} vs {pair[1]['name']} "
                    f"seed={seed} orientation={orientation} failed"
                )
                raise RuntimeError(message) from error
    finally:
        del env, first_agent, second_agent
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return games


def _run_match_worker(
    worker_index: int,
    seeds: list[int],
    checkpoints: list[dict[str, str]],
    device: str,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    torch.set_num_threads(1)
    worker_games = []
    for pair_index, left_index, right_index, orientation in match_pair_orientations(len(checkpoints)):
        pair = (checkpoints[left_index], checkpoints[right_index])
        worker_games.extend(
            _run_orientation(
                worker_index=worker_index,
                pair_index=pair_index,
                pair=pair,
                orientation=orientation,
                seeds=seeds,
                device=device,
            )
        )
    return {
        "worker": worker_index,
        "seeds": seeds,
        "duration_seconds": time.perf_counter() - started_at,
        "games": worker_games,
    }


def aggregate_match_results(
    games: list[dict[str, Any]],
    test_losses: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_names = sorted(test_losses)
    totals = {name: {"games": 0, "wins": 0, "losses": 0, "draws": 0, "points": 0.0} for name in model_names}
    head_to_head_points = {(left, right): 0.0 for left in model_names for right in model_names if left != right}
    pair_totals: dict[tuple[str, str], dict[str, Any]] = {}

    for game in games:
        pair = tuple(game["pair"])
        pair_result = pair_totals.setdefault(
            pair,
            {
                "pair_index": int(game["pair_index"]),
                "models": list(pair),
                "games": 0,
                "draws": 0,
                "wins": {pair[0]: 0, pair[1]: 0},
                "points": {pair[0]: 0.0, pair[1]: 0.0},
            },
        )
        pair_result["games"] += 1
        for name in pair:
            totals[name]["games"] += 1
        if game["draw"]:
            pair_result["draws"] += 1
            for name in pair:
                totals[name]["draws"] += 1
                totals[name]["points"] += 0.5
                pair_result["points"][name] += 0.5
                head_to_head_points[(name, pair[0] if name == pair[1] else pair[1])] += 0.5
        else:
            winner = game["winner"]
            loser = pair[0] if winner == pair[1] else pair[1]
            totals[winner]["wins"] += 1
            totals[winner]["points"] += 1.0
            totals[loser]["losses"] += 1
            pair_result["wins"][winner] += 1
            pair_result["points"][winner] += 1.0
            head_to_head_points[(winner, loser)] += 1.0

    pairwise = []
    for pair_result in pair_totals.values():
        pair_result["score_rate"] = {
            name: pair_result["points"][name] / pair_result["games"] for name in pair_result["models"]
        }
        pairwise.append(pair_result)

    score_rates = {
        name: totals[name]["points"] / totals[name]["games"] if totals[name]["games"] else 0.0 for name in model_names
    }
    tied_names = {
        name: [other for other in model_names if other != name and score_rates[other] == score_rates[name]]
        for name in model_names
    }
    standings = []
    for name in model_names:
        direct_points = sum(head_to_head_points[(name, other)] for other in tied_names[name])
        direct_games = sum(
            head_to_head_points[(name, other)] + head_to_head_points[(other, name)] for other in tied_names[name]
        )
        standings.append(
            {
                "name": name,
                **totals[name],
                "score_rate": score_rates[name],
                "head_to_head_score_rate": direct_points / direct_games if direct_games else 0.0,
                "test_loss": test_losses[name],
            }
        )
    standings.sort(
        key=lambda row: (
            -row["score_rate"],
            -row["head_to_head_score_rate"],
            row["test_loss"],
            row["name"],
        )
    )
    for rank, row in enumerate(standings, start=1):
        row["rank"] = rank
    pairwise.sort(key=lambda row: (row["pair_index"], tuple(row["models"])))
    return pairwise, standings


def select_evaluation_winner(
    test_winner: str,
    match_evaluation: dict[str, Any] | None,
) -> tuple[str, str]:
    if match_evaluation is None:
        return "test_loss", test_winner
    return "round_robin_score_rate", match_evaluation["standings"][0]["name"]


def run_match_evaluation(
    named_paths: list[tuple[str, Path]],
    rows: list[dict[str, Any]],
    *,
    seed_start: int,
    seed_count: int,
    requested_workers: str | int,
    resolved_workers: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoints = [{"name": name, "checkpoint": str(path)} for name, path in named_paths]
    shards = shard_match_seeds(seed_start, seed_count, resolved_workers)
    started_at = time.perf_counter()
    worker_results = []
    multiprocessing_context = get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=resolved_workers, mp_context=multiprocessing_context)
    futures = []
    try:
        futures = [
            executor.submit(_run_match_worker, worker_index, seeds, checkpoints, str(device))
            for worker_index, seeds in enumerate(shards)
        ]
        worker_results = [future.result() for future in as_completed(futures)]
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown()

    worker_results.sort(key=lambda result: result["worker"])
    games = sort_match_games([game for worker_result in worker_results for game in worker_result["games"]])
    expected_games = expected_match_game_count(len(named_paths), seed_count)
    if len(games) != expected_games:
        message = f"Expected {expected_games} match results, received {len(games)}"
        raise RuntimeError(message)
    pairwise, standings = aggregate_match_results(
        games,
        {row["name"]: float(row["test"]["loss"]) for row in rows},
    )
    return {
        "seed_start": seed_start,
        "seed_count": seed_count,
        "requested_workers": requested_workers,
        "resolved_workers": resolved_workers,
        "start_method": "spawn",
        "games_per_pair": seed_count * 2,
        "total_games": len(games),
        "duration_seconds": time.perf_counter() - started_at,
        "worker_durations": [
            {
                "worker": result["worker"],
                "seeds": result["seeds"],
                "duration_seconds": result["duration_seconds"],
            }
            for result in worker_results
        ],
        "pairwise": pairwise,
        "standings": standings,
        "games": games,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BC checkpoints on one shared test split.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=CHECKPOINT",
        help="Named best checkpoint; provide once per model.",
    )
    parser.add_argument("--output", default="models/bc_encoder_compare/evaluation.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--class-weight-exponent", type=float, default=0.5)
    parser.add_argument("--match-seeds", type=int, default=0)
    parser.add_argument("--match-seed-start", type=int, default=0)
    parser.add_argument("--match-workers", type=parse_match_workers, default="auto")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def validate_checkpoint_metadata(
    name: str,
    checkpoint: dict[str, object],
    *,
    expected_split_signature: str | None,
    expected_class_statistics_signature: str | None,
) -> tuple[str, str | None]:
    split_signature = data_split_signature(checkpoint["split"])
    class_statistics_signature = checkpoint.get("class_statistics_signature")
    if expected_split_signature is not None and split_signature != expected_split_signature:
        message = f"Checkpoint {name} has a different test split"
        raise ValueError(message)
    if (
        expected_class_statistics_signature is not None
        and class_statistics_signature != expected_class_statistics_signature
    ):
        message = f"Checkpoint {name} has different class statistics"
        raise ValueError(message)
    if checkpoint.get("class_counts") is None:
        message = f"Checkpoint {name} does not contain class counts"
        raise ValueError(message)
    return split_signature, class_statistics_signature


def main() -> None:
    args = parse_args()
    if args.match_seeds < 0 or args.match_seed_start < 0:
        raise ValueError("Match seed count and start must be zero or greater")
    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(args.amp_dtype, device)
    configure_device(device)
    num_workers = resolve_num_workers(args.num_workers)
    named_paths = [parse_checkpoint(value) for value in args.checkpoint]
    if len({name for name, _ in named_paths}) != len(named_paths):
        raise ValueError("Checkpoint names must be unique")
    if args.match_seeds > 0 and len(named_paths) < _MIN_MATCH_CHECKPOINTS:
        raise ValueError("At least two checkpoints are required for match evaluation")

    expected_split_signature = None
    expected_class_statistics_signature = None
    rows = []
    for name, path in named_paths:
        model, checkpoint = load_bc_checkpoint(str(path), str(device))
        split_signature, class_statistics_signature = validate_checkpoint_metadata(
            name,
            checkpoint,
            expected_split_signature=expected_split_signature,
            expected_class_statistics_signature=expected_class_statistics_signature,
        )
        expected_split_signature = split_signature
        expected_class_statistics_signature = class_statistics_signature
        if device.type == "cuda":
            model.to(memory_format=torch.channels_last)
        dataset = LuxReplayDataset(
            [Path(replay_path) for replay_path in checkpoint["split"]["test"]],
            augment=False,
            team_selection="winner",
        )
        loader_options = {
            "batch_size": args.batch_size,
            "num_workers": num_workers,
            "pin_memory": device.type == "cuda",
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            loader_options["prefetch_factor"] = args.prefetch_factor
        loader = DataLoader(dataset, **loader_options)
        class_weights = make_class_weights(
            checkpoint["class_counts"],
            device,
            exponent=args.class_weight_exponent,
        )
        with torch.inference_mode():
            metrics = run_epoch(
                model,
                loader,
                device,
                class_weights,
                description=f"Evaluate {name}",
                show_progress=not args.no_progress,
                amp_dtype=amp_dtype,
            )
        rows.append(
            {
                "name": name,
                "checkpoint": str(path),
                "encoder_type": model.config.encoder_type,
                "epoch": int(checkpoint["epoch"]),
                "validation_loss": float(checkpoint["metrics"]["validation"]["loss"]),
                "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "test": metrics,
            }
        )
        del class_weights, dataset, loader, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows.sort(key=lambda row: (row["test"]["loss"], row["name"]))
    test_winner = rows[0]["name"]
    match_evaluation = None
    if args.match_seeds > 0:
        resolved_workers = resolve_match_workers(args.match_workers, device, args.match_seeds)
        match_evaluation = run_match_evaluation(
            named_paths,
            rows,
            seed_start=args.match_seed_start,
            seed_count=args.match_seeds,
            requested_workers=args.match_workers,
            resolved_workers=resolved_workers,
            device=device,
        )
    primary_metric, winner = select_evaluation_winner(test_winner, match_evaluation)
    result = {
        "primary_metric": primary_metric,
        "winner": winner,
        "test_winner": test_winner,
        "match_evaluation": match_evaluation,
        "device": str(device),
        "amp_dtype": args.amp_dtype,
        "batch_size": args.batch_size,
        "data_split_signature": expected_split_signature,
        "class_statistics_signature": expected_class_statistics_signature,
        "runs": rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print("name\tencoder\tepoch\tvalidation_loss\ttest_loss\tsamples/s\tpeak_GiB")
    for row in rows:
        test = row["test"]
        peak_bytes = test["peak_cuda_memory_allocated_bytes"]
        peak_gib = "-" if peak_bytes is None else f"{peak_bytes / 2**30:.2f}"
        print(
            f"{row['name']}\t{row['encoder_type']}\t{row['epoch']}\t"
            f"{row['validation_loss']:.6f}\t{test['loss']:.6f}\t"
            f"{test['samples_per_second']:.1f}\t{peak_gib}"
        )
    if result["match_evaluation"] is not None:
        print("rank\tname\tscore_rate\twins\tlosses\tdraws")
        for standing in result["match_evaluation"]["standings"]:
            print(
                f"{standing['rank']}\t{standing['name']}\t{standing['score_rate']:.4f}\t"
                f"{standing['wins']}\t{standing['losses']}\t{standing['draws']}"
            )
    print(f"Winner ({result['primary_metric']}): {result['winner']}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
