from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from luxai2021.imitation.actions import ACTION_SCHEMA, TARGET_NAMES
from luxai2021.imitation.masking import LEGAL_MASK_SCHEMA_VERSION
from luxai2021.imitation.schema import FEATURE_SCHEMA_VERSION

CLASS_STATISTICS_SCHEMA_VERSION = 2


def class_statistics_signature(
    replay_paths: Sequence[str | Path],
    *,
    team_selection: str,
    max_turns: int,
    source_ids: Sequence[int] = (),
) -> str:
    files = []
    seen_paths = set()
    for replay_path in replay_paths:
        path = Path(replay_path).resolve()
        agent_info_path = next(
            (
                parent / "agent_info.json"
                for parent in (path.parent, *path.parents)
                if (parent / "agent_info.json").exists()
            ),
            None,
        )
        related_paths = (path, path.with_name(f"{path.stem}_info.json"), agent_info_path)
        for related in related_paths:
            if related is not None and related.exists() and related not in seen_paths:
                seen_paths.add(related)
                stat = related.stat()
                files.append(
                    {
                        "path": str(related),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
    payload = {
        "schema_version": CLASS_STATISTICS_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "legal_mask_schema_version": LEGAL_MASK_SCHEMA_VERSION,
        "action_schema": ACTION_SCHEMA,
        "team_selection": team_selection,
        "max_turns": max_turns,
        "source_ids": [int(source_id) for source_id in source_ids],
        "files": files,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def _valid_counts(counts: object) -> bool:
    if not isinstance(counts, Mapping) or set(counts) != set(TARGET_NAMES):
        return False
    return all(
        isinstance(counts[name], torch.Tensor) and counts[name].numel() == len(ACTION_SCHEMA[name])
        for name in TARGET_NAMES
    )


def checkpoint_class_statistics(
    checkpoint: Mapping[str, object] | None,
    signature: str,
) -> dict[str, torch.Tensor] | None:
    if checkpoint is None or checkpoint.get("class_statistics_signature") != signature:
        return None
    counts = checkpoint.get("class_counts")
    if not _valid_counts(counts):
        return None
    return {name: counts[name].detach().cpu() for name in TARGET_NAMES}


def load_class_statistics(path: Path, signature: str) -> dict[str, torch.Tensor] | None:
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError):
        return None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != CLASS_STATISTICS_SCHEMA_VERSION
        or payload.get("signature") != signature
        or not _valid_counts(payload.get("counts"))
    ):
        return None
    counts = payload["counts"]
    return {name: counts[name].detach().cpu() for name in TARGET_NAMES}


def save_class_statistics(
    path: Path,
    signature: str,
    counts: Mapping[str, torch.Tensor],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(
        {
            "schema_version": CLASS_STATISTICS_SCHEMA_VERSION,
            "signature": signature,
            "counts": {name: counts[name].detach().cpu() for name in TARGET_NAMES},
        },
        temporary_path,
    )
    temporary_path.replace(path)
