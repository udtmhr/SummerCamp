from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


def model_label(model_path: str) -> str:
    path = Path(model_path)
    parent = path.parent.name
    label = f"{parent}_{path.stem}" if parent else path.stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", label)


def rename_replay(replay_file: str, winner_name: str) -> Path:
    source = Path(replay_file)
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_winner-{model_label(winner_name)}"
    target = source.with_name(f"{stem}.json")
    suffix = 2
    while target.exists():
        target = source.with_name(f"{stem}_{suffix}.json")
        suffix += 1

    source.rename(target)
    replay = json.loads(target.read_text(encoding="utf-8"))
    replay["results"]["replayFile"] = str(target)
    target.write_text(json.dumps(replay), encoding="utf-8")
    return target
