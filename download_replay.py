from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import kagglehub

DATASET_HANDLE = "bomac1/luxai-replay-dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Lux AI Season 1 replay dataset.")
    parser.add_argument(
        "--dataset-file",
        help="Download only this path inside the Kaggle dataset instead of the complete dataset.",
    )
    parser.add_argument("--force", action="store_true", help="Download again even when KaggleHub has a cached copy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    downloaded_path = Path(
        kagglehub.dataset_download(
            DATASET_HANDLE,
            path=args.dataset_file,
            force_download=args.force,
        )
    ).resolve()
    replay_root = downloaded_path if downloaded_path.is_dir() else downloaded_path.parent
    replay_files = [path for path in replay_root.rglob("*.json") if not path.stem.endswith("_info")]

    print(f"Downloaded dataset: {downloaded_path}")
    print(f"JSON replay candidates: {len(replay_files)}")
    print(
        "Train with: uv run python examples/train_bc.py "
        f"--replay-dir {shlex.quote(str(replay_root))} --output-dir models/bc"
    )


if __name__ == "__main__":
    main()
