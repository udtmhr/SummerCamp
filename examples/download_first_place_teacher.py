from __future__ import annotations

# ruff: noqa: INP001
import argparse
import hashlib
import urllib.request
from pathlib import Path

from luxai2021.imitation.first_place import FIRST_PLACE_TEACHER_SHA256, FIRST_PLACE_UPSTREAM_COMMIT

DEFAULT_URL = (
    "https://media.githubusercontent.com/media/IsaiahPressman/Kaggle_Lux_AI_2021/"
    f"{FIRST_PLACE_UPSTREAM_COMMIT}/"
    "internal_testing/hall_of_fame/11-24_12-56-23_062179520_must_research/"
    "lux_ai/rl_agent/062179520_weights.pt"
)

GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Lux AI 2021 first-place teacher checkpoint.")
    parser.add_argument("--output", default="models/teachers/lux_2021_first_place/062179520_weights.pt")
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and sha256_file(output) == FIRST_PLACE_TEACHER_SHA256:
        print(f"Teacher already verified: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.download")
    urllib.request.urlretrieve(args.url, temporary)  # noqa: S310 - explicit upstream model URL
    digest = sha256_file(temporary)
    if digest != FIRST_PLACE_TEACHER_SHA256:
        with temporary.open("rb") as source:
            is_lfs_pointer = source.read(len(GIT_LFS_POINTER_PREFIX)) == GIT_LFS_POINTER_PREFIX
        temporary.unlink(missing_ok=True)
        if is_lfs_pointer:
            message = (
                "Downloaded a Git LFS pointer instead of the teacher checkpoint. "
                "Use the default media.githubusercontent.com URL or another URL that serves the LFS object."
            )
            raise ValueError(message)
        message = f"Teacher SHA-256 mismatch: expected={FIRST_PLACE_TEACHER_SHA256} actual={digest}"
        raise ValueError(message)
    temporary.replace(output)
    print(f"Saved verified teacher: {output}")


if __name__ == "__main__":
    main()
