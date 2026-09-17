"""Check or install the official pretrained FlyVis data used by this project."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import zipfile


ARCHIVE_NAME = "results_pretrained_models.zip"
ARCHIVE_SHA256 = "71c78d4070556a536b13b23ee3139cd2788aa2a9d07d430a223b4edead281db1"
MODEL_RELATIVE = Path("results/flow/0000/000/best_chkpt")
MODEL_SHA256 = "d8a57a022aa18ca338599be713af9db29b89cde4b00098ffe42b78f0d583e46a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(root: Path, verbose: bool = True) -> bool:
    archive = root / ARCHIVE_NAME
    model = root / MODEL_RELATIVE
    checks = (
        (archive, ARCHIVE_SHA256, "official pretrained archive"),
        (model, MODEL_SHA256, "flow/0000/000 best checkpoint"),
    )
    valid = True
    for path, expected, label in checks:
        if not path.is_file():
            if verbose:
                print(f"MISSING: {label}: {path}")
            valid = False
            continue
        actual = sha256(path)
        if actual != expected:
            if verbose:
                print(f"CHECKSUM MISMATCH: {label}: {actual}")
            valid = False
        elif verbose:
            print(f"OK: {label}")
    return valid


def safe_extract(archive: Path, root: Path) -> None:
    root_resolved = root.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            destination = (root / member.filename).resolve()
            if root_resolved != destination and root_resolved not in destination.parents:
                raise RuntimeError(f"unsafe ZIP member: {member.filename}")
        bundle.extractall(root)


def ensure(root: Path) -> int:
    if validate(root, verbose=False):
        print("FlyVis pretrained data is already installed and verified.")
        return 0

    archive = root / ARCHIVE_NAME
    if archive.is_file() and sha256(archive) == ARCHIVE_SHA256:
        print("Verified archive found; extracting it.")
        safe_extract(archive, root)
    else:
        print("Downloading with FlyVis's official download-pretrained command.")
        env = os.environ.copy()
        env["FLYVIS_ROOT_DIR"] = str(root.resolve())
        subprocess.run(
            [sys.executable, "-m", "flyvis_cli.download_pretrained_models",
             "--skip_large_files"],
            check=True,
            env=env,
        )
    if not validate(root):
        raise RuntimeError("FlyVis download/extraction finished but validation failed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/flyvis"))
    parser.add_argument(
        "--ensure", action="store_true",
        help="download or extract missing data, then verify it",
    )
    args = parser.parse_args(argv)
    try:
        if args.ensure:
            return ensure(args.root)
        return 0 if validate(args.root) else 1
    except (OSError, RuntimeError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
