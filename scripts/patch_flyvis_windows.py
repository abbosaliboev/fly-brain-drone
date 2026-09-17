"""Apply the two local Windows compatibility patches used by this project.

Pinned targets:
  * flyvis 1.2.0
  * datamate 1.0.0

The patch is idempotent and refuses unknown source layouts rather than
silently modifying a different package release.
"""
from __future__ import annotations

import argparse
import importlib.util
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys


NODE_OLD = "@dataclass\nclass Node:"
NODE_NEW = "@dataclass(slots=True)\nclass Node:"
EDGE_OLD = "@dataclass\nclass Edge:"
EDGE_NEW = "@dataclass(slots=True)\nclass Edge:"

H5_OLD = '''    val = np.asarray(val)
    try:
        f = h5.File(path, libver="latest", mode="w")
        if f["data"].dtype != val.dtype:
            raise ValueError()
        f["data"][...] = val
        f.swmr_mode = True
        assert f.swmr_mode
    except Exception:
        path.parent.mkdir(parents=True, exist_ok=True)
'''

H5_NEW = '''    val = np.asarray(val)
    f = None
    try:
        f = h5.File(path, libver="latest", mode="a")
        if f["data"].dtype != val.dtype:
            raise ValueError()
        f["data"][...] = val
        f.swmr_mode = True
        assert f.swmr_mode
    except Exception:
        # Windows cannot unlink an HDF5 file while its handle is still open.
        if f is not None:
            f.close()
        path.parent.mkdir(parents=True, exist_ok=True)
'''


def _package_root(package: str) -> Path:
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"{package} is not installed in this Python environment")
    return Path(spec.origin).resolve().parent


def _replace_known(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected one pristine pattern, found {count}; "
            "refusing to patch an unknown package layout"
        )
    return text.replace(old, new), True


def _patch_file(path: Path, replacements: tuple[tuple[str, str, str], ...],
                check: bool) -> bool:
    raw = path.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    changed = False
    for old, new, label in replacements:
        text, item_changed = _replace_known(text, old, new, label)
        changed = changed or item_changed
    if changed and not check:
        path.write_text(text.replace("\n", newline), encoding="utf-8", newline="")
    return changed


def patch(site_packages: Path | None = None, check: bool = False) -> int:
    if site_packages is None:
        for package, expected in (("flyvis", "1.2.0"), ("datamate", "1.0.0")):
            try:
                actual = version(package)
            except PackageNotFoundError as exc:
                raise RuntimeError(f"{package} is not installed") from exc
            if actual != expected:
                raise RuntimeError(
                    f"unsupported {package} version {actual}; expected {expected}"
                )
        flyvis_root = _package_root("flyvis")
        datamate_root = _package_root("datamate")
    else:
        flyvis_root = site_packages / "flyvis"
        datamate_root = site_packages / "datamate"

    targets = (
        (
            flyvis_root / "connectome" / "connectome.py",
            ((NODE_OLD, NODE_NEW, "FlyVis Node slots"),
             (EDGE_OLD, EDGE_NEW, "FlyVis Edge slots")),
        ),
        (
            datamate_root / "io.py",
            ((H5_OLD, H5_NEW, "datamate Windows HDF5 handle"),),
        ),
    )
    pending = []
    for path, replacements in targets:
        if not path.is_file():
            raise RuntimeError(f"required source file not found: {path}")
        if _patch_file(path, replacements, check=check):
            pending.append(str(path))

    if check:
        if pending:
            print("Compatibility patches are required:")
            print("\n".join(f"  {path}" for path in pending))
            return 1
        print("FlyVis Windows compatibility patches are already applied.")
        return 0

    if pending:
        print("Applied FlyVis Windows compatibility patches:")
        print("\n".join(f"  {path}" for path in pending))
    else:
        print("FlyVis Windows compatibility patches were already applied.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check without writing")
    parser.add_argument(
        "--site-packages", type=Path,
        help="explicit site-packages root (primarily for clean-copy validation)",
    )
    args = parser.parse_args(argv)
    try:
        return patch(args.site_packages, args.check)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
