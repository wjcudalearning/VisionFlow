"""Validate direct requirements, the complete lock, and a local build environment."""

from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = (3, 13)
PIN_PATTERN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;#]+)\s*(?:#.*)?$")


def canonicalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = PIN_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"{path}:{line_number}: expected an exact name==version pin")
        name, version = match.groups()
        canonical_name = canonicalize(name)
        if canonical_name in pins:
            raise ValueError(f"{path}:{line_number}: duplicate requirement for {name}")
        pins[canonical_name] = version
    return pins


def check_lock_consistency(
    requirements_path: Path,
    lock_path: Path,
    *,
    installed: dict[str, str] | None = None,
    python_version: tuple[int, int] | None = None,
) -> list[str]:
    requirements = read_pins(requirements_path)
    lock = read_pins(lock_path)
    problems: list[str] = []
    for name, version in sorted(requirements.items()):
        locked_version = lock.get(name)
        if locked_version is None:
            problems.append(f"{name} is pinned in requirements.txt but missing from requirements.lock.txt")
        elif locked_version != version:
            problems.append(
                f"{name} differs between requirements.txt ({version}) and requirements.lock.txt "
                f"({locked_version})"
            )

    if python_version is not None and python_version != EXPECTED_PYTHON:
        problems.append(
            f"Python {python_version[0]}.{python_version[1]} is active; "
            f"the lock targets Python {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}"
        )

    if installed is not None:
        normalized_installed = {canonicalize(name): version for name, version in installed.items()}
        for name, expected_version in sorted(lock.items()):
            actual_version = normalized_installed.get(name)
            if actual_version is None:
                problems.append(f"{name}=={expected_version} is missing from the build environment")
            elif actual_version != expected_version:
                problems.append(
                    f"{name} has {actual_version} in the build environment; "
                    f"requirements.lock.txt pins {expected_version}"
                )
    return problems


def installed_versions() -> dict[str, str]:
    return {
        canonicalize(distribution.metadata["Name"]): distribution.version
        for distribution in metadata.distributions()
        if distribution.metadata.get("Name")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--check-environment",
        action="store_true",
        help="also require Python 3.13 and every locked package at its pinned version",
    )
    args = parser.parse_args()

    requirements_path = args.repo_root / "requirements.txt"
    lock_path = args.repo_root / "requirements.lock.txt"
    try:
        problems = check_lock_consistency(
            requirements_path,
            lock_path,
            installed=installed_versions() if args.check_environment else None,
            python_version=sys.version_info[:2] if args.check_environment else None,
        )
    except (OSError, ValueError) as exc:
        print(f"Requirements validation failed: {exc}", file=sys.stderr)
        return 1

    if problems:
        print("Build environment does not match requirements.lock.txt:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print("requirements.txt agrees with requirements.lock.txt")
    if args.check_environment:
        print("Build Python and all locked package versions match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
