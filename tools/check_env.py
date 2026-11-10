#!/usr/bin/env python3
"""Check that the current interpreter is the pinned ImageToRender environment.

Usage:
    python tools/check_env.py

Exit code 0 when torch imports and every pinned package has the expected version,
1 otherwise. Prints the interpreter path so a wrong environment is obvious.
"""
from __future__ import annotations

import sys
from importlib import metadata

# Pinned versions of the conda env `ImageToRender_GPU`. Do not change these here
# without changing the environment backup and setup/ files as well.
PINNED: dict[str, str] = {
    "torch": "2.4.0+cu121",
    "xformers": "0.0.27.post2",
    "diffusers": "0.20.2",
    "transformers": "4.33.2",
    "huggingface-hub": "0.16.4",
    "accelerate": "0.22.0",
    "numpy": "1.26.4",
    "opencv-python": "4.8.1.78",
}

EXPECTED_PYTHON = (3, 10)


def installed_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def main() -> int:
    problems: list[str] = []

    print(f"Interpreter : {sys.executable}")
    print(f"Python      : {sys.version.split()[0]}")
    if sys.version_info[:2] != EXPECTED_PYTHON:
        problems.append(
            f"Python {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]} expected, "
            f"found {sys.version_info[0]}.{sys.version_info[1]}"
        )

    try:
        import torch  # noqa: F401  (import is the check)

        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else "none"
        print(f"CUDA        : {cuda} ({gpu})")
    except Exception as exc:  # report any import failure, then continue checking versions
        problems.append(f"torch cannot be imported: {exc}")
        print("CUDA        : unknown (torch not importable)")

    print()
    width = max(len(name) for name in PINNED)
    for package, expected in PINNED.items():
        found = installed_version(package)
        ok = found == expected
        mark = "ok " if ok else "BAD"
        print(f"  [{mark}] {package:<{width}}  expected {expected:<14} found {found or 'not installed'}")
        if not ok:
            problems.append(f"{package}: expected {expected}, found {found or 'not installed'}")

    print()
    if problems:
        print("Environment check FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        print("Activate the right env first:  conda activate ImageToRender_GPU")
        return 1

    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
