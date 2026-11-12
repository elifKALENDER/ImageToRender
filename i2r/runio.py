"""Run folders, configuration, JSON I/O with key checks, and logging.

Every pipeline step reads its inputs from, and writes its outputs to, one run folder:
``outputs/<structure_id>/<timestamp>/``. Steps never search for "the newest folder" and never
swallow parsing errors (docs/ARCHITECTURE.md §2, D9).
"""
from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"

STEPS: tuple[str, ...] = ("detect", "review", "damage", "evidence", "complete", "render")
VIEWS: tuple[str, ...] = ("restoration", "first_construction")

# Packages whose versions are written to run_meta.json.
TRACKED_PACKAGES: tuple[str, ...] = (
    "torch",
    "xformers",
    "diffusers",
    "transformers",
    "huggingface-hub",
    "accelerate",
    "numpy",
    "opencv-python",
    "Pillow",
    "PyYAML",
)


class ContractError(RuntimeError):
    """A file a step depends on is missing, empty or lacks a required key."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ContractError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ContractError(f"Config file is empty or not a mapping: {path}")
    require_keys(cfg, ("paths", "classes"), where=str(path))
    return cfg


STRUCTURE_CONFIG = "structure.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``override`` applied; nested mappings are merged, other values replaced."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_structure_config(cfg: dict[str, Any], folder: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge the building's optional structure.yaml over the main config.

    Returns (effective config, the overrides that were applied).
    """
    path = folder / STRUCTURE_CONFIG
    if not path.is_file():
        return cfg, {}
    with path.open(encoding="utf-8") as fh:
        override = yaml.safe_load(fh) or {}
    if not isinstance(override, dict):
        raise ContractError(f"{path}: expected a mapping")
    unknown = [key for key in override if key not in cfg]
    if unknown:
        raise ContractError(f"{path}: unknown top-level section(s): {', '.join(unknown)}")
    return deep_merge(cfg, override), override


def apply_pixel_scale(cfg: dict[str, Any], long_side: int) -> dict[str, Any]:
    """Return ``cfg`` with the pixel settings listed in ``scale`` scaled to an image whose long side is
    ``long_side`` (config.yaml, section ``scale``). Lengths are multiplied by f = long_side / reference,
    areas by f^2; integers are rounded and kept >= 1 unless they are 0; lists are scaled element-wise.
    Records f and every original value under ``scale.applied``. Applying it twice is an error."""
    require_keys(cfg, ("scale",), "config")
    scfg = cfg["scale"]
    require_keys(scfg, ("reference_long_side_px", "length_keys", "area_keys"), "config: scale")
    if "applied" in scfg:
        raise ContractError("config: pixel scale already applied (scale.applied exists); use the main config.yaml")
    factor = float(long_side) / float(scfg["reference_long_side_px"])
    out = deep_merge(cfg, {})
    out["scale"] = dict(scfg)
    changed: dict[str, list[Any]] = {}

    def scaled(value: Any, power: int) -> Any:
        if isinstance(value, list):
            return [scaled(v, power) for v in value]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"config: scale applies to numbers only, found {value!r}")
        new = value * factor ** power
        if isinstance(value, int):
            return 0 if value == 0 else max(1, int(round(new)))
        return round(new, 4)

    for power, keys in ((1, scfg["length_keys"]), (2, scfg["area_keys"])):
        for dotted in keys:
            section, _, key = str(dotted).partition(".")
            if section not in out or not isinstance(out[section], dict) or key not in out[section]:
                raise ContractError(f"config: scale lists {dotted!r}, which is not a setting")
            out[section] = dict(out[section])
            old = out[section][key]
            out[section][key] = scaled(old, power)
            changed[dotted] = [old, out[section][key]]
    out["scale"]["applied"] = {"image_long_side_px": int(long_side), "factor": round(factor, 4), "values": changed}
    return out


def effective_config(config_path: Path | str, structure_id: str) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Main config + the building's structure.yaml + pixel scaling for its image.

    Returns (config, structure folder, structure overrides). Used by run.py and the tools, so a run, a
    synthetic case and an evaluation see the same settings."""
    cfg = load_config(config_path)
    sdir = structure_dir(cfg, structure_id)
    cfg, overrides = apply_structure_config(cfg, sdir)
    image = sdir / cfg["paths"]["image_file"]
    if image.is_file():
        from PIL import Image

        with Image.open(image) as im:
            cfg = apply_pixel_scale(cfg, max(im.size))
    return cfg, sdir, overrides


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def read_json(path: Path | str) -> Any:
    path = Path(path)
    if not path.is_file():
        raise ContractError(f"Required file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path | str, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return path


def require_keys(obj: Any, keys: Iterable[str], where: str) -> None:
    """Raise ContractError naming the file (``where``) and every missing key."""
    if not isinstance(obj, dict):
        raise ContractError(f"{where}: expected a JSON object, found {type(obj).__name__}")
    missing = [key for key in keys if key not in obj]
    if missing:
        raise ContractError(f"{where}: missing required key(s): {', '.join(missing)}")


# ---------------------------------------------------------------------------
# Structures and runs
# ---------------------------------------------------------------------------

@dataclass
class Run:
    structure_id: str
    structure_dir: Path
    run_dir: Path
    config: dict[str, Any]
    logger: logging.Logger
    meta: dict[str, Any] = field(default_factory=dict)

    def path(self, name: str) -> Path:
        """Path of a file inside the run folder."""
        return self.run_dir / name

    @property
    def image_path(self) -> Path:
        return self.structure_dir / self.config["paths"]["image_file"]

    @property
    def damage_mask_path(self) -> Path:
        return self.structure_dir / self.config["paths"]["damage_mask_file"]


def structure_dir(cfg: dict[str, Any], structure_id: str) -> Path:
    root = REPO_ROOT / cfg["paths"]["data_root"]
    folder = root / structure_id
    if not folder.is_dir():
        known = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        raise ContractError(
            f"Structure '{structure_id}' not found in {root}. Known: {', '.join(known) or 'none'}"
        )
    return folder


def structure_runs_root(cfg: dict[str, Any], structure_id: str) -> Path:
    return REPO_ROOT / cfg["paths"]["output_root"] / structure_id


def create_run(
    structure_id: str,
    config_path: Path | str = DEFAULT_CONFIG,
    *,
    resume_from: Path | None = None,
) -> Run:
    """Create a new run folder, copy the config into it and write run_meta.json.

    ``resume_from``: an earlier run folder of the same structure whose files are copied in, so
    later steps can start from them (``run.py --from``).
    """
    config_path = Path(config_path)
    cfg, sdir, overrides = effective_config(config_path, structure_id)

    runs_root = structure_runs_root(cfg, structure_id)
    runs_root.mkdir(parents=True, exist_ok=True)
    while True:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = runs_root / stamp
        try:
            run_dir.mkdir(exist_ok=False)
            break
        except FileExistsError:
            time.sleep(1.05)

    if resume_from is not None:
        for item in resume_from.iterdir():
            if item.name in ("config.yaml", "run_meta.json", "run.log"):
                continue
            target = run_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

    # The run folder gets the effective settings (main config + structure overrides).
    with (run_dir / "config.yaml").open("w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    logger = setup_logger(run_dir)

    meta = {
        "run_id": f"{structure_id}/{stamp}",
        "structure_id": structure_id,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "running",
        "resumed_from": relative(resume_from) if resume_from else None,
        "config_file": relative(config_path),
        "structure_overrides": overrides,
        "pixel_scale": cfg.get("scale", {}).get("applied"),
        "steps": {},
        "environment": environment_info(),
        "git_commit": git_commit(),
    }
    run = Run(structure_id, sdir, run_dir, cfg, logger, meta)
    save_meta(run)
    return run


def latest_run(cfg: dict[str, Any], structure_id: str, exclude: Path | None = None) -> Path:
    """Most recent run folder of one structure (used only by ``run.py --from``)."""
    root = structure_runs_root(cfg, structure_id)
    runs = sorted(
        (p for p in root.glob("*") if p.is_dir() and p != exclude),
        key=lambda p: p.name,
        reverse=True,
    ) if root.is_dir() else []
    if not runs:
        raise ContractError(f"No earlier run of '{structure_id}' in {root} to resume from")
    return runs[0]


def save_meta(run: Run) -> None:
    write_json(run.path("run_meta.json"), run.meta)


def relative(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Environment and logging
# ---------------------------------------------------------------------------

def environment_info() -> dict[str, Any]:
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def git_commit() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            )
            return out.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    return {
        "hash": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"i2r.{run_dir.parent.name}.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def close_logger(run: Run) -> None:
    for handler in list(run.logger.handlers):
        handler.close()
        run.logger.removeHandler(handler)
