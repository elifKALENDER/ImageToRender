#!/usr/bin/env python3
"""ImageToRender — single entry point.

Examples:
    python run.py --structure s01_test_church                     # full chain, both views
    python run.py --structure s01_test_church --view restoration
    python run.py --structure s01_test_church --from evidence     # reuse earlier steps of the latest run
    python run.py --structure s01_test_church --no-review         # non-interactive

See docs/ARCHITECTURE.md for what each step does.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from datetime import datetime
from typing import Callable

from i2r import runio
from i2r.runio import STEPS, VIEWS, ContractError, Run


# Each step is a module in i2r/ exposing run_step(run, opts). Modules are imported only when the
# step runs, so a later step never pays for loading an earlier step's models. Steps not yet
# implemented (tasks T13 to T21) only log their name.
STEP_MODULES: dict[str, str] = {
    "detect": "i2r.detect",
    "review": "i2r.review",
}


def _stub(name: str) -> Callable[[Run, argparse.Namespace], None]:
    def step(run: Run, opts: argparse.Namespace) -> None:
        run.logger.info("[%s] not implemented yet", name)

    step.__name__ = f"stub_{name}"
    return step


def step_function(name: str) -> Callable[[Run, argparse.Namespace], None]:
    if name in STEP_MODULES:
        return importlib.import_module(STEP_MODULES[name]).run_step
    return _stub(name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ImageToRender: evidence-based completion of damaged historical facades."
    )
    parser.add_argument("--structure", required=True, help="folder name under data/structures/")
    parser.add_argument(
        "--view",
        choices=[*VIEWS, "both"],
        default="both",
        help="which output view to produce (default: both)",
    )
    parser.add_argument(
        "--from",
        dest="from_step",
        choices=STEPS,
        default=STEPS[0],
        help="start at this step, reusing the earlier steps' files from the latest run",
    )
    parser.add_argument(
        "--no-review", action="store_true", help="skip interactive expert review (tests, evaluation)"
    )
    parser.add_argument("--config", default=str(runio.DEFAULT_CONFIG), help="settings file")
    opts = parser.parse_args(argv)
    opts.views = list(VIEWS) if opts.view == "both" else [opts.view]
    return opts


def main(argv: list[str] | None = None) -> int:
    opts = parse_args(argv)

    try:
        resume_from = None
        if opts.from_step != STEPS[0]:
            cfg = runio.load_config(opts.config)
            resume_from = runio.latest_run(cfg, opts.structure)
        run = runio.create_run(opts.structure, opts.config, resume_from=resume_from)
    except ContractError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    log = run.logger
    log.info("=" * 72)
    log.info("ImageToRender run %s", run.meta["run_id"])
    log.info("Views: %s | start: %s | review: %s",
             ", ".join(opts.views), opts.from_step, "off" if opts.no_review else "on")
    if resume_from is not None:
        log.info("Reusing files from %s", runio.relative(resume_from))
    log.info("=" * 72)

    start_index = STEPS.index(opts.from_step)
    status = "completed"
    try:
        for name in STEPS[start_index:]:
            t0 = time.perf_counter()
            log.info("--- step: %s", name)
            step_function(name)(run, opts)
            run.meta["steps"][name] = {"status": "done", "seconds": round(time.perf_counter() - t0, 2)}
            runio.save_meta(run)
    except ContractError as exc:
        status = "failed"
        log.error("%s", exc)
    except Exception:
        status = "failed"
        log.error("Unexpected error:\n%s", traceback.format_exc())
    finally:
        run.meta["status"] = status
        run.meta["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        runio.save_meta(run)
        log.info("Run %s: %s", status, runio.relative(run.run_dir))
        runio.close_logger(run)

    return 0 if status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
