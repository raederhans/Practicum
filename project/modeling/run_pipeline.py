#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPRODUCIBILITY_MANIFEST = (
    ROOT / "project" / "data" / "manifests" / "reproducibility_inputs_v1.json"
)


def _load_main(rel_path: str):
    target = Path(__file__).resolve().parent / rel_path
    spec = importlib.util.spec_from_file_location(target.stem.replace('-', '_'), target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.main


def _load_reproducibility_module():
    target = Path(__file__).resolve().parent / "reproducibility.py"
    spec = importlib.util.spec_from_file_location("practicum_reproducibility", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    if not forwarded:
        forwarded = ["full-run"]

    reproducibility = _load_reproducibility_module()
    if forwarded[0] == "verify-inputs":
        return reproducibility.main(forwarded[1:])

    report = reproducibility.validate_scope(
        root=ROOT,
        manifest_path=REPRODUCIBILITY_MANIFEST,
        scope="full-upstream",
    )
    if report["status"] != "ready":
        reproducibility.print_report(report)
        return 2

    pipeline_main = _load_main("pipelines/01_in_sample_pipeline.py")
    return pipeline_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
