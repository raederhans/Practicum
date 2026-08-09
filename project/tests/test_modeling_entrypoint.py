from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = ROOT / "project" / "modeling" / "run_pipeline.py"


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("practicum_modeling_entrypoint", ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert hasattr(module, "main"), "the entrypoint needs a testable deterministic main()"
    return module


@pytest.mark.parametrize(
    ("argv", "expected_forwarded"),
    [
        ([], ["full-run"]),
        (["full-run"], ["full-run"]),
        (["strict-v2"], ["strict-v2"]),
    ],
)
def test_modeling_command_is_forwarded_once_after_full_upstream_preflight(
    argv, expected_forwarded
) -> None:
    entrypoint = _load_entrypoint()
    calls: list[tuple[str, object]] = []

    reproducibility = SimpleNamespace(
        validate_scope=lambda **kwargs: calls.append(("preflight", kwargs["scope"]))
        or {"status": "ready"},
        print_report=lambda report: calls.append(("report", report)),
    )

    def pipeline_main(forwarded):
        calls.append(("pipeline", forwarded))
        return 0

    entrypoint._load_reproducibility_module = lambda: reproducibility
    entrypoint._load_main = lambda relative_path: pipeline_main

    assert entrypoint.main(argv) == 0
    assert calls == [
        ("preflight", "full-upstream"),
        ("pipeline", expected_forwarded),
    ]


def test_full_run_does_not_start_when_input_preflight_is_blocked() -> None:
    entrypoint = _load_entrypoint()
    calls: list[tuple[str, object]] = []
    blocked_report = {"status": "blocked", "blockers": [{"code": "checksum-mismatch"}]}
    reproducibility = SimpleNamespace(
        validate_scope=lambda **kwargs: blocked_report,
        print_report=lambda report: calls.append(("report", report)),
    )
    entrypoint._load_reproducibility_module = lambda: reproducibility
    entrypoint._load_main = lambda relative_path: pytest.fail("pipeline must not load")

    assert entrypoint.main(["full-run"]) == 2
    assert calls == [("report", blocked_report)]


def test_verify_inputs_routes_without_loading_the_modeling_stack() -> None:
    entrypoint = _load_entrypoint()
    calls: list[list[str]] = []
    reproducibility = SimpleNamespace(
        main=lambda forwarded: calls.append(forwarded) or 1,
    )
    entrypoint._load_reproducibility_module = lambda: reproducibility
    entrypoint._load_main = lambda relative_path: pytest.fail("modeling stack must not load")

    result = entrypoint.main(["verify-inputs", "--scope", "full-upstream", "--json"])

    assert result == 1
    assert calls == [["--scope", "full-upstream", "--json"]]
