import copy
import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    PROJECT_ROOT
    / "modeling"
    / "config"
    / "evidence_passport_composite_research_v1.json"
)
MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "evidence_passport_composite_sensitivity_v1.json"
)
MODULE_PATH = PROJECT_ROOT / "modeling" / "support" / "composite_sensitivity.py"
PUBLIC_MANIFEST_PATH = (
    PROJECT_ROOT
    / "nightlight-public"
    / "src"
    / "content"
    / "evidencePassportManifest.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("_composite_sensitivity", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _protocol():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_snapshot_matches_bounded_public_component_source_without_modifying_it():
    manifest = _manifest()
    public = json.loads(PUBLIC_MANIFEST_PATH.read_text(encoding="utf-8"))

    expected_rows = {
        passport["eventId"]: passport["componentPoints"]
        for passport in public["passports"]
    }
    actual_rows = {
        row["event_id"]: row["components"] for row in manifest["rows"]
    }
    expected_maxima = {
        definition["id"]: definition["maxPoints"]
        for definition in public["componentDefinitions"]
    }
    assert actual_rows == expected_rows
    assert manifest["component_maxima"] == expected_maxima
    assert manifest["snapshot_scope"]["public_artifact_modified"] is False


def test_research_protocol_preserves_public_no_score_no_rank_no_outcome_boundary():
    module = _load_module()
    protocol = _protocol()

    module.validate_protocol(protocol)
    assert set(protocol["public_boundary"].values()) >= {False}
    for gate in (
        "export_to_public_allowed",
        "score_allowed",
        "rank_allowed",
        "outcome_label_allowed",
        "forecast_or_probability_allowed",
    ):
        assert protocol["public_boundary"][gate] is False


def test_analysis_is_reproducible_and_truthfully_no_go():
    module = _load_module()
    protocol = _protocol()
    manifest = _manifest()

    actual = module.analyze_composite_sensitivity(manifest, protocol)

    assert actual == manifest["analysis"]
    assert actual["decision"] == "no_go"
    assert actual["decision_reasons"] == [
        "complete-event-count-below-minimum",
        "monte-carlo-weight-rank-instability",
    ]
    assert actual["complete_event_count"] == 9
    assert actual["diagnostics"]["maximum_monte_carlo_rank_span"] == 4.0
    assert actual["public_export_allowed"] is False


def test_analysis_covers_normalization_weighting_loo_and_monte_carlo():
    analysis = _manifest()["analysis"]

    assert set(analysis["scenarios"]) == {
        "fixed_component_max__component_max_proportional",
        "fixed_component_max__equal_components",
        "cohort_min_max__equal_components",
    }
    assert set(analysis["leave_one_component_out"]) == set(
        _manifest()["component_maxima"]
    )
    assert analysis["monte_carlo"]["draws"] == 2000
    assert analysis["monte_carlo"]["seed"] == 20260811
    assert analysis["monte_carlo"]["events"]["laura"]["maximum_rank"] - analysis[
        "monte_carlo"
    ]["events"]["laura"]["minimum_rank"] == 4.0


def test_schema_window_and_missingness_gates_fail_closed():
    module = _load_module()
    protocol = _protocol()
    manifest = _manifest()

    wrong_family = copy.deepcopy(manifest)
    wrong_family["window_family"] = "other-window"
    with pytest.raises(module.CompositeSensitivityError, match="window-family-mismatch"):
        module.analyze_composite_sensitivity(wrong_family, protocol)

    one_missing = copy.deepcopy(manifest)
    one_missing["rows"][0]["components"]["observation-quality"] = None
    result = module.analyze_composite_sensitivity(one_missing, protocol)
    assert result["complete_event_count"] == 8
    assert result["excluded_events"] == ["eq-pr"]
    assert "complete-event-count-below-minimum" in result["decision_reasons"]
