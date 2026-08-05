import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "modeling" / "output"
CONFIG_DIR = PROJECT_ROOT / "modeling" / "config"
READINESS_SCORING = PROJECT_ROOT / "modeling" / "support" / "readiness_scoring.py"
PUBLIC_MANIFEST = (
    PROJECT_ROOT
    / "nightlight-public"
    / "src"
    / "content"
    / "evidencePassportManifest.json"
)


def _score_observation(row: pd.Series, rules: list[dict[str, float | int]]) -> int:
    observed_rate = float(row["observed_rate_v2"])
    high_censoring_share = float(row["high_censoring_share"])
    for rule in rules:
        if observed_rate < float(rule["observed_rate_min"]):
            continue
        censoring_limit = rule.get("high_censoring_max")
        if censoring_limit is not None and high_censoring_share > float(censoring_limit):
            continue
        return int(rule["score"])
    raise AssertionError("Observation-quality rules do not contain a fallback.")


def _load_readiness_scoring_module():
    spec = importlib.util.spec_from_file_location("_readiness_scoring", READINESS_SCORING)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zero_high_censoring_share_remains_a_valid_zero():
    module = _load_readiness_scoring_module()

    assert module.score_observation(0.995, 0.0) == 30


def test_local_private_readiness_components_match_versioned_observation_rules():
    audit_path = OUTPUT_DIR / "target_quality_audit_stage_10_dorian_freeport.csv"
    components_path = OUTPUT_DIR / "event_readiness_components_v1.csv"
    if not audit_path.exists() or not components_path.exists():
        pytest.skip("private readiness outputs are unavailable in this clone")

    rules = json.loads(
        (CONFIG_DIR / "readiness_score_rules_v1.json").read_text(encoding="utf-8")
    )["obs_quality_score"]["rules"]
    audit = pd.read_csv(audit_path).set_index("event_id")
    components = pd.read_csv(components_path).set_index("event_id")

    expected = {
        event_id: _score_observation(row, rules)
        for event_id, row in audit.iterrows()
    }
    actual = components["obs_quality_score"].astype(int).to_dict()

    assert actual == expected


def test_public_readiness_bands_are_recomputable_from_reviewed_components():
    rules = json.loads(
        (CONFIG_DIR / "readiness_score_rules_v1.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(PUBLIC_MANIFEST.read_text(encoding="utf-8"))
    for passport in manifest["passports"]:
        total = sum(passport["componentPoints"].values())
        expected_band = (
            "mainline_ready"
            if total >= rules["bands"]["mainline_ready_min"]
            else "sensitivity_only"
            if total >= rules["bands"]["sensitivity_only_min"]
            else "repair_first"
        )
        assert passport["readinessBand"] == expected_band
