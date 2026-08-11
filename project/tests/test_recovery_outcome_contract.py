import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    PROJECT_ROOT / "modeling" / "config" / "recovery_outcome_contract_v1.json"
)
MODULE_PATH = PROJECT_ROOT / "modeling" / "support" / "recovery_outcomes.py"
README_PATH = PROJECT_ROOT / "modeling" / "README.md"


def _load_module():
    spec = importlib.util.spec_from_file_location("_recovery_outcomes", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_contract_freezes_distinct_semantic_classes_and_prohibited_claims():
    module = _load_module()
    contract = _contract()

    module.validate_contract(contract)
    assert set(contract["semantic_classes"]) == {
        "admission_readiness",
        "observed_recovery_outcome",
        "forecast",
        "probability",
    }
    assert contract["status"] == "construct-frozen-not-empirically-admitted"
    assert contract["observation_policy"]["zero_is_observed"] is True
    assert contract["observation_policy"]["missing_is_zero"] is False
    assert contract["observation_policy"]["missing_is_censoring"] is False
    prohibited = " ".join(contract["prohibited_interpretations"]).lower()
    for phrase in ("electricity service", "forecast", "probability", "causality"):
        assert phrase in prohibited


def test_measure_ids_cannot_be_relabelled_across_semantic_classes():
    module = _load_module()
    contract = _contract()

    valid_records = (
        {"semantic_class": "admission_readiness", "measure_id": "readiness_band"},
        {"semantic_class": "observed_recovery_outcome", "measure_id": "t90_days"},
        {"semantic_class": "forecast", "measure_id": "recovery_time_forecast"},
        {
            "semantic_class": "probability",
            "measure_id": "recovery_threshold_probability",
        },
    )
    for record in valid_records:
        module.validate_artifact_record(record, contract)

    with pytest.raises(module.RecoveryContractError, match="measure-class-mismatch"):
        module.validate_artifact_record(
            {"semantic_class": "admission_readiness", "measure_id": "t90_days"},
            contract,
        )
    with pytest.raises(module.RecoveryContractError, match="measure-class-mismatch"):
        module.validate_artifact_record(
            {
                "semantic_class": "observed_recovery_outcome",
                "measure_id": "readiness_band",
            },
            contract,
        )


def test_observed_outcomes_keep_zero_missing_and_censoring_distinct():
    module = _load_module()
    contract = _contract()
    observations = []
    for day in range(1, 91):
        if day == 2:
            observations.append(
                {"day": day, "normalized_signal": None, "quality_ok": False}
            )
        elif day == 1:
            observations.append(
                {"day": day, "normalized_signal": 0.0, "quality_ok": True}
            )
        else:
            observations.append(
                {
                    "day": day,
                    "normalized_signal": 0.95 if day >= 20 else 0.6,
                    "quality_ok": True,
                }
            )

    result = module.derive_observed_recovery_outcomes(observations, contract)

    assert result["observed_fraction"] == pytest.approx(89 / 90)
    assert result["t50_days"] == {"status": "observed", "value": 3}
    assert result["t90_days"] == {"status": "observed", "value": 20}
    assert result["deficit_burden_observed_day_sum"]["status"] == "observed"
    assert result["deficit_burden_observed_day_sum"]["value"] > 0


def test_threshold_is_right_censored_only_with_sufficient_horizon_followup():
    module = _load_module()
    contract = _contract()
    observations = [
        {"day": day, "normalized_signal": 0.4, "quality_ok": True}
        for day in range(1, 91)
    ]

    result = module.derive_observed_recovery_outcomes(observations, contract)

    assert result["t50_days"]["status"] == "right_censored"
    assert result["t90_days"]["status"] == "right_censored"
    assert result["t90_days"]["lower_bound_days"] == 90


def test_insufficient_coverage_is_unavailable_not_zero_or_censored():
    module = _load_module()
    contract = _contract()
    observations = [
        {"day": day, "normalized_signal": 0.95, "quality_ok": True}
        for day in range(1, 31)
    ]

    result = module.derive_observed_recovery_outcomes(observations, contract)

    for outcome_id in (
        "t50_days",
        "t90_days",
        "deficit_burden_observed_day_sum",
    ):
        assert result[outcome_id]["status"] == "unavailable"
        assert result[outcome_id]["value"] is None


def test_existing_metric_and_passport_names_remain_intact():
    readme = README_PATH.read_text(encoding="utf-8")

    for existing_name in ("AUC", "readiness scoring"):
        assert existing_name in readme
    assert "recovery_days" in readme
    assert "Evidence Passport" in readme
