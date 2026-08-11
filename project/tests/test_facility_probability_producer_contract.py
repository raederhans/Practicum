import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "modeling"
    / "config"
    / "facility_probability_producer_contract_v1.json"
)
MODULE_PATH = (
    PROJECT_ROOT / "modeling" / "support" / "facility_probability_contract.py"
)
ASSESSMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "facility_probability_legacy_v0_assessment_v1.json"
)
DASHBOARD_DATA = PROJECT_ROOT / "nightlight-dashboard" / "public" / "data"
PANEL_PATH = PROJECT_ROOT / "data" / "result" / "stage2" / "pixel_panel.parquet"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_facility_probability_contract", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _assessment():
    return json.loads(ASSESSMENT_PATH.read_text(encoding="utf-8"))


def _artifact(probability):
    contract = _contract()
    return {
        "schemaVersion": "1.0.0",
        "artifactType": "nightlight-facility-probabilities",
        "source": {
            "producer": "modelD-facility-probability-export",
            "producerVersion": "producer-fixture-v1",
            "producerReceipt": "receipt:producer-fixture-v1",
            "model": "modelD-rf-xgb-ensemble",
            "modelVersion": "modelD-fixture-v1",
            "modelReceipt": "receipt:modelD-fixture-v1",
            "inputArtifact": "modelD-event-probability-map",
            "inputVersion": "probability-map-fixture-v1",
            "inputReceipt": "receipt:probability-map-fixture-v1",
        },
        "provenance": {
            "generatedAtUtc": "2026-08-11T00:00:00Z",
            "eventId": "fixture-event",
            "facilityCatalogVersion": "facility-fixture-v1",
            "facilityCatalogReceipt": "receipt:facility-fixture-v1",
            "facilityTypeMatchRule": contract["quantity"]["facilityPixelRule"],
            "bufferRuleVersion": "strict-buffer-fixture-v1",
            "aggregationMethod": contract["quantity"]["aggregationMethod"],
        },
        "records": [
            {
                "facilityId": "facility-1",
                "name": "Fixture Facility",
                "type": "hospital",
                "coordinates": [-80.1, 25.8],
                "radiusM": 1000,
                "probability": probability,
            }
        ],
    }


def _probability(*, value, status, reason, eligible, finite, **extra):
    contract = _contract()
    provenance = {
        "eligiblePixelCount": eligible,
        "finiteProbabilityCount": finite,
        "aggregationMethod": contract["quantity"]["aggregationMethod"],
    }
    provenance.update(extra)
    return {
        "value": value,
        "status": status,
        "reason": reason,
        "provenance": provenance,
    }


def test_contract_freezes_states_reasons_provenance_and_legacy_window():
    module = _load_module()
    contract = _contract()

    module.validate_contract(contract)
    assert set(contract["recordContract"]["statuses"]) == {
        "available",
        "unavailable",
        "not_assessed",
        "computation_failed",
        "validation_failed",
    }
    taxonomy = contract["recordContract"]["reasonTaxonomy"]
    for reason in (
        "no_eligible_pixels_in_facility_type_buffer",
        "all_eligible_probabilities_missing",
        "pixel_probability_computation_failed",
        "facility_outside_assessment_scope",
        "source_version_unverified",
    ):
        assert reason in taxonomy
    requirements = contract["artifactRequirements"]
    for field in (
        "producerVersion",
        "producerReceipt",
        "modelVersion",
        "modelReceipt",
        "inputVersion",
        "inputReceipt",
    ):
        assert field in requirements["sourceRequiredFields"]
    assert "facilityCatalogReceipt" in requirements["provenanceRequiredFields"]
    legacy = contract["legacyCompatibility"]
    assert legacy["legacySchema"] == "legacy-v0"
    assert legacy["window"]["dualReadEnds"]
    assert legacy["window"]["legacyReaderRetirementEarliest"]
    assert legacy["window"]["calendarDeadline"] is None


def test_true_model_point_five_is_available_only_with_conforming_provenance_round_trip():
    module = _load_module()
    contract = _contract()
    artifact = _artifact(
        _probability(
            value=0.5,
            status="available",
            reason=None,
            eligible=8,
            finite=8,
        )
    )

    payload = module.dumps_artifact(artifact, contract)
    parsed = module.loads_artifact(payload, contract)

    assert parsed == artifact
    assert parsed["records"][0]["probability"] == {
        "value": 0.5,
        "status": "available",
        "reason": None,
        "provenance": {
            "eligiblePixelCount": 8,
            "finiteProbabilityCount": 8,
            "aggregationMethod": "arithmetic_mean_of_finite_pixel_probabilities",
        },
    }


@pytest.mark.parametrize(
    "probability",
    [
        _probability(
            value=None,
            status="unavailable",
            reason="no_eligible_pixels_in_facility_type_buffer",
            eligible=0,
            finite=0,
        ),
        _probability(
            value=None,
            status="unavailable",
            reason="all_eligible_probabilities_missing",
            eligible=6,
            finite=0,
        ),
        _probability(
            value=None,
            status="not_assessed",
            reason="facility_outside_assessment_scope",
            eligible=None,
            finite=None,
        ),
        _probability(
            value=None,
            status="computation_failed",
            reason="facility_aggregation_failed",
            eligible=6,
            finite=6,
            failureStage="facility-buffer-mean",
        ),
        _probability(
            value=None,
            status="validation_failed",
            reason="source_version_unverified",
            eligible=None,
            finite=None,
            validationErrors=["inputVersion is not linked to a source receipt"],
        ),
    ],
)
def test_all_nonavailable_states_use_null_and_a_status_bound_reason(probability):
    module = _load_module()
    module.validate_artifact(_artifact(probability), _contract())


def test_negative_cases_reject_numeric_sentinels_bad_counts_and_reason_guessing():
    module = _load_module()
    contract = _contract()
    cases = []

    sentinel = _probability(
        value=0.5,
        status="unavailable",
        reason="no_eligible_pixels_in_facility_type_buffer",
        eligible=0,
        finite=0,
    )
    cases.append((sentinel, "nonavailable-value-must-be-null"))

    fake_available = _probability(
        value=0.5,
        status="available",
        reason=None,
        eligible=0,
        finite=0,
    )
    cases.append((fake_available, "available-requires-finite-pixels"))

    mismatched_reason = _probability(
        value=None,
        status="not_assessed",
        reason="no_eligible_pixels_in_facility_type_buffer",
        eligible=0,
        finite=0,
    )
    cases.append((mismatched_reason, "probability-reason-status-mismatch"))

    partial_counts = _probability(
        value=None,
        status="computation_failed",
        reason="facility_aggregation_failed",
        eligible=4,
        finite=None,
        failureStage="facility-buffer-mean",
    )
    cases.append((partial_counts, "probability-counts-partially-null"))

    for probability, error in cases:
        with pytest.raises(module.FacilityProbabilityContractError, match=error):
            module.validate_artifact(_artifact(probability), contract)


def test_negative_cases_reject_legacy_shape_unpinned_source_and_nonfinite_value():
    module = _load_module()
    contract = _contract()

    with pytest.raises(
        module.FacilityProbabilityContractError, match="artifact-root-must-be-object"
    ):
        module.loads_artifact("[]", contract)

    unpinned = _artifact(
        _probability(
            value=0.4,
            status="available",
            reason=None,
            eligible=2,
            finite=2,
        )
    )
    unpinned["source"]["modelVersion"] = "unknown"
    with pytest.raises(
        module.FacilityProbabilityContractError, match="source-version-unpinned"
    ):
        module.validate_artifact(unpinned, contract)

    bad_timestamp = copy.deepcopy(unpinned)
    bad_timestamp["source"]["modelVersion"] = "modelD-fixture-v1"
    bad_timestamp["provenance"]["generatedAtUtc"] = "2026-08-11"
    with pytest.raises(
        module.FacilityProbabilityContractError,
        match="artifact-generated-at-utc-invalid",
    ):
        module.validate_artifact(bad_timestamp, contract)

    nonfinite = copy.deepcopy(unpinned)
    nonfinite["source"]["modelVersion"] = "modelD-fixture-v1"
    nonfinite["records"][0]["probability"]["value"] = np.nan
    with pytest.raises(
        module.FacilityProbabilityContractError, match="available-value-must-be-finite"
    ):
        module.validate_artifact(nonfinite, contract)


def test_legacy_v0_assessment_matches_tracked_artifacts_without_reclassifying_point_five():
    assessment = _assessment()
    paths = sorted(DASHBOARD_DATA.glob("facilities_*.json"))
    artifacts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    records = [record for artifact in artifacts for record in artifact]

    assert len(paths) == assessment["scope"]["artifactCount"] == 25
    assert all(isinstance(artifact, list) for artifact in artifacts)
    assert len(records) == assessment["scope"]["recordCount"] == 6225
    assert (
        sum(record.get("probability") == 0.5 for record in records)
        == assessment["scope"]["numericPointFiveCount"]
        == 10
    )
    assert all("schemaVersion" not in record for record in records)
    assert all("status" not in record and "reason" not in record for record in records)
    assert assessment["limitation"]["recoverability"].startswith(
        "Record-level status and reason cannot be recovered"
    )
    assert assessment["migration"]["silentValueRewriteAllowed"] is False
    assert assessment["migration"]["consumerInferenceAllowed"] is False


def test_fillna_assessment_is_bounded_to_the_tracked_panel_and_remains_blocked():
    assessment = _assessment()["preprocessingAssessment"]
    panel = pd.read_parquet(PANEL_PATH)
    engineered = panel.copy()
    engineered["drop_magnitude"] = -engineered["delta_ntl"].clip(upper=0)
    engineered["log_pre_ntl"] = np.log1p(engineered["pre_mean_ntl"])
    engineered["log_post_ntl"] = np.log1p(engineered["post_mean_ntl"])
    engineered["log_city_pre_mean"] = np.log1p(engineered["city_pre_mean"])
    engineered["ntl_relative"] = engineered["pre_mean_ntl"] / (
        engineered["city_pre_mean"] + 1e-6
    )
    city_median = engineered.groupby("event_id")["pre_mean_ntl"].transform(
        "median"
    )
    engineered["below_city_median"] = (
        engineered["pre_mean_ntl"] < city_median
    ).astype(np.uint8)
    engineered["city_size_code"] = engineered["city_size"].map(
        {"large": 0, "medium": 1, "small": 2}
    )
    engineered["is_hurricane"] = (
        engineered["disaster_type"] == "hurricane"
    ).astype(np.uint8)
    engineered["is_earthquake"] = (
        engineered["disaster_type"] == "earthquake"
    ).astype(np.uint8)

    audit = assessment["trackedPanelAudit"]
    assert len(engineered) == audit["rows"] == 61903
    assert engineered["event_id"].nunique() == audit["events"] == 25
    assert int(engineered[audit["sourceColumnsChecked"]].isna().sum().sum()) == 0
    assert int(engineered[audit["engineeredFeaturesChecked"]].isna().sum().sum()) == 0
    assert audit["rowsWithAnyCheckedFeatureMissing"] == 0
    assert assessment["decision"].endswith("scientific-admissibility-blocked")
    assert assessment["currentAction"].startswith("Preserve existing code and values")


def test_fillna_implementation_evidence_is_present_but_not_a_scientific_receipt():
    assessment = _assessment()["preprocessingAssessment"]

    for evidence in assessment["implementationEvidence"]:
        source = (PROJECT_ROOT.parent / evidence["path"]).read_text(encoding="utf-8")
        assert ".fillna(0)" in source
    assert len(assessment["blockedBecause"]) >= 4
    assert len(assessment["evidenceRequiredToUnblock"]) >= 4
