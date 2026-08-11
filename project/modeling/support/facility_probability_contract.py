"""Dependency-light validation for versioned facility probability artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any


class FacilityProbabilityContractError(ValueError):
    """Raised when a producer contract or artifact is semantically invalid."""


EXPECTED_SCHEMA_VERSION = "1.0.0"
EXPECTED_ARTIFACT_TYPE = "nightlight-facility-probabilities"
EXPECTED_CONTRACT_STATUS = "contract-frozen-producer-not-yet-migrated"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _require_mapping(value: Any, error: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FacilityProbabilityContractError(error)
    return value


def _require_fields(value: Mapping[str, Any], fields: list[str], prefix: str) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise FacilityProbabilityContractError(
            f"{prefix}-missing-fields:{','.join(sorted(missing))}"
        )


def _require_nonempty_string(value: Any, error: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FacilityProbabilityContractError(error)
    return value


def _require_utc_timestamp(value: Any, error: str) -> None:
    text = _require_nonempty_string(value, error)
    if not text.endswith("Z"):
        raise FacilityProbabilityContractError(error)
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise FacilityProbabilityContractError(error) from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise FacilityProbabilityContractError(error)


def validate_contract(contract: Mapping[str, Any]) -> None:
    """Validate the frozen producer contract before validating wire artifacts."""

    if contract.get("schemaVersion") != EXPECTED_SCHEMA_VERSION:
        raise FacilityProbabilityContractError("unsupported-contract-schema")
    if contract.get("artifactType") != EXPECTED_ARTIFACT_TYPE:
        raise FacilityProbabilityContractError("artifact-type-invalid")
    if contract.get("status") != EXPECTED_CONTRACT_STATUS:
        raise FacilityProbabilityContractError("contract-status-invalid")
    if contract.get("semanticClass") != "probability":
        raise FacilityProbabilityContractError("semantic-class-must-be-probability")

    quantity = _require_mapping(contract.get("quantity"), "quantity-missing")
    if quantity.get("range") != [0.0, 1.0]:
        raise FacilityProbabilityContractError("probability-range-invalid")
    aggregation = _require_nonempty_string(
        quantity.get("aggregationMethod"), "aggregation-method-missing"
    )

    artifact_requirements = _require_mapping(
        contract.get("artifactRequirements"), "artifact-requirements-missing"
    )
    for field_name in (
        "requiredFields",
        "sourceRequiredFields",
        "provenanceRequiredFields",
    ):
        fields = artifact_requirements.get(field_name)
        if not isinstance(fields, list) or not fields or not all(
            isinstance(field, str) and field for field in fields
        ):
            raise FacilityProbabilityContractError(
                f"artifact-requirement-invalid:{field_name}"
            )

    record_contract = _require_mapping(
        contract.get("recordContract"), "record-contract-missing"
    )
    statuses = _require_mapping(record_contract.get("statuses"), "statuses-missing")
    expected_statuses = {
        "available",
        "unavailable",
        "not_assessed",
        "computation_failed",
        "validation_failed",
    }
    if set(statuses) != expected_statuses:
        raise FacilityProbabilityContractError("status-dictionary-incomplete")

    taxonomy = _require_mapping(
        record_contract.get("reasonTaxonomy"), "reason-taxonomy-missing"
    )
    represented_statuses = set()
    for reason, definition in taxonomy.items():
        _require_nonempty_string(reason, "reason-code-invalid")
        definition = _require_mapping(definition, f"reason-definition-invalid:{reason}")
        status = definition.get("status")
        if status not in expected_statuses - {"available"}:
            raise FacilityProbabilityContractError(f"reason-status-invalid:{reason}")
        represented_statuses.add(status)
    if represented_statuses != expected_statuses - {"available"}:
        raise FacilityProbabilityContractError("reason-taxonomy-status-coverage-incomplete")

    legacy = _require_mapping(
        contract.get("legacyCompatibility"), "legacy-compatibility-missing"
    )
    if legacy.get("legacySchema") != "legacy-v0":
        raise FacilityProbabilityContractError("legacy-schema-invalid")
    limitation = _require_nonempty_string(
        legacy.get("limitationCode"), "legacy-limitation-missing"
    )
    if "0.5" not in limitation or "fallback" not in limitation:
        raise FacilityProbabilityContractError("legacy-limitation-insufficient")
    window = _require_mapping(legacy.get("window"), "legacy-window-missing")
    for field in (
        "dualReadStarts",
        "dualReadEnds",
        "legacyReaderRetirementEarliest",
    ):
        _require_nonempty_string(window.get(field), f"legacy-window-field-missing:{field}")

    if aggregation != "arithmetic_mean_of_finite_pixel_probabilities":
        raise FacilityProbabilityContractError("aggregation-method-not-frozen")


def _validate_counts(
    eligible: Any,
    finite: Any,
    *,
    allow_null: bool,
) -> None:
    if eligible is None or finite is None:
        if allow_null and eligible is None and finite is None:
            return
        raise FacilityProbabilityContractError("probability-counts-partially-null")
    if not _is_count(eligible) or not _is_count(finite):
        raise FacilityProbabilityContractError("probability-count-invalid")
    if finite > eligible:
        raise FacilityProbabilityContractError("finite-count-exceeds-eligible-count")


def _validate_probability(
    probability: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    record_contract = contract["recordContract"]
    _require_fields(
        probability,
        record_contract["probabilityRequiredFields"],
        "probability",
    )
    provenance = _require_mapping(
        probability.get("provenance"), "probability-provenance-missing"
    )
    _require_fields(
        provenance,
        record_contract["probabilityProvenanceRequiredFields"],
        "probability-provenance",
    )

    expected_aggregation = contract["quantity"]["aggregationMethod"]
    if provenance.get("aggregationMethod") != expected_aggregation:
        raise FacilityProbabilityContractError("record-aggregation-method-mismatch")

    eligible = provenance.get("eligiblePixelCount")
    finite = provenance.get("finiteProbabilityCount")
    status = probability.get("status")
    value = probability.get("value")
    reason = probability.get("reason")

    if status not in record_contract["statuses"]:
        raise FacilityProbabilityContractError(f"probability-status-invalid:{status}")

    if status == "available":
        if reason is not None:
            raise FacilityProbabilityContractError("available-reason-must-be-null")
        if not _is_number(value) or not math.isfinite(float(value)):
            raise FacilityProbabilityContractError("available-value-must-be-finite")
        if not 0.0 <= float(value) <= 1.0:
            raise FacilityProbabilityContractError("available-value-out-of-range")
        _validate_counts(eligible, finite, allow_null=False)
        if finite < 1:
            raise FacilityProbabilityContractError("available-requires-finite-pixels")
        return

    if value is not None:
        raise FacilityProbabilityContractError("nonavailable-value-must-be-null")
    if reason not in record_contract["reasonTaxonomy"]:
        raise FacilityProbabilityContractError(f"probability-reason-invalid:{reason}")
    reason_definition = record_contract["reasonTaxonomy"][reason]
    if reason_definition.get("status") != status:
        raise FacilityProbabilityContractError("probability-reason-status-mismatch")

    if reason == "no_eligible_pixels_in_facility_type_buffer":
        _validate_counts(eligible, finite, allow_null=False)
        if eligible != 0 or finite != 0:
            raise FacilityProbabilityContractError("no-eligible-pixels-count-mismatch")
    elif reason == "all_eligible_probabilities_missing":
        _validate_counts(eligible, finite, allow_null=False)
        if eligible < 1 or finite != 0:
            raise FacilityProbabilityContractError("all-probabilities-missing-count-mismatch")
    elif reason in {
        "source_probability_pixels_unavailable",
        "facility_outside_assessment_scope",
        "required_facility_metadata_missing",
    }:
        _validate_counts(eligible, finite, allow_null=True)
        if eligible is not None or finite is not None:
            raise FacilityProbabilityContractError("reason-requires-null-counts")
    else:
        _validate_counts(eligible, finite, allow_null=True)

    if status == "computation_failed":
        _require_nonempty_string(
            provenance.get("failureStage"), "computation-failure-stage-missing"
        )
    if status == "validation_failed":
        errors = provenance.get("validationErrors")
        if not isinstance(errors, list) or not errors or not all(
            isinstance(error, str) and error for error in errors
        ):
            raise FacilityProbabilityContractError("validation-errors-missing")


def _validate_record(record: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    record_contract = contract["recordContract"]
    _require_fields(record, record_contract["requiredFields"], "record")
    _require_nonempty_string(record.get("facilityId"), "facility-id-invalid")
    _require_nonempty_string(record.get("name"), "facility-name-invalid")
    _require_nonempty_string(record.get("type"), "facility-type-invalid")

    coordinates = record.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) != 2:
        raise FacilityProbabilityContractError("facility-coordinates-invalid")
    lon, lat = coordinates
    if not _is_number(lon) or not _is_number(lat):
        raise FacilityProbabilityContractError("facility-coordinates-not-numeric")
    if not math.isfinite(float(lon)) or not math.isfinite(float(lat)):
        raise FacilityProbabilityContractError("facility-coordinates-nonfinite")
    if not -180 <= float(lon) <= 180 or not -90 <= float(lat) <= 90:
        raise FacilityProbabilityContractError("facility-coordinates-out-of-range")

    radius = record.get("radiusM")
    if not _is_number(radius) or not math.isfinite(float(radius)) or radius <= 0:
        raise FacilityProbabilityContractError("facility-radius-invalid")
    probability = _require_mapping(
        record.get("probability"), "probability-observation-missing"
    )
    _validate_probability(probability, contract)


def validate_artifact(
    artifact: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    """Validate one v1 artifact, including negative-state provenance."""

    validate_contract(contract)
    requirements = contract["artifactRequirements"]
    _require_fields(artifact, requirements["requiredFields"], "artifact")
    if artifact.get("schemaVersion") != EXPECTED_SCHEMA_VERSION:
        raise FacilityProbabilityContractError("artifact-schema-version-mismatch")
    if artifact.get("artifactType") != EXPECTED_ARTIFACT_TYPE:
        raise FacilityProbabilityContractError("artifact-type-mismatch")

    source = _require_mapping(artifact.get("source"), "artifact-source-missing")
    _require_fields(source, requirements["sourceRequiredFields"], "source")
    forbidden_versions = {
        value.casefold() for value in requirements["forbiddenVersionValues"]
    }
    for field in requirements["sourceRequiredFields"]:
        text = _require_nonempty_string(source.get(field), f"source-field-invalid:{field}")
        if field.endswith("Version") and text.strip().casefold() in forbidden_versions:
            raise FacilityProbabilityContractError(f"source-version-unpinned:{field}")

    provenance = _require_mapping(
        artifact.get("provenance"), "artifact-provenance-missing"
    )
    _require_fields(
        provenance, requirements["provenanceRequiredFields"], "artifact-provenance"
    )
    for field in requirements["provenanceRequiredFields"]:
        _require_nonempty_string(
            provenance.get(field), f"artifact-provenance-field-invalid:{field}"
        )
    _require_utc_timestamp(
        provenance.get("generatedAtUtc"), "artifact-generated-at-utc-invalid"
    )
    if provenance.get("aggregationMethod") != contract["quantity"]["aggregationMethod"]:
        raise FacilityProbabilityContractError("artifact-aggregation-method-mismatch")
    if provenance.get("facilityTypeMatchRule") != contract["quantity"]["facilityPixelRule"]:
        raise FacilityProbabilityContractError("facility-type-match-rule-mismatch")

    records = artifact.get("records")
    if not isinstance(records, list):
        raise FacilityProbabilityContractError("artifact-records-must-be-list")
    seen_ids: set[str] = set()
    for record in records:
        record = _require_mapping(record, "artifact-record-invalid")
        _validate_record(record, contract)
        facility_id = record["facilityId"]
        if facility_id in seen_ids:
            raise FacilityProbabilityContractError(
                f"duplicate-facility-id:{facility_id}"
            )
        seen_ids.add(facility_id)


def dumps_artifact(artifact: Mapping[str, Any], contract: Mapping[str, Any]) -> str:
    """Serialize a conforming artifact without permitting NaN or Infinity."""

    validate_artifact(artifact, contract)
    return json.dumps(
        artifact,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def loads_artifact(payload: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Parse and validate a v1 artifact, rejecting legacy arrays and bad states."""

    artifact = json.loads(payload)
    if not isinstance(artifact, dict):
        raise FacilityProbabilityContractError("artifact-root-must-be-object")
    validate_artifact(artifact, contract)
    return artifact
