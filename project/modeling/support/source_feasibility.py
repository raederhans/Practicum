"""Validation helpers for recovery label source feasibility metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class SourceFeasibilityError(ValueError):
    """Raised when a source inventory overstates or omits a required gate."""


REQUIRED_SOURCE_FIELDS = {
    "id",
    "role",
    "source_identity",
    "grain",
    "publication_rights",
    "access",
    "receipts",
    "missingness",
    "rebuildability",
    "label_eligibility",
}

REQUIRED_SOURCE_IDS = {
    "nasa_vnp46a2",
    "eagle_i",
    "doe_oe417",
    "eia_861",
    "direct_utility_outage_maps",
}


def evaluate_label_pilot_gate(manifest: Mapping[str, Any]) -> dict[str, Any]:
    gate = manifest.get("label_pilot_gate")
    if not isinstance(gate, Mapping):
        raise SourceFeasibilityError("label-pilot-gate-missing")
    required = gate.get("required_gates")
    if not isinstance(required, Mapping) or not required:
        raise SourceFeasibilityError("required-gates-missing")
    invalid_values = [key for key, value in required.items() if not isinstance(value, bool)]
    if invalid_values:
        raise SourceFeasibilityError(
            f"required-gates-must-be-boolean:{','.join(sorted(invalid_values))}"
        )
    failed = sorted(key for key, value in required.items() if not value)
    decision = "admitted" if not failed else "blocked"
    return {"decision": decision, "failed_gates": failed}


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise SourceFeasibilityError("unsupported-manifest-schema")
    if manifest.get("status") not in {"admitted", "evidence-backed-blocked"}:
        raise SourceFeasibilityError("invalid-manifest-status")

    evidence_policy = manifest.get("evidence_policy")
    if not isinstance(evidence_policy, Mapping):
        raise SourceFeasibilityError("evidence-policy-missing")
    for invariant in (
        "external_data_downloaded",
        "credential_content_read",
        "raw_or_cache_bytes_added_to_git",
    ):
        if evidence_policy.get(invariant) is not False:
            raise SourceFeasibilityError(f"evidence-boundary-violated:{invariant}")

    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise SourceFeasibilityError("source-inventory-empty")
    ids: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            raise SourceFeasibilityError("source-entry-invalid")
        missing_fields = REQUIRED_SOURCE_FIELDS - set(source)
        if missing_fields:
            raise SourceFeasibilityError(
                f"source-fields-missing:{source.get('id')}:{','.join(sorted(missing_fields))}"
            )
        source_id = source.get("id")
        if not isinstance(source_id, str) or source_id in ids:
            raise SourceFeasibilityError(f"source-id-duplicate-or-invalid:{source_id}")
        ids.add(source_id)
        grain = source["grain"]
        if not all(key in grain for key in ("spatial", "temporal", "geography")):
            raise SourceFeasibilityError(f"grain-incomplete:{source_id}")
        missingness = source["missingness"]
        if not isinstance(missingness.get("zero_distinguishable_from_missing"), bool):
            raise SourceFeasibilityError(f"missingness-ambiguous:{source_id}")
        if source["rebuildability"].get("ready") is not True and not source[
            "rebuildability"
        ].get("blockers"):
            raise SourceFeasibilityError(f"rebuildability-blockers-missing:{source_id}")

    missing_sources = REQUIRED_SOURCE_IDS - ids
    if missing_sources:
        raise SourceFeasibilityError(
            f"required-sources-missing:{','.join(sorted(missing_sources))}"
        )

    evaluated = evaluate_label_pilot_gate(manifest)
    gate = manifest["label_pilot_gate"]
    if gate.get("decision") != evaluated["decision"]:
        raise SourceFeasibilityError("gate-decision-inconsistent")
    expected_status = (
        "admitted" if evaluated["decision"] == "admitted" else "evidence-backed-blocked"
    )
    if manifest.get("status") != expected_status:
        raise SourceFeasibilityError("manifest-status-inconsistent")
    if evaluated["decision"] == "blocked":
        if not gate.get("blocker_codes") or not gate.get("executable_handoff"):
            raise SourceFeasibilityError("blocked-gate-needs-codes-and-handoff")
