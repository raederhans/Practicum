"""Dependency-light recovery outcome contracts and retrospective derivation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


class RecoveryContractError(ValueError):
    """Raised when contract or outcome inputs violate the frozen semantics."""


REQUIRED_SEMANTIC_CLASSES = {
    "admission_readiness",
    "observed_recovery_outcome",
    "forecast",
    "probability",
}


def validate_contract(contract: Mapping[str, Any]) -> None:
    """Validate the machine-readable boundary before any outcome is derived."""

    if contract.get("schema_version") != 1:
        raise RecoveryContractError("unsupported-contract-schema")
    if contract.get("status") != "construct-frozen-not-empirically-admitted":
        raise RecoveryContractError("contract-status-must-remain-non-admitted")

    semantic_classes = contract.get("semantic_classes")
    if not isinstance(semantic_classes, Mapping):
        raise RecoveryContractError("semantic-classes-missing")
    if set(semantic_classes) != REQUIRED_SEMANTIC_CLASSES:
        raise RecoveryContractError("semantic-classes-must-remain-distinct")

    measure_owners: dict[str, str] = {}
    for class_id, definition in semantic_classes.items():
        measures = definition.get("allowed_measure_ids", [])
        if not isinstance(measures, list) or not measures:
            raise RecoveryContractError(f"semantic-class-empty:{class_id}")
        for measure_id in measures:
            if measure_id in measure_owners:
                raise RecoveryContractError(f"measure-has-multiple-owners:{measure_id}")
            measure_owners[measure_id] = class_id

    outcomes = contract.get("outcomes")
    if not isinstance(outcomes, Mapping) or set(outcomes) != {
        "t50_days",
        "t90_days",
        "deficit_burden_observed_day_sum",
    }:
        raise RecoveryContractError("outcome-dictionary-incomplete")
    if measure_owners.get("t50_days") != "observed_recovery_outcome":
        raise RecoveryContractError("observed-outcome-owner-invalid")
    if float(outcomes["t50_days"]["threshold"]) >= float(
        outcomes["t90_days"]["threshold"]
    ):
        raise RecoveryContractError("threshold-order-invalid")

    target = contract.get("target_construct", {})
    horizon = target.get("horizon_days")
    policy = contract.get("observation_policy", {})
    minimum_fraction = policy.get("minimum_observed_fraction")
    sustain_days = policy.get("threshold_sustain_days")
    if not isinstance(horizon, int) or horizon <= 0:
        raise RecoveryContractError("horizon-invalid")
    if not isinstance(minimum_fraction, (int, float)) or not 0 < minimum_fraction <= 1:
        raise RecoveryContractError("minimum-observed-fraction-invalid")
    if not isinstance(sustain_days, int) or sustain_days <= 0:
        raise RecoveryContractError("threshold-sustain-days-invalid")
    if policy.get("zero_is_observed") is not True:
        raise RecoveryContractError("zero-must-remain-observed")
    if policy.get("missing_is_zero") is not False:
        raise RecoveryContractError("missing-must-not-be-zero")
    if policy.get("missing_is_censoring") is not False:
        raise RecoveryContractError("missing-must-not-be-censoring")


def validate_artifact_record(
    record: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    """Reject semantic relabeling across readiness, outcome, forecast, and probability."""

    validate_contract(contract)
    class_id = record.get("semantic_class")
    measure_id = record.get("measure_id")
    semantic_classes = contract["semantic_classes"]
    if class_id not in semantic_classes:
        raise RecoveryContractError(f"unknown-semantic-class:{class_id}")
    if measure_id not in semantic_classes[class_id]["allowed_measure_ids"]:
        raise RecoveryContractError(
            f"measure-class-mismatch:{measure_id}:{class_id}"
        )


def _eligible_signal(record: Mapping[str, Any]) -> float | None:
    if record.get("quality_ok") is not True:
        return None
    value = record.get("normalized_signal")
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _threshold_result(
    eligible_by_day: Mapping[int, float],
    *,
    threshold: float,
    sustain_days: int,
    horizon_days: int,
) -> dict[str, Any]:
    for start_day in range(1, horizon_days - sustain_days + 2):
        run = range(start_day, start_day + sustain_days)
        if all(
            day in eligible_by_day and eligible_by_day[day] >= threshold
            for day in run
        ):
            return {"status": "observed", "value": start_day}
    if horizon_days in eligible_by_day:
        return {
            "status": "right_censored",
            "value": None,
            "lower_bound_days": horizon_days,
        }
    return {"status": "unavailable", "value": None, "reason": "followup-incomplete"}


def derive_observed_recovery_outcomes(
    observations: Iterable[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive retrospective proxy outcomes without imputing missing daily records."""

    validate_contract(contract)
    horizon_days = int(contract["target_construct"]["horizon_days"])
    policy = contract["observation_policy"]
    sustain_days = int(policy["threshold_sustain_days"])

    eligible_by_day: dict[int, float] = {}
    seen_days: set[int] = set()
    for record in observations:
        day = record.get("day")
        if isinstance(day, bool) or not isinstance(day, int):
            raise RecoveryContractError("observation-day-must-be-integer")
        if day < 1 or day > horizon_days:
            raise RecoveryContractError(f"observation-day-out-of-horizon:{day}")
        if day in seen_days:
            raise RecoveryContractError(f"duplicate-observation-day:{day}")
        seen_days.add(day)
        signal = _eligible_signal(record)
        if signal is not None:
            eligible_by_day[day] = signal

    observed_fraction = len(eligible_by_day) / horizon_days
    if observed_fraction < float(policy["minimum_observed_fraction"]):
        unavailable = {
            "status": "unavailable",
            "value": None,
            "reason": "eligible-observation-fraction-below-gate",
        }
        return {
            "semantic_class": "observed_recovery_outcome",
            "observed_fraction": observed_fraction,
            "t50_days": dict(unavailable),
            "t90_days": dict(unavailable),
            "deficit_burden_observed_day_sum": dict(unavailable),
        }

    outcomes = contract["outcomes"]
    burden = sum(max(0.0, 1.0 - signal) for signal in eligible_by_day.values())
    return {
        "semantic_class": "observed_recovery_outcome",
        "observed_fraction": observed_fraction,
        "t50_days": _threshold_result(
            eligible_by_day,
            threshold=float(outcomes["t50_days"]["threshold"]),
            sustain_days=sustain_days,
            horizon_days=horizon_days,
        ),
        "t90_days": _threshold_result(
            eligible_by_day,
            threshold=float(outcomes["t90_days"]["threshold"]),
            sustain_days=sustain_days,
            horizon_days=horizon_days,
        ),
        "deficit_burden_observed_day_sum": {
            "status": "observed",
            "value": burden,
        },
    }
