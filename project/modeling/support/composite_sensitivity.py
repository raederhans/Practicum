"""Research-only composite sensitivity with no Public export surface."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any


class CompositeSensitivityError(ValueError):
    """Raised when a composite analysis violates its research contract."""


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise CompositeSensitivityError("unsupported-protocol-schema")
    if protocol.get("status") != "research-only-not-public-admitted":
        raise CompositeSensitivityError("protocol-status-must-remain-research-only")
    boundary = protocol.get("public_boundary")
    if not isinstance(boundary, Mapping):
        raise CompositeSensitivityError("public-boundary-missing")
    for prohibited in (
        "export_to_public_allowed",
        "score_allowed",
        "rank_allowed",
        "outcome_label_allowed",
        "forecast_or_probability_allowed",
    ):
        if boundary.get(prohibited) is not False:
            raise CompositeSensitivityError(f"public-boundary-violated:{prohibited}")
    missingness = protocol.get("missingness", {})
    if missingness.get("policy") != "complete_case_only":
        raise CompositeSensitivityError("missingness-policy-must-fail-closed")
    if missingness.get("imputation_allowed") is not False:
        raise CompositeSensitivityError("imputation-must-remain-disabled")


def validate_snapshot(snapshot: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    validate_protocol(protocol)
    if snapshot.get("schema_version") != 1:
        raise CompositeSensitivityError("unsupported-snapshot-schema")
    comparability = protocol["comparability"]
    if snapshot.get("schema_family") != comparability["required_schema_family"]:
        raise CompositeSensitivityError("schema-family-mismatch")
    if snapshot.get("window_family") != comparability["required_window_family"]:
        raise CompositeSensitivityError("window-family-mismatch")

    maxima = snapshot.get("component_maxima")
    rows = snapshot.get("rows")
    if not isinstance(maxima, Mapping) or not maxima:
        raise CompositeSensitivityError("component-maxima-missing")
    if not isinstance(rows, list) or not rows:
        raise CompositeSensitivityError("snapshot-rows-empty")
    components = set(maxima)
    if any(
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not math.isfinite(float(maximum))
        or maximum <= 0
        for maximum in maxima.values()
    ):
        raise CompositeSensitivityError("component-maximum-invalid")

    event_ids: set[str] = set()
    for row in rows:
        event_id = row.get("event_id")
        values = row.get("components")
        if not isinstance(event_id, str) or not event_id or event_id in event_ids:
            raise CompositeSensitivityError(f"event-id-duplicate-or-invalid:{event_id}")
        event_ids.add(event_id)
        if not isinstance(values, Mapping) or set(values) != components:
            raise CompositeSensitivityError(f"component-schema-mismatch:{event_id}")
        for component, value in values.items():
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
                or float(value) > float(maxima[component])
            ):
                raise CompositeSensitivityError(
                    f"component-value-out-of-range:{event_id}:{component}"
                )


def _complete_rows(snapshot: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    complete: list[dict[str, Any]] = []
    excluded: list[str] = []
    for row in snapshot["rows"]:
        if any(value is None for value in row["components"].values()):
            excluded.append(row["event_id"])
        else:
            complete.append(row)
    return complete, excluded


def _normalized_rows(
    rows: Sequence[Mapping[str, Any]],
    maxima: Mapping[str, float],
    *,
    method: str,
    constant_component_value: float,
) -> dict[str, dict[str, float]]:
    components = list(maxima)
    if method == "fixed_component_max":
        return {
            row["event_id"]: {
                component: float(row["components"][component]) / float(maxima[component])
                for component in components
            }
            for row in rows
        }
    if method != "cohort_min_max":
        raise CompositeSensitivityError(f"normalization-method-unknown:{method}")

    bounds = {
        component: (
            min(float(row["components"][component]) for row in rows),
            max(float(row["components"][component]) for row in rows),
        )
        for component in components
    }
    normalized: dict[str, dict[str, float]] = {}
    for row in rows:
        event_values: dict[str, float] = {}
        for component in components:
            low, high = bounds[component]
            value = float(row["components"][component])
            event_values[component] = (
                constant_component_value if high == low else (value - low) / (high - low)
            )
        normalized[row["event_id"]] = event_values
    return normalized


def _weights_for_mode(
    maxima: Mapping[str, float], components: Sequence[str], mode: str
) -> dict[str, float]:
    if mode == "component_max_proportional":
        raw = {component: float(maxima[component]) for component in components}
    elif mode == "equal_components":
        raw = {component: 1.0 for component in components}
    else:
        raise CompositeSensitivityError(f"weight-mode-unknown:{mode}")
    total = sum(raw.values())
    return {component: value / total for component, value in raw.items()}


def _scores(
    normalized: Mapping[str, Mapping[str, float]], weights: Mapping[str, float]
) -> dict[str, float]:
    return {
        event_id: sum(values[component] * weight for component, weight in weights.items())
        for event_id, values in normalized.items()
    }


def _average_ranks(scores: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and math.isclose(
            ordered[end][1], ordered[index][1], rel_tol=0.0, abs_tol=1e-12
        ):
            end += 1
        average_rank = ((index + 1) + end) / 2
        for event_id, _ in ordered[index:end]:
            ranks[event_id] = average_rank
        index = end
    return ranks


def _rank_correlation(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float:
    if set(left) != set(right) or len(left) < 2:
        raise CompositeSensitivityError("rank-correlation-domain-invalid")
    event_ids = sorted(left)
    left_mean = sum(left[event_id] for event_id in event_ids) / len(event_ids)
    right_mean = sum(right[event_id] for event_id in event_ids) / len(event_ids)
    numerator = sum(
        (left[event_id] - left_mean) * (right[event_id] - right_mean)
        for event_id in event_ids
    )
    left_ss = sum((left[event_id] - left_mean) ** 2 for event_id in event_ids)
    right_ss = sum((right[event_id] - right_mean) ** 2 for event_id in event_ids)
    if left_ss == 0 or right_ss == 0:
        raise CompositeSensitivityError("rank-correlation-constant")
    return numerator / math.sqrt(left_ss * right_ss)


def analyze_composite_sensitivity(
    snapshot: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate normalization, weighting, LOO, and Monte Carlo rank stability."""

    validate_snapshot(snapshot, protocol)
    complete_rows, excluded_events = _complete_rows(snapshot)
    maxima = {key: float(value) for key, value in snapshot["component_maxima"].items()}
    components = list(maxima)
    constant_value = float(protocol["normalization"]["constant_component_value"])

    fixed = _normalized_rows(
        complete_rows,
        maxima,
        method="fixed_component_max",
        constant_component_value=constant_value,
    )
    reference_weights = _weights_for_mode(
        maxima, components, "component_max_proportional"
    )
    reference_scores = _scores(fixed, reference_weights)
    reference_ranks = _average_ranks(reference_scores)

    scenario_specs = [
        ("fixed_component_max__component_max_proportional", "fixed_component_max", "component_max_proportional"),
        ("fixed_component_max__equal_components", "fixed_component_max", "equal_components"),
        ("cohort_min_max__equal_components", "cohort_min_max", "equal_components"),
    ]
    scenarios: dict[str, Any] = {}
    for scenario_id, normalization, weight_mode in scenario_specs:
        normalized = _normalized_rows(
            complete_rows,
            maxima,
            method=normalization,
            constant_component_value=constant_value,
        )
        weights = _weights_for_mode(maxima, components, weight_mode)
        ranks = _average_ranks(_scores(normalized, weights))
        scenarios[scenario_id] = {
            "normalization": normalization,
            "weighting": weight_mode,
            "spearman_to_reference": _rank_correlation(reference_ranks, ranks),
        }

    leave_one_out: dict[str, Any] = {}
    for omitted in components:
        retained = [component for component in components if component != omitted]
        weights = _weights_for_mode(maxima, retained, "component_max_proportional")
        ranks = _average_ranks(_scores(fixed, weights))
        leave_one_out[omitted] = {
            "spearman_to_reference": _rank_correlation(reference_ranks, ranks)
        }

    monte_carlo = protocol["sensitivity"]["monte_carlo"]
    rng = random.Random(int(monte_carlo["seed"]))
    observed_ranks = {event_id: [] for event_id in reference_ranks}
    for _ in range(int(monte_carlo["draws"])):
        raw = {component: rng.gammavariate(1.0, 1.0) for component in components}
        total = sum(raw.values())
        weights = {component: value / total for component, value in raw.items()}
        ranks = _average_ranks(_scores(fixed, weights))
        for event_id, rank in ranks.items():
            observed_ranks[event_id].append(rank)
    monte_carlo_events = {
        event_id: {
            "minimum_rank": min(ranks),
            "maximum_rank": max(ranks),
            "mean_rank": sum(ranks) / len(ranks),
            "top_three_frequency": sum(rank <= 3 for rank in ranks) / len(ranks),
        }
        for event_id, ranks in observed_ranks.items()
    }

    thresholds = protocol["admission_thresholds"]
    minimum_scenario = min(
        scenario["spearman_to_reference"] for scenario in scenarios.values()
    )
    minimum_loo = min(
        scenario["spearman_to_reference"] for scenario in leave_one_out.values()
    )
    maximum_rank_span = max(
        result["maximum_rank"] - result["minimum_rank"]
        for result in monte_carlo_events.values()
    )
    reasons: list[str] = []
    if len(complete_rows) < int(thresholds["minimum_complete_events"]):
        reasons.append("complete-event-count-below-minimum")
    if minimum_scenario < float(thresholds["minimum_scenario_spearman"]):
        reasons.append("normalization-or-weighting-rank-instability")
    if minimum_loo < float(thresholds["minimum_leave_one_out_spearman"]):
        reasons.append("leave-one-component-out-rank-instability")
    if maximum_rank_span > float(thresholds["maximum_monte_carlo_rank_span"]):
        reasons.append("monte-carlo-weight-rank-instability")

    return {
        "decision": "no_go" if reasons else "research_candidate",
        "decision_reasons": reasons,
        "complete_event_count": len(complete_rows),
        "excluded_events": excluded_events,
        "reference": {
            "normalization": "fixed_component_max",
            "weighting": "component_max_proportional",
            "scores": reference_scores,
            "ranks": reference_ranks,
        },
        "scenarios": scenarios,
        "leave_one_component_out": leave_one_out,
        "monte_carlo": {
            "seed": int(monte_carlo["seed"]),
            "draws": int(monte_carlo["draws"]),
            "events": monte_carlo_events,
        },
        "diagnostics": {
            "minimum_scenario_spearman": minimum_scenario,
            "minimum_leave_one_out_spearman": minimum_loo,
            "maximum_monte_carlo_rank_span": maximum_rank_span,
        },
        "public_export_allowed": False,
    }
