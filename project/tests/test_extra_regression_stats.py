"""Statistical-contract tests for the extra Stage 3 regressions."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


SCRIPT = Path(__file__).parents[1] / "script" / "stage3_modelD_extra_regressions.py"


def _load_helpers():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {
            "event_dummies",
            "fit_ols",
            "prepare_moran_frame",
            "moran_with_seed",
        }
    ]
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"np": np, "pd": pd, "sm": sm}
    exec(compile(module, str(SCRIPT), "exec"), namespace)
    return namespace


def _clustered_frame():
    rows = []
    for event_number in range(8):
        event_id = f"event_{event_number}"
        event_shift = event_number * 0.15
        for observation in range(12):
            x = observation + event_number * 0.31
            noise = ((observation * 7 + event_number * 3) % 11 - 5) * 0.013
            rows.append(
                {
                    "event_id": event_id,
                    "x": x,
                    "y": 1.2 + 0.73 * x + event_shift + noise,
                }
            )
    return pd.DataFrame(rows)


def test_fit_ols_uses_event_cluster_covariance_and_preserves_pvalue_precision():
    fit_ols = _load_helpers()["fit_ols"]

    result = fit_ols(_clustered_frame(), "y", ["x"], "cluster contract")

    assert result["status"] == "ok"
    assert result["unit_of_analysis"] == "ZIP-event observation"
    assert result["covariance"] == "cluster-robust"
    assert result["inference_distribution"] == "Student t"
    assert result["cluster_variable"] == "event_id"
    assert result["n_clusters"] == 8
    p_value = result["coefs"]["x"][1]
    assert 0.0 < p_value < 1e-6


def test_fit_ols_rejects_event_constant_covariate_with_event_fixed_effects():
    fit_ols = _load_helpers()["fit_ols"]
    frame = _clustered_frame()
    frame["event_constant"] = frame["event_id"].map(
        {event_id: index for index, event_id in enumerate(frame["event_id"].unique())}
    )

    result = fit_ols(
        frame,
        "y",
        ["x", "event_constant"],
        "not identifiable",
        use_fe=True,
    )

    assert result["status"] == "not_identifiable"
    assert result["nonidentifiable_covariates"] == ["event_constant"]
    assert "event fixed effects" in result["reason"]
    assert "coefs" not in result


def test_fit_ols_supports_county_event_main_cluster_and_event_sensitivity():
    fit_ols = _load_helpers()["fit_ols"]
    frame = _clustered_frame()
    frame["fips"] = [1000 + (index % 4) for index in range(len(frame))]

    result = fit_ols(
        frame,
        "y",
        ["x"],
        "severity contract",
        cluster_variable=["event_id", "fips"],
        sensitivity_cluster_variable="event_id",
    )

    assert result["cluster_variable"] == "event_id+fips"
    assert result["n_clusters"] == 32
    assert result["cluster_definition"] == "unique county-event (event_id, fips)"
    assert result["sensitivity"]["cluster_variable"] == "event_id"
    assert result["sensitivity"]["n_clusters"] == 8
    assert result["sensitivity"]["coefs"]["x"][1] > 0.0


def test_severity_models_request_county_event_main_and_event_sensitivity():
    source = SCRIPT.read_text(encoding="utf-8")

    assert source.count("cluster_variable=['event_id', 'fips']") >= 4
    assert source.count("sensitivity_cluster_variable='event_id'") >= 4


def test_naive_group_tests_are_labeled_exploratory_and_have_clustered_checks():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "'naive_ttest':" in source
    assert source.count("'inference': 'exploratory_naive'") == 2
    assert "'clustered_difference': severity_clustered_difference" in source
    assert "'clustered_difference': income_clustered_difference" in source


def test_moran_frame_drops_missing_coordinates_before_residual_fit():
    prepare = _load_helpers()["prepare_moran_frame"]
    frame = pd.DataFrame(
        {
            "ZCTA5CE20": ["00001", "00002", "00003"],
            "mean_prob": [0.1, 0.9, 0.3],
            "fac_density": [1.0, 2.0, 3.0],
        }
    )
    centroids = pd.DataFrame(
        {
            "ZCTA5CE20": ["00001", "00003"],
            "cx": [-80.0, -78.0],
            "cy": [35.0, 37.0],
        }
    )

    prepared = prepare(frame, centroids, ["mean_prob", "fac_density"])

    assert prepared["ZCTA5CE20"].tolist() == ["00001", "00003"]
    assert prepared["mean_prob"].tolist() == [0.1, 0.3]
    assert prepared[["cx", "cy"]].notna().all().all()


def test_moran_residuals_are_not_positionally_truncated_after_coordinate_merge():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "prepare_moran_frame" in source
    assert "res_no[:len(ev)]" not in source


def test_moran_permutation_inference_is_reproducible_without_changing_global_rng():
    helpers = _load_helpers()

    class FakeMoran:
        def __init__(self, _values, _weights):
            self.random_draw = np.random.random()

    helpers["Moran"] = FakeMoran
    moran_with_seed = helpers["moran_with_seed"]

    np.random.seed(2026)
    expected_first = np.random.random()
    expected_second = np.random.random()
    np.random.seed(2026)
    assert np.random.random() == expected_first

    first = moran_with_seed([1.0, 2.0], object())
    second = moran_with_seed([1.0, 2.0], object())

    assert first.random_draw == second.random_draw
    assert np.random.random() == expected_second
