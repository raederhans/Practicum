import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "script" / "stage3_zipcode_analysis_modelD.py"
FULL_SCRIPT_PATH = PROJECT_ROOT / "script" / "stage3_modelD_full_regressions.py"
EXTRA_SCRIPT_PATH = PROJECT_ROOT / "script" / "stage3_modelD_extra_regressions.py"
DASHBOARD_ROOT = PROJECT_ROOT / "nightlight-dashboard"


def _load_stage3_module():
    spec = importlib.util.spec_from_file_location("stage3_zipcode_analysis_modelD", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_full_regression_helpers(*function_names):
    source = FULL_SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in function_names
    ]
    namespace = {"np": np, "pd": pd, "sm": sm}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(FULL_SCRIPT_PATH), "exec"), namespace)
    return namespace


def test_stage3_uses_canonical_ensemble_band_and_equal_area_crs():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    read_bands = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            read_bands.append(node.args[0].value)

    assert 3 in read_bands, "Stage3 must aggregate the canonical ensemble stored in TIF band 3"
    assert "EPSG:5070" in source, "ZIP areas must use a CONUS equal-area projection"
    assert "EPSG:3857" not in source, "Web Mercator is not valid for ZIP area calculations"


def test_stage3_supports_isolated_paths_and_fails_before_writing(tmp_path):
    module = _load_stage3_module()
    raw_dir = tmp_path / "raw"
    stage2_dir = tmp_path / "stage2"
    output_dir = tmp_path / "output"
    raw_dir.mkdir()
    stage2_dir.mkdir()

    args = module.parse_args(
        [
            "--raw-dir",
            str(raw_dir),
            "--stage2-dir",
            str(stage2_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    with pytest.raises(FileNotFoundError, match="Missing required input"):
        module.run(args)

    assert not output_dir.exists(), "validation must happen before creating partial formal output"


def test_stage3_treats_facility_inputs_as_required(tmp_path):
    module = _load_stage3_module()
    raw_dir = tmp_path / "raw"
    stage2_dir = tmp_path / "stage2"
    output_dir = tmp_path / "output"
    (raw_dir / "zcta520").mkdir(parents=True)
    (raw_dir / "zcta520" / "zcta.shp").touch()
    stage2_dir.mkdir()
    (stage2_dir / "pixel_panel.parquet").touch()

    module.EVENTS = {"Irma_Miami": module.EVENTS["Irma_Miami"]}
    (stage2_dir / "Irma_Miami_prob_map_modelD.tif").touch()
    args = module.parse_args(
        ["--raw-dir", str(raw_dir), "--stage2-dir", str(stage2_dir), "--output-dir", str(output_dir)]
    )
    module._configure_paths(args)

    with pytest.raises(FileNotFoundError, match=r"poi_cache.*Irma_Miami_poi\.csv"):
        module.validate_required_inputs()


def test_followup_regressions_support_isolated_input_and_output_dirs():
    for script in (FULL_SCRIPT_PATH, EXTRA_SCRIPT_PATH):
        source = script.read_text(encoding="utf-8")
        assert '"--input-dir"' in source, script.name
        assert '"--output-dir"' in source, script.name
        assert '"--raw-dir"' in source, script.name


def test_stage3_entrypoints_do_not_globally_hide_scientific_warnings():
    for script in (SCRIPT_PATH, EXTRA_SCRIPT_PATH):
        source = script.read_text(encoding="utf-8")
        assert "warnings.filterwarnings('ignore')" not in source, script.name
        assert 'warnings.filterwarnings("ignore")' not in source, script.name


def test_full_regression_projects_before_zip_centroid_join():
    source = FULL_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "to_crs('EPSG:5070')" in source
    assert "geometry.centroid.to_crs('EPSG:4326')" in source
    assert "zcta_gdf['centroid'] = zcta_gdf.geometry.centroid" not in source


def test_full_regression_ols_clusters_by_event_and_preserves_small_pvalues():
    helpers = _load_full_regression_helpers("event_dummies", "fit_ols")
    rows = []
    for event_index in range(12):
        event_slope = 0.5 + (event_index - 5.5) * 0.5 / 5.5
        for unit_index in range(12):
            x = float(unit_index)
            rows.append(
                {
                    "event_id": f"event_{event_index}",
                    "x": x,
                    "y": event_slope * x + event_index + (unit_index % 3) * 0.5,
                }
            )
    result = helpers["fit_ols"](pd.DataFrame(rows), "y", ["x"], "synthetic")

    assert result["unit_of_analysis"] == "ZIP-event observation"
    assert result["covariance"] == "cluster-robust"
    assert result["inference_distribution"] == "Student t"
    assert result["cluster_variable"] == "event_id"
    assert result["n_clusters"] == 12
    assert 0.0 < result["coefs"]["x"][1] < 1e-3


def test_full_regression_event_block_knn_has_no_cross_event_edges_for_overlapping_coordinates():
    helpers = _load_full_regression_helpers("build_event_block_knn")
    rows = []
    overlapping_coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5), (0.5, 1.5)]
    for event_id in ("event_a", "event_b"):
        for cx, cy in overlapping_coords:
            rows.append({"event_id": event_id, "cx": cx, "cy": cy})
    frame = pd.DataFrame(rows)

    weights, metadata = helpers["build_event_block_knn"](frame, k=5)

    assert metadata["event_blocks"] == 2
    assert metadata["cross_event_edges"] == 0
    assert metadata["components"] == 2
    for source, targets in weights.neighbors.items():
        assert all(frame.iloc[source]["event_id"] == frame.iloc[target]["event_id"] for target in targets)


def test_full_regression_m3_contract_includes_event_fixed_effects_and_spatial_diagnostics():
    source = FULL_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "build_event_block_knn" in source
    assert "event_dummies(sub)" in source
    assert "'event_blocks'" in source
    assert "'cross_event_edges'" in source
    assert "'components'" in source
    assert "'features'" in source
    assert "'covariates'" in source


def test_all_stage3_entrypoints_expose_help_without_optional_model_packages():
    for script in (SCRIPT_PATH, FULL_SCRIPT_PATH, EXTRA_SCRIPT_PATH):
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "--output-dir" in completed.stdout


def test_stage3_regression_output_is_safe_on_strict_gbk_stdout():
    code = f"""
import importlib.util
import pandas as pd

spec = importlib.util.spec_from_file_location('stage3_zip', {str(SCRIPT_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
panel = pd.DataFrame({{
    'event_id': ['event_a'] * 6 + ['event_b'] * 6,
    'fac_density': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] * 2,
    'fac_count': [0, 0, 0, 1, 1, 1] * 2,
    'mean_prob': [0.10, 0.15, 0.20, 0.55, 0.60, 0.65,
                  0.12, 0.17, 0.22, 0.57, 0.62, 0.67],
}})
module.run_regressions(panel)
"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp936:strict"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode("ascii", errors="backslashreplace")
    assert b"UnicodeEncodeError" not in completed.stderr


def test_stage3_exclusions_and_dashboard_event_assets_are_consistent():
    module = _load_stage3_module()
    assert set(module.NON_ZCTA_EVENTS) == {
        "Maria_SanJuan",
        "Earthquake_SanJuan",
        "Earthquake_Hatay",
    }

    events_source = (DASHBOARD_ROOT / "src" / "data" / "events.js").read_text(encoding="utf-8")
    dashboard_ids = re.findall(r"^\s*id:\s*'([^']+)'", events_source, flags=re.MULTILINE)
    assert len(dashboard_ids) == 25
    assert len(set(dashboard_ids)) == 25

    script_dash_ids = {cfg["dash"] for cfg in module.EVENTS.values()}
    assert set(dashboard_ids) == script_dash_ids

    public_data = DASHBOARD_ROOT / "public" / "data"
    for dash_id in dashboard_ids:
        assert (public_data / f"prob_{dash_id}.geojson").is_file(), dash_id
        assert (public_data / f"ts_{dash_id}.json").is_file(), dash_id

    results = json.loads((public_data / "results_summary.json").read_text(encoding="utf-8"))
    loeo_events = {row["held_out"] for row in results["loeo"]}
    assert loeo_events == set(module.EVENTS)
