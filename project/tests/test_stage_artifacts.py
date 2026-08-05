import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE2_DIR = PROJECT_ROOT / "data" / "result" / "stage2"
STAGE3_DIR = PROJECT_ROOT / "data" / "result" / "stage3"


def test_stage2_modeld_probability_maps_are_complete_and_valid():
    tifs = sorted(STAGE2_DIR.glob("*_prob_map_modelD.tif"))
    assert len(tifs) == 25

    for tif in tifs:
        with rasterio.open(tif) as src:
            assert src.count == 3, tif.name
            assert src.crs is not None, tif.name
            ensemble = src.read(3)
        finite = ensemble[np.isfinite(ensemble)]
        assert finite.size > 0, tif.name
        assert float(finite.min()) >= 0.0, tif.name
        assert float(finite.max()) <= 1.0, tif.name


def test_stage2_loeo_modeld_matches_published_25_event_result():
    loeo = pd.read_csv(STAGE2_DIR / "loeo_modelD_25events.csv")
    assert len(loeo) == 25
    assert loeo["held_out"].nunique() == 25
    assert loeo["rf_auc"].mean() == pytest.approx(0.7040403776, abs=1e-10)


def test_stage3_panel_contract():
    panel = pd.read_parquet(STAGE3_DIR / "zipcode_panel_modelD.parquet")
    required = {
        "ZCTA5CE20",
        "mean_prob",
        "max_prob",
        "n_pixels",
        "fac_count",
        "area_km2",
        "fac_density",
        "ntl_drop_pct",
        "event_id",
        "disaster_type",
        "state",
    }
    assert required <= set(panel.columns)
    assert len(panel) == 1002
    assert panel["event_id"].nunique() == 22
    assert panel["state"].nunique() == 15
    assert not panel.duplicated(["event_id", "ZCTA5CE20"]).any()
    assert np.isfinite(panel[list(required - {"ZCTA5CE20", "event_id", "disaster_type", "state"})].to_numpy()).all()
    assert panel["mean_prob"].between(0, 1).all()
    assert panel["max_prob"].between(0, 1).all()
    assert (panel["n_pixels"] > 0).all()
    assert (panel["area_km2"] > 0).all()
    assert (panel["fac_density"] >= 0).all()
    assert set(panel["event_id"]).isdisjoint(
        {"Maria_SanJuan", "Earthquake_SanJuan", "Earthquake_Hatay"}
    )


def test_stage3_regression_results_match_canonical_descriptive_metrics():
    full = json.loads((STAGE3_DIR / "regression_results_modelD_full.json").read_text(encoding="utf-8"))
    extra = json.loads((STAGE3_DIR / "regression_results_modelD_extra.json").read_text(encoding="utf-8"))

    assert full["m1_plus"]["n"] == 977
    assert full["m1_plus"]["r_squared"] == pytest.approx(0.7603)
    assert full["m1_plus"]["adj_r_squared"] == pytest.approx(0.7543)
    assert full["m1_plus"]["unit_of_analysis"] == "ZIP-event observation"
    assert full["m1_plus"]["cluster_variable"] == "event_id"
    assert full["m1_plus"]["n_clusters"] == 22
    assert extra["equity_gap_severity"]["ratio_q3_q1"] == pytest.approx(0.551)
    assert extra["equity_gap_severity"]["status"] == "descriptive-only"


def test_stage3_formal_artifact_hashes_match_canonical_manifest():
    manifest = json.loads(
        (PROJECT_ROOT / "data" / "manifests" / "canonical_results_v1.json").read_text(
            encoding="utf-8"
        )
    )

    for artifact in manifest["formal_artifacts"].values():
        path = PROJECT_ROOT.parent / artifact["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == artifact["sha256"], artifact["path"]
