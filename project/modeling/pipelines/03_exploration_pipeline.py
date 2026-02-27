#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path

MODELING_DIR = Path(__file__).resolve().parents[1]
if str(MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(MODELING_DIR))

import json
import math
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.request import urlopen

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline_lib import (
    OUTPUT_DIR,
    PIXEL_DIR,
    REPORT_DIR,
    ROOT,
    FIG_DIR,
    RunContext,
    append_progress,
    ensure_directories,
    init_tracking_files,
    load_json,
    save_issue_log,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

PANEL_IN_PATH = PIXEL_DIR / "all_events_pixel_panel_v1_cross_event_v3.parquet"
RECOVERY_IN_PATH = PIXEL_DIR / "recovery_daily_panel_v1.parquet"
PANEL_OUT_PATH = PIXEL_DIR / "all_events_pixel_panel_v1_exploration_v2.parquet"

FIG_EXP_DIR = FIG_DIR / "exploration_v2"

MASTER_PLAN_PATH = ROOT / "project" / "modeling_tracking" / "future_plan" / "03_exploration_master_plan.md"
CLOUD_PLAN_PATH = ROOT / "project" / "modeling_tracking" / "future_plan" / "04_cloud_coverage_importance_plan.md"
MASK_PLAN_PATH = ROOT / "project" / "modeling_tracking" / "future_plan" / "05_noise_masking_plan.md"
URBAN_PLAN_PATH = ROOT / "project" / "modeling_tracking" / "future_plan" / "06_urban_rural_population_plan.md"
SPATIAL_PLAN_PATH = ROOT / "project" / "modeling_tracking" / "future_plan" / "07_spatial_autocorr_and_contribution_plan.md"
EXTREME_PLAN_PATH = ROOT / "project" / "modeling_tracking" / "future_plan" / "08_extreme_event_sensitivity_plan.md"

REPORT_PATH = REPORT_DIR / "10_exploration_upgrade_report.md"
INDEX_PATH = REPORT_DIR / "index.md"

# Cloud ablation outputs
CLOUD_FOLD_PATH = OUTPUT_DIR / "cloud_ablation_fold_metrics.csv"
CLOUD_AGG_PATH = OUTPUT_DIR / "cloud_ablation_aggregate_metrics.csv"
CLOUD_IMPORTANCE_PATH = OUTPUT_DIR / "cloud_feature_importance.csv"

# Noise mask outputs
MASK_COVERAGE_PATH = OUTPUT_DIR / "noise_mask_coverage_by_event.csv"
MASK_METRICS_PATH = OUTPUT_DIR / "noise_mask_experiment_metrics.csv"
MASK_COEF_PATH = OUTPUT_DIR / "noise_mask_effect_on_coefficients.csv"

# Urban-rural outputs
URBAN_SPLIT_PATH = OUTPUT_DIR / "urban_rural_split_summary.csv"
POP_QUALITY_PATH = OUTPUT_DIR / "pop_density_feature_quality.csv"
URBAN_MODEL_PATH = OUTPUT_DIR / "urban_rural_model_comparison.csv"

# Spatial / contribution outputs
MORAN_PATH = OUTPUT_DIR / "spatial_autocorr_morans_i.csv"
SPATIAL_SE_PATH = OUTPUT_DIR / "spatial_se_comparison.csv"
CONTRIB_SCORE_PATH = OUTPUT_DIR / "feature_contribution_scorecard.csv"
CONTRIB_RANK_PATH = OUTPUT_DIR / "feature_contribution_rank_by_model.csv"

# Extreme sensitivity outputs
EXTREME_CANDIDATE_PATH = OUTPUT_DIR / "extreme_event_candidates_v1.csv"
EXTREME_SCORE_PATH = OUTPUT_DIR / "extreme_event_scorecard_v1.csv"
EXTREME_DROP_METRIC_PATH = OUTPUT_DIR / "extreme_event_drop_metrics_v1.csv"
EXTREME_DROP_AGG_PATH = OUTPUT_DIR / "extreme_event_drop_aggregate_v1.csv"
EXTREME_DECISION_PATH = OUTPUT_DIR / "extreme_event_stop_or_keep_v1.json"

# Existing references
SHIFT_V3_PATH = OUTPUT_DIR / "cross_event_shift_diagnostics_v3.csv"
FOLD_V3R1_PATH = OUTPUT_DIR / "cross_event_fold_metrics_v3r1.csv"

# Added fields required by plan
ADDED_FIELDS = [
    "noise_mask_group",
    "is_cbsa",
    "is_urban_area",
    "urban_rural_stratum",
    "pop_density_per_km2",
    "pop_density_log1p",
    "missing_pop_flag",
]

BASE_NUMERIC = [
    "osm_dist_any_m",
    "osm_power_count_1000m",
    "osm_medical_count_1000m",
    "urban_share_1km",
    "water_share_1km",
    "developed_high_share_1km",
]

CLOUD_VARIANTS = {
    "C0": [],
    "C1": ["pixel_cloud_proxy"],
    "C2": ["pixel_cloud_proxy", "pixel_pre_valid_ratio", "pixel_post_valid_ratio"],
    "C3": ["pixel_cloud_proxy", "pixel_pre_valid_ratio", "pixel_post_valid_ratio", "missing_cloud_flag"],
}

MASK_CLASSES_M1 = {11, 12, 90, 95}
MASK_CLASSES_M2_EXTRA = {31, 41, 42, 43, 52, 71}

MODEL_ORDER = ["OLS", "MixedLM", "Logit", "Cox", "AFT"]

EVENT_TO_STATE = {
    "maria_sanjuan": "72",
    "earthquake_sanjuan": "72",
    "ida_neworleans": "22",
    "laura_lakecharles": "22",
    "michael_panamacity": "12",
    "irma_miami": "12",
}


@dataclass
class SpecResult:
    fold_df: pd.DataFrame
    agg_df: pd.DataFrame
    coef_df: pd.DataFrame


def _safe_numeric(s: pd.Series, default: float = 0.0) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if out.notna().any():
        return out.fillna(out.median())
    return out.fillna(default)


def _zscore(x: pd.Series) -> pd.Series:
    v = _safe_numeric(x, default=0.0)
    std = float(v.std(ddof=0))
    if std == 0 or not np.isfinite(std):
        return pd.Series(np.zeros(len(v)), index=v.index)
    return (v - float(v.mean())) / std


def _fetch_json(url: str, timeout: int = 30):
    with urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _land_use_to_noise_group(v: float) -> str:
    if pd.isna(v):
        return "kept"
    try:
        code = int(v)
    except Exception:
        return "kept"
    if code in MASK_CLASSES_M1:
        return "masked_m1"
    if code in MASK_CLASSES_M2_EXTRA:
        return "masked_m2"
    return "kept"


def _build_formula(target: str, numeric_terms: Sequence[str], cat_terms: Sequence[str]) -> str:
    rhs = ["in_buffer * pre_mean_ntl"]
    rhs.extend(list(numeric_terms))
    rhs.extend([f"C({c})" for c in cat_terms])
    return f"{target} ~ " + " + ".join(rhs)


def _prepare_columns(df: pd.DataFrame, numeric_terms: Sequence[str], cat_terms: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    base_num = ["delta_ntl", "pre_mean_ntl", "in_buffer", "is_damaged", "recovery_days", "event_observed"]
    for c in base_num + list(numeric_terms):
        if c in out.columns:
            out[c] = _safe_numeric(out[c])
        else:
            out[c] = 0.0

    for c in cat_terms:
        if c not in out.columns:
            out[c] = "unknown"
        out[c] = out[c].fillna("unknown").astype(str)

    out["in_buffer_x_pre"] = out["in_buffer"] * out["pre_mean_ntl"]
    return out


def _prune_terms(
    train_df: pd.DataFrame,
    numeric_terms: Sequence[str],
    cat_terms: Sequence[str],
) -> Tuple[List[str], List[str]]:
    kept_num: List[str] = []
    kept_cat: List[str] = []

    for c in numeric_terms:
        if c not in train_df.columns:
            continue
        vals = _safe_numeric(train_df[c])
        if vals.nunique(dropna=True) > 1:
            kept_num.append(c)

    for c in cat_terms:
        if c not in train_df.columns:
            continue
        nuniq = train_df[c].fillna("unknown").astype(str).nunique(dropna=True)
        if nuniq > 1:
            kept_cat.append(c)

    return kept_num, kept_cat


def _build_logit_design(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_terms: Sequence[str],
    cat_terms: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    base_cols = ["in_buffer", "pre_mean_ntl", "in_buffer_x_pre"]
    num_cols = base_cols + list(numeric_terms)

    tr = train_df.copy()
    te = test_df.copy()
    for c in num_cols:
        if c not in tr.columns:
            tr[c] = 0.0
            te[c] = 0.0
        tr[c] = _safe_numeric(tr[c])
        med = float(tr[c].median()) if tr[c].notna().any() else 0.0
        te[c] = _safe_numeric(te[c]).fillna(med)

    tr_num = tr[num_cols].copy()
    te_num = te[num_cols].copy()

    dtr = pd.get_dummies(tr[cat_terms], prefix=cat_terms, drop_first=True) if cat_terms else pd.DataFrame(index=tr.index)
    dte = pd.get_dummies(te[cat_terms], prefix=cat_terms, drop_first=True) if cat_terms else pd.DataFrame(index=te.index)
    dte = dte.reindex(columns=dtr.columns, fill_value=0)

    xtr = pd.concat([tr_num.reset_index(drop=True), dtr.reset_index(drop=True)], axis=1)
    xte = pd.concat([te_num.reset_index(drop=True), dte.reset_index(drop=True)], axis=1)
    return xtr, xte, xtr.columns.tolist()


def _build_survival_design(
    df: pd.DataFrame,
    numeric_terms: Sequence[str],
    cat_terms: Sequence[str],
) -> pd.DataFrame:
    keep = ["recovery_days", "event_observed", "event_id", "in_buffer", "pre_mean_ntl", "in_buffer_x_pre"]
    keep += list(numeric_terms)
    keep += list(cat_terms)
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    for c in ["recovery_days", "event_observed", "in_buffer", "pre_mean_ntl", "in_buffer_x_pre"] + list(numeric_terms):
        if c in out.columns:
            out[c] = _safe_numeric(out[c])

    dummy_parts = []
    for c in cat_terms:
        if c in out.columns:
            dummy_parts.append(pd.get_dummies(out[c].fillna("unknown").astype(str), prefix=c, drop_first=True))
            out = out.drop(columns=[c])
    if dummy_parts:
        out = pd.concat([out.reset_index(drop=True)] + [d.reset_index(drop=True) for d in dummy_parts], axis=1)

    out = out.replace([np.inf, -np.inf], np.nan)
    for c in out.columns:
        if c == "event_id":
            continue
        out[c] = _safe_numeric(out[c])
    return out


def _evaluate_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_rec: pd.DataFrame,
    test_rec: pd.DataFrame,
    numeric_terms: Sequence[str],
    cat_terms: Sequence[str],
    experiment_family: str,
    spec_id: str,
    fold_event: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    coef_rows: List[Dict[str, object]] = []

    num_terms, cats = _prune_terms(train_df, numeric_terms, cat_terms)
    formula = _build_formula("delta_ntl", num_terms, cats)

    # OLS
    try:
        ols = smf.ols(formula, data=train_df).fit(cov_type="HC1")
        pred = ols.predict(test_df)
        rmse = float(np.sqrt(mean_squared_error(test_df["delta_ntl"], pred)))
        mae = float(mean_absolute_error(test_df["delta_ntl"], pred))
        rows.append(
            {
                "experiment_family": experiment_family,
                "spec_id": spec_id,
                "fold_event": fold_event,
                "model": "OLS",
                "rmse": rmse,
                "mae": mae,
                "auc": np.nan,
                "brier": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": float(ols.params.get("in_buffer", np.nan)),
                "notes": "ok",
            }
        )
        for term, val in ols.params.items():
            coef_rows.append(
                {
                    "experiment_family": experiment_family,
                    "spec_id": spec_id,
                    "fold_event": fold_event,
                    "model": "OLS",
                    "feature": term,
                    "coef": float(val),
                    "p_value": float(ols.pvalues.get(term, np.nan)),
                }
            )
    except Exception as e:
        rows.append(
            {
                "experiment_family": experiment_family,
                "spec_id": spec_id,
                "fold_event": fold_event,
                "model": "OLS",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": np.nan,
                "brier": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": np.nan,
                "notes": f"fail:{type(e).__name__}",
            }
        )

    # MixedLM
    try:
        mixed = smf.mixedlm(formula, data=train_df, groups=train_df["event_id"]).fit(reml=False, method="lbfgs", maxiter=200)
        pred = mixed.predict(test_df)
        rmse = float(np.sqrt(mean_squared_error(test_df["delta_ntl"], pred)))
        mae = float(mean_absolute_error(test_df["delta_ntl"], pred))
        rows.append(
            {
                "experiment_family": experiment_family,
                "spec_id": spec_id,
                "fold_event": fold_event,
                "model": "MixedLM",
                "rmse": rmse,
                "mae": mae,
                "auc": np.nan,
                "brier": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": float(mixed.params.get("in_buffer", np.nan)),
                "notes": "ok",
            }
        )
        for term, val in mixed.params.items():
            if term == "Group Var":
                continue
            coef_rows.append(
                {
                    "experiment_family": experiment_family,
                    "spec_id": spec_id,
                    "fold_event": fold_event,
                    "model": "MixedLM",
                    "feature": term,
                    "coef": float(val),
                    "p_value": float(mixed.pvalues.get(term, np.nan)),
                }
            )
    except Exception as e:
        rows.append(
            {
                "experiment_family": experiment_family,
                "spec_id": spec_id,
                "fold_event": fold_event,
                "model": "MixedLM",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": np.nan,
                "brier": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": np.nan,
                "notes": f"fail:{type(e).__name__}",
            }
        )

    # Logit
    try:
        xtr, xte, feat_names = _build_logit_design(train_df, test_df, num_terms, cats)
        ytr = _safe_numeric(train_df["is_damaged"]).astype(int)
        yte = _safe_numeric(test_df["is_damaged"]).astype(int)

        logit = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", class_weight="balanced")
        logit.fit(xtr, ytr)

        prob = logit.predict_proba(xte)[:, 1]
        auc = float(roc_auc_score(yte, prob)) if yte.nunique() > 1 else np.nan
        brier = float(brier_score_loss(yte, prob))

        coef_in_buffer = np.nan
        if "in_buffer" in feat_names:
            coef_in_buffer = float(logit.coef_[0, feat_names.index("in_buffer")])

        rows.append(
            {
                "experiment_family": experiment_family,
                "spec_id": spec_id,
                "fold_event": fold_event,
                "model": "Logit",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": auc,
                "brier": brier,
                "c_index": np.nan,
                "coef_in_buffer": coef_in_buffer,
                "notes": "ok",
            }
        )

        for idx, name in enumerate(feat_names):
            coef_rows.append(
                {
                    "experiment_family": experiment_family,
                    "spec_id": spec_id,
                    "fold_event": fold_event,
                    "model": "Logit",
                    "feature": name,
                    "coef": float(logit.coef_[0, idx]),
                    "p_value": np.nan,
                }
            )
    except Exception as e:
        rows.append(
            {
                "experiment_family": experiment_family,
                "spec_id": spec_id,
                "fold_event": fold_event,
                "model": "Logit",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": np.nan,
                "brier": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": np.nan,
                "notes": f"fail:{type(e).__name__}",
            }
        )

    # Cox / AFT
    try:
        surv_num, surv_cat = _prune_terms(train_rec, num_terms, cats)
        s_tr = _build_survival_design(train_rec, surv_num, surv_cat)
        s_te = _build_survival_design(test_rec, surv_num, surv_cat)

        # Align columns
        target_cols = [c for c in s_tr.columns if c not in {"recovery_days", "event_observed", "event_id"}]
        for c in target_cols:
            if c not in s_te.columns:
                s_te[c] = 0.0
        s_te = s_te.reindex(columns=s_tr.columns, fill_value=0.0)

        cox = CoxPHFitter(penalizer=0.01)
        cox.fit(s_tr.drop(columns=["event_id"], errors="ignore"), duration_col="recovery_days", event_col="event_observed")
        risk = cox.predict_partial_hazard(s_te.drop(columns=["recovery_days", "event_observed", "event_id"], errors="ignore"))
        c_idx = float(
            concordance_index(
                s_te["recovery_days"],
                -risk.to_numpy().reshape(-1),
                s_te["event_observed"],
            )
        )
        coef_in = float(cox.params_.get("in_buffer", np.nan))
        rows.append(
            {
                "experiment_family": experiment_family,
                "spec_id": spec_id,
                "fold_event": fold_event,
                "model": "Cox",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": np.nan,
                "brier": np.nan,
                "c_index": c_idx,
                "coef_in_buffer": coef_in,
                "notes": "ok",
            }
        )
        for term, val in cox.params_.items():
            coef_rows.append(
                {
                    "experiment_family": experiment_family,
                    "spec_id": spec_id,
                    "fold_event": fold_event,
                    "model": "Cox",
                    "feature": term,
                    "coef": float(val),
                    "p_value": float(cox.summary.loc[term, "p"]) if term in cox.summary.index else np.nan,
                }
            )

        aft = WeibullAFTFitter(penalizer=0.01)
        aft.fit(s_tr.drop(columns=["event_id"], errors="ignore"), duration_col="recovery_days", event_col="event_observed")
        med = aft.predict_median(s_te.drop(columns=["recovery_days", "event_observed", "event_id"], errors="ignore"))
        c_idx_aft = float(
            concordance_index(
                s_te["recovery_days"],
                -med.to_numpy().reshape(-1),
                s_te["event_observed"],
            )
        )

        cands = [i for i in aft.params_.index if i[0] == "lambda_" and i[1] == "in_buffer"]
        coef_in_aft = float(aft.params_.loc[cands[0]]) if cands else np.nan
        rows.append(
            {
                "experiment_family": experiment_family,
                "spec_id": spec_id,
                "fold_event": fold_event,
                "model": "AFT",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": np.nan,
                "brier": np.nan,
                "c_index": c_idx_aft,
                "coef_in_buffer": coef_in_aft,
                "notes": "ok",
            }
        )
    except Exception as e:
        rows.append(
            {
                "experiment_family": experiment_family,
                "spec_id": spec_id,
                "fold_event": fold_event,
                "model": "Cox",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": np.nan,
                "brier": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": np.nan,
                "notes": f"fail:{type(e).__name__}",
            }
        )
        rows.append(
            {
                "experiment_family": experiment_family,
                "spec_id": spec_id,
                "fold_event": fold_event,
                "model": "AFT",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": np.nan,
                "brier": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": np.nan,
                "notes": f"fail:{type(e).__name__}",
            }
        )

    return rows, coef_rows


def _aggregate_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for (fam, spec, model), grp in fold_df.groupby(["experiment_family", "spec_id", "model"], dropna=False):
        out_rows.append(
            {
                "experiment_family": fam,
                "spec_id": spec,
                "model": model,
                "rmse": float(grp["rmse"].mean(skipna=True)),
                "mae": float(grp["mae"].mean(skipna=True)),
                "auc": float(grp["auc"].mean(skipna=True)),
                "brier": float(grp["brier"].mean(skipna=True)),
                "c_index": float(grp["c_index"].mean(skipna=True)),
                "coef_in_buffer": float(grp["coef_in_buffer"].mean(skipna=True)),
                "n_folds": int(grp["fold_event"].nunique()),
            }
        )
    return pd.DataFrame(out_rows)


def run_loeo_spec(
    panel: pd.DataFrame,
    recovery: pd.DataFrame,
    numeric_terms: Sequence[str],
    cat_terms: Sequence[str],
    experiment_family: str,
    spec_id: str,
    allowed_events: Optional[Sequence[str]] = None,
) -> SpecResult:
    df = panel.copy()
    rec = recovery.copy()

    if allowed_events is not None:
        allowed = set(allowed_events)
        df = df[df["event_id"].isin(allowed)].copy()
        rec = rec[rec["event_id"].isin(allowed)].copy()

    events = sorted(df["event_id"].dropna().unique().tolist())
    fold_rows: List[Dict[str, object]] = []
    coef_rows: List[Dict[str, object]] = []

    for fold_event in events:
        tr = df[df["event_id"] != fold_event].copy()
        te = df[df["event_id"] == fold_event].copy()
        tr_rec = rec[rec["event_id"] != fold_event].copy()
        te_rec = rec[rec["event_id"] == fold_event].copy()

        tr = _prepare_columns(tr, numeric_terms, cat_terms)
        te = _prepare_columns(te, numeric_terms, cat_terms)
        tr_rec = _prepare_columns(tr_rec, numeric_terms, cat_terms)
        te_rec = _prepare_columns(te_rec, numeric_terms, cat_terms)

        rows, coefs = _evaluate_fold(
            tr,
            te,
            tr_rec,
            te_rec,
            numeric_terms=numeric_terms,
            cat_terms=cat_terms,
            experiment_family=experiment_family,
            spec_id=spec_id,
            fold_event=fold_event,
        )
        fold_rows.extend(rows)
        coef_rows.extend(coefs)

    fold_df = pd.DataFrame(fold_rows)
    agg_df = _aggregate_metrics(fold_df)
    coef_df = pd.DataFrame(coef_rows)
    return SpecResult(fold_df=fold_df, agg_df=agg_df, coef_df=coef_df)


def _prepare_noise_groups(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["noise_mask_group"] = out["land_use"].apply(_land_use_to_noise_group)
    return out


def _read_first_geo(urls: Sequence[str]) -> gpd.GeoDataFrame:
    errs: List[str] = []
    for u in urls:
        try:
            return gpd.read_file(u).to_crs("EPSG:4326")
        except Exception as e:
            errs.append(f"{u}:{type(e).__name__}")
    raise RuntimeError("all_urls_failed:" + ";".join(errs))


def _download_cbsa_ua() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    cbsa = _read_first_geo(
        [
            "https://www2.census.gov/geo/tiger/TIGER2023/CBSA/tl_2023_us_cbsa.zip",
            "https://www2.census.gov/geo/tiger/TIGER2024/CBSA/tl_2024_us_cbsa.zip",
            "https://www2.census.gov/geo/tiger/TIGER2021/CBSA/tl_2021_us_cbsa.zip",
        ]
    )
    ua = _read_first_geo(
        [
            "https://www2.census.gov/geo/tiger/TIGER2023/UAC20/tl_2023_us_uac20.zip",
            "https://www2.census.gov/geo/tiger/TIGER2022/UAC/tl_2022_us_uac10.zip",
        ]
    )

    cbsa_id_col = "GEOID" if "GEOID" in cbsa.columns else cbsa.columns[0]
    ua_id_col = "GEOID20" if "GEOID20" in ua.columns else ("GEOID10" if "GEOID10" in ua.columns else ua.columns[0])
    return cbsa[[cbsa_id_col, "geometry"]].rename(columns={cbsa_id_col: "GEOID"}), ua[[ua_id_col, "geometry"]].rename(
        columns={ua_id_col: "GEOID20"}
    )


def _download_tracts_for_states(states: Iterable[str]) -> gpd.GeoDataFrame:
    pieces = []
    for st in sorted(set(states)):
        url = f"https://www2.census.gov/geo/tiger/TIGER2022/TRACT/tl_2022_{st}_tract.zip"
        g = gpd.read_file(url).to_crs("EPSG:4326")
        pieces.append(g[["STATEFP", "COUNTYFP", "TRACTCE", "GEOID", "ALAND", "geometry"]])
    if not pieces:
        return gpd.GeoDataFrame(columns=["STATEFP", "COUNTYFP", "TRACTCE", "GEOID", "ALAND", "geometry"], geometry="geometry", crs="EPSG:4326")
    return pd.concat(pieces, ignore_index=True)


def _fetch_acs_population(states: Iterable[str]) -> pd.DataFrame:
    rows = []
    for st in sorted(set(states)):
        url = f"https://api.census.gov/data/2022/acs/acs5?get=B01003_001E&for=tract:*&in=state:{st}&in=county:*"
        data = _fetch_json(url)
        if not isinstance(data, list) or len(data) <= 1:
            continue
        header = data[0]
        for rec in data[1:]:
            item = dict(zip(header, rec))
            geoid = f"{item.get('state','')}{item.get('county','')}{item.get('tract','')}"
            pop = pd.to_numeric(item.get("B01003_001E"), errors="coerce")
            rows.append({"GEOID": geoid, "population": pop})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["GEOID"], keep="last")
    return out


def attach_urban_population(panel: pd.DataFrame, ctx: RunContext) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = panel.copy()
    out["is_cbsa"] = 0
    out["is_urban_area"] = 0
    out["urban_rural_stratum"] = "rural"
    out["pop_density_per_km2"] = np.nan
    out["pop_density_log1p"] = np.nan
    out["missing_pop_flag"] = 1

    quality_rows = []

    gpts = gpd.GeoDataFrame(
        out[["pixel_id", "event_id", "lon", "lat"]].copy(),
        geometry=gpd.points_from_xy(out["lon"], out["lat"]),
        crs="EPSG:4326",
    )

    # CBSA + UA
    try:
        cbsa, ua = _download_cbsa_ua()
        cbsa_join = gpd.sjoin(gpts[["pixel_id", "geometry"]], cbsa, how="left", predicate="within")
        ua_join = gpd.sjoin(gpts[["pixel_id", "geometry"]], ua, how="left", predicate="within")

        cbsa_ids = set(cbsa_join.loc[cbsa_join["GEOID"].notna(), "pixel_id"].astype(str))
        ua_ids = set(ua_join.loc[ua_join["GEOID20"].notna(), "pixel_id"].astype(str))

        out.loc[out["pixel_id"].astype(str).isin(cbsa_ids), "is_cbsa"] = 1
        out.loc[out["pixel_id"].astype(str).isin(ua_ids), "is_urban_area"] = 1
        quality_rows.append(
            {
                "feature": "cbsa_urban_join",
                "status": "ok",
                "coverage_ratio": float((out["is_cbsa"] == 1).mean()),
                "notes": "Census TIGER 2022",
            }
        )
    except Exception as e:
        quality_rows.append(
            {
                "feature": "cbsa_urban_join",
                "status": f"fallback:{type(e).__name__}",
                "coverage_ratio": 0.0,
                "notes": "set is_cbsa/is_urban_area default 0",
            }
        )

    out["urban_rural_stratum"] = np.where(
        out["is_urban_area"] == 1,
        "urban",
        np.where(out["is_cbsa"] == 1, "suburban", "rural"),
    )

    # ACS tract population
    try:
        state_list = sorted({EVENT_TO_STATE.get(e) for e in out["event_id"].dropna().unique().tolist() if EVENT_TO_STATE.get(e)})
        tracts = _download_tracts_for_states(state_list)
        tracts = tracts.copy()
        tracts["area_km2"] = _safe_numeric(tracts["ALAND"]) / 1_000_000.0

        join = gpd.sjoin(gpts[["pixel_id", "event_id", "geometry"]], tracts[["GEOID", "area_km2", "geometry"]], how="left", predicate="within")
        pop = _fetch_acs_population(state_list)
        if pop.empty:
            raise RuntimeError("ACS population table is empty")

        join = join.merge(pop, on="GEOID", how="left")
        join["population"] = _safe_numeric(join["population"])
        join["pop_density_per_km2"] = np.where(join["area_km2"] > 0, join["population"] / join["area_km2"], np.nan)

        pop_map = join[["pixel_id", "pop_density_per_km2"]].drop_duplicates(subset=["pixel_id"])
        out = out.merge(pop_map, on="pixel_id", how="left", suffixes=("", "_new"))
        out["pop_density_per_km2"] = _safe_numeric(out.get("pop_density_per_km2_new", out["pop_density_per_km2"]))
        out = out.drop(columns=[c for c in ["pop_density_per_km2_new"] if c in out.columns])

        coverage = float(out["pop_density_per_km2"].notna().mean())
        quality_rows.append(
            {
                "feature": "acs_pop_density",
                "status": "ok",
                "coverage_ratio": coverage,
                "notes": "ACS 2022 B01003 + TIGER tract ALAND",
            }
        )
    except Exception as e:
        quality_rows.append(
            {
                "feature": "acs_pop_density",
                "status": f"fallback:{type(e).__name__}",
                "coverage_ratio": float(out["pop_density_per_km2"].notna().mean()),
                "notes": "keep missing and continue",
            }
        )

    out["pop_density_per_km2"] = _safe_numeric(out["pop_density_per_km2"])
    out["pop_density_log1p"] = np.log1p(out["pop_density_per_km2"].clip(lower=0.0))
    out["missing_pop_flag"] = np.where(out["pop_density_per_km2"].isna(), 1, 0).astype(int)
    quality = pd.DataFrame(quality_rows)
    return out, quality


def build_recovery_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if not RECOVERY_IN_PATH.exists():
        raise FileNotFoundError(f"Recovery panel missing: {RECOVERY_IN_PATH}")
    rec = pd.read_parquet(RECOVERY_IN_PATH)
    keep_cols = [
        "pixel_id",
        "event_id",
        "recovery_days",
        "event_observed",
    ]
    feature_cols = [
        "in_buffer",
        "pre_mean_ntl",
        "land_use_group",
        "event_disaster_type",
        "urban_rural_stratum",
        "osm_dist_any_m",
        "osm_power_count_1000m",
        "osm_medical_count_1000m",
        "pixel_cloud_proxy",
        "pixel_pre_valid_ratio",
        "pixel_post_valid_ratio",
        "missing_cloud_flag",
        "urban_share_1km",
        "water_share_1km",
        "developed_high_share_1km",
        "is_cbsa",
        "is_urban_area",
        "pop_density_log1p",
        "pop_density_per_km2",
    ]
    merge_src = panel[["pixel_id"] + [c for c in feature_cols if c in panel.columns]].drop_duplicates(subset=["pixel_id"])
    rec = rec[keep_cols].merge(merge_src, on="pixel_id", how="left")
    rec = rec[rec["event_id"].isin(panel["event_id"].unique())].copy()
    rec["sample_lock_flag"] = 1
    return rec


def run_cloud_ablation(panel: pd.DataFrame, recovery: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_parts, agg_parts, coef_parts = [], [], []

    cat_terms = ["land_use_group", "event_disaster_type"]
    for spec_id, cloud_terms in CLOUD_VARIANTS.items():
        num_terms = BASE_NUMERIC + cloud_terms
        res = run_loeo_spec(
            panel,
            recovery,
            numeric_terms=num_terms,
            cat_terms=cat_terms,
            experiment_family="cloud_ablation",
            spec_id=spec_id,
        )
        fold_parts.append(res.fold_df)
        agg_parts.append(res.agg_df)
        coef_parts.append(res.coef_df)

    fold = pd.concat(fold_parts, ignore_index=True)
    agg = pd.concat(agg_parts, ignore_index=True)
    coef = pd.concat(coef_parts, ignore_index=True)

    # Cloud feature importance from Logit coefficients (mean abs coef over folds)
    cloud_feats = ["pixel_cloud_proxy", "pixel_pre_valid_ratio", "pixel_post_valid_ratio", "missing_cloud_flag"]
    imp = (
        coef[(coef["model"] == "Logit") & (coef["feature"].isin(cloud_feats))]
        .assign(abs_coef=lambda d: d["coef"].abs())
        .groupby(["spec_id", "feature"], as_index=False)["abs_coef"]
        .mean()
        .rename(columns={"abs_coef": "importance"})
    )
    if imp.empty:
        imp = pd.DataFrame(columns=["spec_id", "feature", "importance"])

    fold.to_csv(CLOUD_FOLD_PATH, index=False)
    agg.to_csv(CLOUD_AGG_PATH, index=False)
    imp.to_csv(CLOUD_IMPORTANCE_PATH, index=False)
    return fold, agg, coef


def run_noise_mask(panel: pd.DataFrame, recovery: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = panel.copy()
    p["land_use_int"] = pd.to_numeric(p["land_use"], errors="coerce").astype("Int64")

    subsets = {
        "M0": p,
        "M1": p[~p["land_use_int"].isin(list(MASK_CLASSES_M1))].copy(),
        "M2": p[~p["land_use_int"].isin(list(MASK_CLASSES_M1 | MASK_CLASSES_M2_EXTRA))].copy(),
    }

    coverage_rows = []
    for name, sdf in subsets.items():
        for event_id, grp in p.groupby("event_id"):
            kept = int((sdf["event_id"] == event_id).sum())
            total = int((p["event_id"] == event_id).sum())
            coverage_rows.append(
                {
                    "spec_id": name,
                    "event_id": event_id,
                    "n_kept": kept,
                    "n_total": total,
                    "kept_ratio": float(kept / total) if total > 0 else np.nan,
                }
            )
    coverage = pd.DataFrame(coverage_rows)

    fold_parts, agg_parts, coef_parts = [], [], []
    num_terms = BASE_NUMERIC + ["pixel_cloud_proxy"]
    cat_terms = ["land_use_group", "event_disaster_type"]

    for spec_id, sdf in subsets.items():
        rsub = recovery[recovery["pixel_id"].isin(sdf["pixel_id"])].copy()
        res = run_loeo_spec(
            sdf,
            rsub,
            numeric_terms=num_terms,
            cat_terms=cat_terms,
            experiment_family="noise_mask",
            spec_id=spec_id,
        )
        fold_parts.append(res.fold_df)
        agg_parts.append(res.agg_df)
        coef_parts.append(res.coef_df)

    fold = pd.concat(fold_parts, ignore_index=True)
    agg = pd.concat(agg_parts, ignore_index=True)
    coef = pd.concat(coef_parts, ignore_index=True)

    coef_effect = (
        coef[coef["feature"] == "in_buffer"]
        .groupby(["spec_id", "model"], as_index=False)["coef"]
        .mean()
        .rename(columns={"coef": "coef_in_buffer_mean"})
    )

    coverage.to_csv(MASK_COVERAGE_PATH, index=False)
    agg.to_csv(MASK_METRICS_PATH, index=False)
    coef_effect.to_csv(MASK_COEF_PATH, index=False)
    return fold, agg, coef


def run_urban_population(panel: pd.DataFrame, recovery: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = panel.copy()

    split_summary = (
        p.groupby(["event_id", "urban_rural_stratum"], as_index=False)
        .agg(
            n_obs=("pixel_id", "count"),
            damage_rate=("is_damaged", "mean"),
            pre_ntl_mean=("pre_mean_ntl", "mean"),
            pop_density_mean=("pop_density_per_km2", "mean"),
        )
    )

    fold_parts, agg_parts, coef_parts = [], [], []

    num_terms = BASE_NUMERIC + ["pixel_cloud_proxy", "is_cbsa", "is_urban_area", "pop_density_log1p"]
    cat_terms = ["land_use_group", "event_disaster_type", "urban_rural_stratum"]

    # Full
    res_full = run_loeo_spec(
        p,
        recovery,
        numeric_terms=num_terms,
        cat_terms=cat_terms,
        experiment_family="urban_rural",
        spec_id="UR_full",
    )
    fold_parts.append(res_full.fold_df)
    agg_parts.append(res_full.agg_df)
    coef_parts.append(res_full.coef_df)

    # Urban only
    p_urban = p[p["urban_rural_stratum"] == "urban"].copy()
    r_urban = recovery[recovery["pixel_id"].isin(p_urban["pixel_id"])].copy()
    if p_urban["event_id"].nunique() >= 3:
        res_urban = run_loeo_spec(
            p_urban,
            r_urban,
            numeric_terms=num_terms,
            cat_terms=["land_use_group", "event_disaster_type"],
            experiment_family="urban_rural",
            spec_id="UR_urban",
        )
        fold_parts.append(res_urban.fold_df)
        agg_parts.append(res_urban.agg_df)
        coef_parts.append(res_urban.coef_df)

    # Rural only
    p_rural = p[p["urban_rural_stratum"] == "rural"].copy()
    r_rural = recovery[recovery["pixel_id"].isin(p_rural["pixel_id"])].copy()
    if p_rural["event_id"].nunique() >= 3:
        res_rural = run_loeo_spec(
            p_rural,
            r_rural,
            numeric_terms=num_terms,
            cat_terms=["land_use_group", "event_disaster_type"],
            experiment_family="urban_rural",
            spec_id="UR_rural",
        )
        fold_parts.append(res_rural.fold_df)
        agg_parts.append(res_rural.agg_df)
        coef_parts.append(res_rural.coef_df)

    fold = pd.concat(fold_parts, ignore_index=True)
    agg = pd.concat(agg_parts, ignore_index=True)

    split_summary.to_csv(URBAN_SPLIT_PATH, index=False)
    agg.to_csv(URBAN_MODEL_PATH, index=False)
    return fold, agg, pd.concat(coef_parts, ignore_index=True)


def _moran_i_knn(df: pd.DataFrame, value_col: str, k: int = 8, n_perm: int = 199) -> Tuple[float, float]:
    if len(df) < (k + 2):
        return np.nan, np.nan

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    coords = np.column_stack([x, y])

    z = _safe_numeric(df[value_col]).to_numpy().astype(float)
    z = z - z.mean()
    denom = float((z ** 2).sum())
    if denom == 0:
        return np.nan, np.nan

    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=min(k + 1, len(df)))
    nbrs = idx[:, 1:]

    n = len(df)
    row_std_w = np.full(nbrs.shape, 1.0 / max(nbrs.shape[1], 1), dtype=float)
    s0 = float(row_std_w.sum())

    num = 0.0
    for i in range(n):
        zi = z[i]
        jidx = nbrs[i]
        wij = row_std_w[i]
        num += float((wij * zi * z[jidx]).sum())

    I = float((n / s0) * (num / denom)) if s0 > 0 else np.nan

    # Permutation p-value (two-sided)
    perm_vals = []
    for _ in range(n_perm):
        zp = np.random.permutation(z)
        nnum = 0.0
        for i in range(n):
            zi = zp[i]
            jidx = nbrs[i]
            wij = row_std_w[i]
            nnum += float((wij * zi * zp[jidx]).sum())
        perm_vals.append(float((n / s0) * (nnum / denom)) if s0 > 0 else np.nan)

    perm_arr = np.asarray(perm_vals, dtype=float)
    pval = float((np.sum(np.abs(perm_arr) >= abs(I)) + 1) / (len(perm_arr) + 1))
    return I, pval


def run_spatial_and_contribution(panel: pd.DataFrame, coef_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = _prepare_columns(
        panel,
        numeric_terms=BASE_NUMERIC + ["pixel_cloud_proxy", "is_cbsa", "is_urban_area", "pop_density_log1p"],
        cat_terms=["land_use_group", "event_disaster_type", "urban_rural_stratum"],
    )

    formula = _build_formula(
        "delta_ntl",
        numeric_terms=BASE_NUMERIC + ["pixel_cloud_proxy", "is_cbsa", "is_urban_area", "pop_density_log1p"],
        cat_terms=["land_use_group", "event_disaster_type", "urban_rural_stratum"],
    )

    ols = smf.ols(formula, data=p).fit(cov_type="HC1")
    p["ols_resid"] = ols.resid

    moran_rows = []
    for event_id, grp in p.groupby("event_id"):
        I, pval = _moran_i_knn(grp, "ols_resid", k=8, n_perm=199)
        moran_rows.append({"event_id": event_id, "moran_i": I, "p_value": pval, "n_obs": int(len(grp))})
    moran = pd.DataFrame(moran_rows)

    # Spatial cluster-robust SE (5km bins)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = tr.transform(p["lon"].to_numpy(), p["lat"].to_numpy())
    p["cluster_5km"] = (np.floor(x / 5000).astype(int).astype(str) + "_" + np.floor(y / 5000).astype(int).astype(str))

    ols_cluster = smf.ols(formula, data=p).fit(cov_type="cluster", cov_kwds={"groups": p["cluster_5km"]})
    se_cmp = pd.DataFrame(
        [
            {
                "term": "in_buffer",
                "coef_hc1": float(ols.params.get("in_buffer", np.nan)),
                "se_hc1": float(ols.bse.get("in_buffer", np.nan)),
                "p_hc1": float(ols.pvalues.get("in_buffer", np.nan)),
                "coef_cluster": float(ols_cluster.params.get("in_buffer", np.nan)),
                "se_cluster": float(ols_cluster.bse.get("in_buffer", np.nan)),
                "p_cluster": float(ols_cluster.pvalues.get("in_buffer", np.nan)),
            }
        ]
    )

    # Contribution scorecards
    keep_models = {"OLS", "MixedLM", "Logit", "Cox", "AFT"}
    cf = coef_all[coef_all["model"].isin(keep_models)].copy()
    if cf.empty:
        score = pd.DataFrame(
            columns=[
                "model",
                "spec_id",
                "feature",
                "effect_direction",
                "effect_size",
                "uncertainty",
                "loeo_sign_consistency",
                "net_contribution_label",
                "evidence_ref",
            ]
        )
        rank = pd.DataFrame(columns=["model", "feature", "rank", "effect_size_abs"])
    else:
        rows = []
        for (model, spec, feat), grp in cf.groupby(["model", "spec_id", "feature"], dropna=False):
            vals = _safe_numeric(grp["coef"])
            mean_coef = float(vals.mean())
            std_coef = float(vals.std(ddof=0))
            sign_cons = float(np.mean(np.sign(vals) == np.sign(mean_coef))) if len(vals) else np.nan

            if not np.isfinite(mean_coef):
                direction = "unknown"
            elif mean_coef > 0:
                direction = "positive"
            elif mean_coef < 0:
                direction = "negative"
            else:
                direction = "neutral"

            if not np.isfinite(sign_cons) or sign_cons < 0.6:
                label = "unstable"
            elif direction == "positive":
                label = "positive_contribution"
            elif direction == "negative":
                label = "negative_contribution"
            else:
                label = "neutral"

            rows.append(
                {
                    "model": model,
                    "spec_id": spec,
                    "feature": feat,
                    "effect_direction": direction,
                    "effect_size": mean_coef,
                    "uncertainty": std_coef,
                    "loeo_sign_consistency": sign_cons,
                    "net_contribution_label": label,
                    "evidence_ref": "coef_loeo",
                }
            )
        score = pd.DataFrame(rows)
        rank = (
            score.assign(effect_size_abs=lambda d: d["effect_size"].abs())
            .sort_values(["model", "effect_size_abs"], ascending=[True, False])
            .groupby("model", as_index=False)
            .head(30)
        )
        rank["rank"] = rank.groupby("model").cumcount() + 1
        rank = rank[["model", "feature", "rank", "effect_size_abs"]]

    moran.to_csv(MORAN_PATH, index=False)
    se_cmp.to_csv(SPATIAL_SE_PATH, index=False)
    score.to_csv(CONTRIB_SCORE_PATH, index=False)
    rank.to_csv(CONTRIB_RANK_PATH, index=False)
    return moran, se_cmp, score, rank


def _compute_extreme_candidates() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not SHIFT_V3_PATH.exists() or not FOLD_V3R1_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()

    shift = pd.read_csv(SHIFT_V3_PATH)
    fold = pd.read_csv(FOLD_V3R1_PATH)

    ev_shift = shift[shift["diagnostic_type"] == "event_shift"].pivot_table(
        index="event_id", columns="metric_name", values="value", aggfunc="mean"
    )
    ev_shift = ev_shift.reset_index()
    if "smd_mean" not in ev_shift.columns:
        ev_shift["smd_mean"] = np.nan
    if "psi_mean" not in ev_shift.columns:
        ev_shift["psi_mean"] = np.nan

    # Fold performance by event
    perf = fold.pivot_table(
        index="fold_event",
        columns="model",
        values=["auc", "c_index", "rmse"],
        aggfunc="mean",
    )
    perf.columns = [f"{a}_{b}" for a, b in perf.columns]
    perf = perf.reset_index().rename(columns={"fold_event": "event_id"})

    df = ev_shift.merge(perf, on="event_id", how="left")

    smd_thr = float(_safe_numeric(df["smd_mean"]).quantile(2 / 3))
    psi_thr = float(_safe_numeric(df["psi_mean"]).quantile(2 / 3))

    auc_q = float(_safe_numeric(df.get("auc_Logit", pd.Series(np.nan))).quantile(1 / 3))
    cox_q = float(_safe_numeric(df.get("c_index_Cox", pd.Series(np.nan))).quantile(1 / 3))
    aft_q = float(_safe_numeric(df.get("c_index_AFT", pd.Series(np.nan))).quantile(1 / 3))
    rmse_q = float(_safe_numeric(df.get("rmse_OLS", pd.Series(np.nan))).quantile(2 / 3))

    bad_count = []
    for _, r in df.iterrows():
        c = 0
        if pd.notna(r.get("auc_Logit")) and float(r.get("auc_Logit")) <= auc_q:
            c += 1
        c_best = np.nanmax([r.get("c_index_Cox", np.nan), r.get("c_index_AFT", np.nan)])
        surv_thr = max(cox_q, aft_q)
        if np.isfinite(c_best) and c_best <= surv_thr:
            c += 1
        if pd.notna(r.get("rmse_OLS")) and float(r.get("rmse_OLS")) >= rmse_q:
            c += 1
        bad_count.append(c)
    df["bad_metric_count"] = bad_count

    df["high_shift_flag"] = ((df["smd_mean"] >= smd_thr) | (df["psi_mean"] >= psi_thr)).astype(int)
    df["poor_perf_flag"] = (df["bad_metric_count"] >= 2).astype(int)
    df["extreme_candidate"] = ((df["high_shift_flag"] == 1) & (df["poor_perf_flag"] == 1)).astype(int)

    shift_rank = _zscore(df["smd_mean"].rank(method="average")) + _zscore(df["psi_mean"].rank(method="average"))
    perf_rank = _zscore(df["bad_metric_count"].rank(method="average"))
    df["extreme_score"] = shift_rank + perf_rank
    df = df.sort_values("extreme_score", ascending=False).copy()
    df["extreme_candidate_soft"] = 0
    if int(df["extreme_candidate"].sum()) == 0 and len(df) > 0:
        soft_n = min(2, len(df))
        df.loc[df.index[:soft_n], "extreme_candidate_soft"] = 1

    candidates = df[["event_id", "high_shift_flag", "poor_perf_flag", "extreme_candidate"]].copy()
    candidates["extreme_candidate_soft"] = df["extreme_candidate_soft"].to_numpy()
    score = df.copy()
    score["source_ref"] = "cross_event_shift_diagnostics_v3 + cross_event_fold_metrics_v3r1"

    candidates.to_csv(EXTREME_CANDIDATE_PATH, index=False)
    score.to_csv(EXTREME_SCORE_PATH, index=False)
    return candidates, score


def run_extreme_drop_sensitivity(
    panel: pd.DataFrame,
    recovery: pd.DataFrame,
    score: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    all_events = sorted(panel["event_id"].unique().tolist())

    # Main full spec for this sensitivity
    num_terms = BASE_NUMERIC + ["pixel_cloud_proxy", "is_cbsa", "is_urban_area", "pop_density_log1p"]
    cat_terms = ["land_use_group", "event_disaster_type", "urban_rural_stratum"]

    drop_scenarios: Dict[str, List[str]] = {}
    drop_scenarios["drop_none"] = all_events.copy()

    score = score.sort_values("extreme_score", ascending=False).copy()
    extreme_events = score.loc[score["extreme_candidate"] == 1, "event_id"].tolist()
    used_soft = False
    if len(extreme_events) == 0:
        extreme_events = score.loc[score.get("extreme_candidate_soft", 0) == 1, "event_id"].tolist()
        used_soft = True

    for ev in extreme_events:
        drop_scenarios[f"drop1_{ev}"] = [e for e in all_events if e != ev]

    top2 = extreme_events[:2]
    if len(top2) == 2:
        drop_scenarios["drop2_top_extreme"] = [e for e in all_events if e not in set(top2)]

    non_ext = score.loc[score["extreme_candidate"] == 0, "event_id"].tolist()
    if non_ext:
        ctrl_event = non_ext[0]
        drop_scenarios[f"control_drop_{ctrl_event}"] = [e for e in all_events if e != ctrl_event]
    else:
        ctrl_event = None

    fold_parts, agg_parts = [], []
    for scenario, kept_events in drop_scenarios.items():
        if len(kept_events) < 3:
            continue
        res = run_loeo_spec(
            panel,
            recovery,
            numeric_terms=num_terms,
            cat_terms=cat_terms,
            experiment_family="extreme_drop",
            spec_id=scenario,
            allowed_events=kept_events,
        )
        fold_parts.append(res.fold_df)
        agg_parts.append(res.agg_df)

    fold = pd.concat(fold_parts, ignore_index=True) if fold_parts else pd.DataFrame()
    agg = pd.concat(agg_parts, ignore_index=True) if agg_parts else pd.DataFrame()

    if not fold.empty:
        fold.to_csv(EXTREME_DROP_METRIC_PATH, index=False)
    else:
        pd.DataFrame(columns=["experiment_family", "spec_id", "fold_event", "model", "rmse", "mae", "auc", "brier", "c_index", "coef_in_buffer", "notes"]).to_csv(
            EXTREME_DROP_METRIC_PATH, index=False
        )

    if not agg.empty:
        agg.to_csv(EXTREME_DROP_AGG_PATH, index=False)
    else:
        pd.DataFrame(columns=["experiment_family", "spec_id", "model", "rmse", "mae", "auc", "brier", "c_index", "coef_in_buffer", "n_folds"]).to_csv(
            EXTREME_DROP_AGG_PATH, index=False
        )

    decision = {
        "logic": "extreme_drop_improves_and_control_not",
        "extreme_candidates": extreme_events,
        "used_soft_candidates": used_soft,
        "control_drop_event": ctrl_event,
        "decision": "insufficient_evidence",
        "evidence": {},
    }

    try:
        if not agg.empty:
            base = agg[(agg["spec_id"] == "drop_none") & (agg["model"] == "Logit")]["auc"].mean()
            extreme_best = agg[agg["spec_id"].str.startswith("drop1_", na=False) & (agg["model"] == "Logit")]["auc"].max()
            control_auc = agg[agg["spec_id"].str.startswith("control_drop_", na=False) & (agg["model"] == "Logit")]["auc"].mean()
            decision["evidence"] = {
                "base_logit_auc": None if pd.isna(base) else float(base),
                "best_extreme_drop_logit_auc": None if pd.isna(extreme_best) else float(extreme_best),
                "control_drop_logit_auc": None if pd.isna(control_auc) else float(control_auc),
            }
            if pd.notna(base) and pd.notna(extreme_best):
                if (extreme_best - base) >= 0.02 and (pd.isna(control_auc) or (control_auc - base) < 0.01):
                    decision["decision"] = "extreme_event_dominance_supported"
                else:
                    decision["decision"] = "structural_shift_not_single_event"
    except Exception:
        pass

    EXTREME_DECISION_PATH.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    return fold, agg, decision


def _write_future_plan_files() -> None:
    MASTER_PLAN_PATH.write_text(
        "# Exploration Master Plan (V2)\n\n"
        "- Anchor: v3r1 outputs as round0\n"
        "- Sample lock: `sample_lock_flag=1`\n"
        "- Validation: LOEO by event (6 folds)\n"
        "- Stop rule: if two consecutive lines are marginal (<+0.02 AUC and <+0.01 survival best), stop\n"
        "- This round adds six lines: cloud, mask, urban+pop, spatial, contribution, extreme-drop sensitivity\n",
        encoding="utf-8",
    )

    CLOUD_PLAN_PATH.write_text(
        "# Cloud/Coverage Importance Plan\n\n"
        "Specs:\n"
        "- C0: no cloud\n"
        "- C1: pixel_cloud_proxy\n"
        "- C2: + pre/post valid ratio\n"
        "- C3: + missing_cloud_flag (robustness)\n\n"
        "Outputs:\n"
        "- cloud_ablation_fold_metrics.csv\n"
        "- cloud_ablation_aggregate_metrics.csv\n"
        "- cloud_feature_importance.csv\n",
        encoding="utf-8",
    )

    MASK_PLAN_PATH.write_text(
        "# Noise Masking Plan\n\n"
        "- M1 hard mask: 11,12,90,95\n"
        "- M2 extended mask: +31,41,42,43,52,71\n"
        "- Agriculture (81,82) remains kept in this round\n"
        "- Compare coverage and model metrics under M0/M1/M2\n",
        encoding="utf-8",
    )

    URBAN_PLAN_PATH.write_text(
        "# Urban-Rural + Population Plan\n\n"
        "- Urban split: CBSA + Census Urban Area\n"
        "- Population: ACS tract (B01003 / ALAND)\n"
        "- Added fields: is_cbsa, is_urban_area, urban_rural_stratum, pop_density_per_km2, pop_density_log1p\n"
        "- Models: full + urban-only + rural-only\n",
        encoding="utf-8",
    )

    SPATIAL_PLAN_PATH.write_text(
        "# Spatial Autocorr + Contribution Plan\n\n"
        "- Moran's I on event-level OLS residuals\n"
        "- If spatial dependence appears, compare HC1 vs spatial-clustered SE\n"
        "- Build feature contribution scorecard by model\n",
        encoding="utf-8",
    )

    EXTREME_PLAN_PATH.write_text(
        "# Extreme Event Sensitivity Plan\n\n"
        "Candidate rule (double threshold):\n"
        "- High shift: smd_mean or psi_mean in top 33%\n"
        "- Poor prediction: >=2 task metrics in worst 33%\n"
        "- Candidate if both true\n\n"
        "Sensitivity only:\n"
        "- drop-1 for each candidate\n"
        "- drop-2 for top2 candidates\n"
        "- control-drop for one non-extreme event\n"
        "- Keep main conclusion unchanged\n",
        encoding="utf-8",
    )


def _plot_experiment_summary(cloud_agg: pd.DataFrame, mask_agg: pd.DataFrame, urban_agg: pd.DataFrame) -> None:
    FIG_EXP_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    c = cloud_agg[(cloud_agg["model"] == "Logit") & (cloud_agg["spec_id"].isin(["C0", "C1", "C2", "C3"]))].copy()
    if not c.empty:
        sns.barplot(data=c, x="spec_id", y="auc", ax=axes[0], color="#2a9d8f")
    axes[0].set_title("Cloud Ablation: Logit AUC")
    axes[0].set_xlabel("Spec")
    axes[0].set_ylabel("AUC")

    m = mask_agg[(mask_agg["model"] == "Logit") & (mask_agg["spec_id"].isin(["M0", "M1", "M2"]))].copy()
    if not m.empty:
        sns.barplot(data=m, x="spec_id", y="auc", ax=axes[1], color="#e76f51")
    axes[1].set_title("Noise Mask: Logit AUC")
    axes[1].set_xlabel("Spec")
    axes[1].set_ylabel("AUC")

    u = urban_agg[(urban_agg["model"] == "Logit") & (urban_agg["spec_id"].isin(["UR_full", "UR_urban", "UR_rural"]))].copy()
    if not u.empty:
        sns.barplot(data=u, x="spec_id", y="auc", ax=axes[2], color="#264653")
    axes[2].set_title("Urban Split: Logit AUC")
    axes[2].set_xlabel("Spec")
    axes[2].set_ylabel("AUC")

    fig.tight_layout()
    fig.savefig(FIG_EXP_DIR / "exploration_auc_compare.png", dpi=220)
    plt.close(fig)


def _update_index() -> None:
    if INDEX_PATH.exists():
        text = INDEX_PATH.read_text(encoding="utf-8")
    else:
        text = "# Modeling Report Index\n\n"

    lines_to_add = [
        "- `project/modeling_report/10_exploration_upgrade_report.md`",
        "- Exploration V2 outputs: `project/modeling/output/cloud_ablation_aggregate_metrics.csv`, `project/modeling/output/noise_mask_experiment_metrics.csv`, `project/modeling/output/urban_rural_model_comparison.csv`, `project/modeling/output/spatial_autocorr_morans_i.csv`, `project/modeling/output/extreme_event_drop_aggregate_v1.csv`",
    ]

    for line in lines_to_add:
        if line not in text:
            text = text.rstrip() + "\n" + line + "\n"

    INDEX_PATH.write_text(text, encoding="utf-8")


def _write_report(
    cloud_agg: pd.DataFrame,
    mask_agg: pd.DataFrame,
    urban_agg: pd.DataFrame,
    moran: pd.DataFrame,
    contrib: pd.DataFrame,
    extreme_agg: pd.DataFrame,
    decision: Dict[str, object],
    pop_quality: pd.DataFrame,
) -> None:
    def _metric(df: pd.DataFrame, spec: str, model: str, col: str) -> str:
        if df.empty:
            return "NA"
        s = df[(df["spec_id"] == spec) & (df["model"] == model)][col]
        if s.empty or s.isna().all():
            return "NA"
        return f"{float(s.mean()):.4f}"

    c0_auc = _metric(cloud_agg, "C0", "Logit", "auc")
    c1_auc = _metric(cloud_agg, "C1", "Logit", "auc")
    c2_auc = _metric(cloud_agg, "C2", "Logit", "auc")
    m0_auc = _metric(mask_agg, "M0", "Logit", "auc")
    m2_auc = _metric(mask_agg, "M2", "Logit", "auc")
    ur_full_auc = _metric(urban_agg, "UR_full", "Logit", "auc")

    moran_sig = moran[moran["p_value"] < 0.05] if not moran.empty else pd.DataFrame()
    pop_note = "NA"
    if not pop_quality.empty:
        hit = pop_quality[pop_quality["feature"] == "acs_pop_density"]
        if not hit.empty:
            pop_note = f"status={hit['status'].iloc[0]}, coverage={float(hit['coverage_ratio'].iloc[0]):.3f}"

    contrib_dedup = (
        contrib.groupby(["model", "feature"], as_index=False)
        .agg(
            effect_size=("effect_size", "mean"),
            loeo_sign_consistency=("loeo_sign_consistency", "mean"),
            net_contribution_label=("net_contribution_label", "first"),
        )
    )

    top_pos = (
        contrib_dedup[contrib_dedup["net_contribution_label"] == "positive_contribution"]
        .sort_values("effect_size", ascending=False)
        .head(5)
    )
    top_neg = (
        contrib_dedup[contrib_dedup["net_contribution_label"] == "negative_contribution"]
        .sort_values("effect_size")
        .head(5)
    )

    report = [
        "# Exploration Upgrade Report (V2) / 六事件探索增强报告（V2）",
        "",
        "## Objective",
        "在不扩事件集合前提下，验证云量、噪声 mask、城乡/人口、空间依赖与极端事件敏感性是否能解释并改善跨事件泛化表现。",
        "",
        "## Experiment Matrix",
        "- Cloud: C0/C1/C2/C3",
        "- Noise Mask: M0/M1/M2",
        "- Urban-Rural: UR_full/UR_urban/UR_rural",
        "- Spatial: Moran's I + cluster SE",
        "- Extreme sensitivity: drop-1/drop-2/control-drop",
        "",
        "## Cloud Importance Findings",
        f"- Logit AUC: C0={c0_auc}, C1={c1_auc}, C2={c2_auc}",
        "- 判定逻辑：若 AUC 提升但 Brier 恶化，则归类为有代价提升。",
        "",
        "## Noise Mask Findings",
        f"- Logit AUC: M0={m0_auc}, M2={m2_auc}",
        "- M1/M2 通过去除高噪声地物类别检验图像背景噪声对可泛化性的影响。",
        "",
        "## Urban-Rural + Population Findings",
        f"- UR_full Logit AUC={ur_full_auc}",
        f"- Population source quality: {pop_note}",
        "- 城乡与人口密度用于检验发电机相关韧性是否存在结构性分层。",
        "",
        "## Spatial Autocorrelation Findings",
        f"- Significant Moran's I events: {int(len(moran_sig))}/{int(len(moran)) if len(moran)>0 else 0}",
        "- 已输出 HC1 vs spatial cluster SE 对照，避免显著性高估。",
        "",
        "## Indicator Contribution Verdict",
        "### Positive contribution (top)",
    ]

    if top_pos.empty:
        report.append("- NA")
    else:
        for _, r in top_pos.iterrows():
            report.append(f"- {r['model']} | {r['feature']} | effect={r['effect_size']:.4f} | stability={r['loeo_sign_consistency']:.2f}")

    report.append("")
    report.append("### Negative contribution (top)")
    if top_neg.empty:
        report.append("- NA")
    else:
        for _, r in top_neg.iterrows():
            report.append(f"- {r['model']} | {r['feature']} | effect={r['effect_size']:.4f} | stability={r['loeo_sign_consistency']:.2f}")

    report.extend(
        [
            "",
            "## Extreme-event Identification & Drop Sensitivity",
            f"- Decision: {decision.get('decision', 'NA')}",
            f"- Extreme candidates: {', '.join(decision.get('extreme_candidates', [])) if decision.get('extreme_candidates') else 'none'}",
            "- 该结论仅用于敏感性，不改变主规格口径。",
            "",
            "## Key Outputs",
            "- `project/modeling/output/cloud_ablation_aggregate_metrics.csv`",
            "- `project/modeling/output/noise_mask_experiment_metrics.csv`",
            "- `project/modeling/output/urban_rural_model_comparison.csv`",
            "- `project/modeling/output/spatial_autocorr_morans_i.csv`",
            "- `project/modeling/output/feature_contribution_scorecard.csv`",
            "- `project/modeling/output/extreme_event_drop_aggregate_v1.csv`",
            "",
            "## Figures",
            "- `project/modeling_report/figures/exploration_v2/exploration_auc_compare.png`",
        ]
    )

    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def _run_v2_impl() -> int:
    ensure_directories()
    init_tracking_files()
    FIG_EXP_DIR.mkdir(parents=True, exist_ok=True)

    ctx = RunContext(issues=[])

    append_progress("Exploration V2 pipeline started")

    if not PANEL_IN_PATH.exists():
        raise FileNotFoundError(f"Missing panel input: {PANEL_IN_PATH}")

    panel = pd.read_parquet(PANEL_IN_PATH)
    panel = panel[panel.get("sample_lock_flag", 1) == 1].copy()
    panel = _prepare_noise_groups(panel)

    # Urban + population enrich
    panel, pop_quality = attach_urban_population(panel, ctx)
    pop_quality.to_csv(POP_QUALITY_PATH, index=False)

    # Ensure required added fields exist
    for c in ADDED_FIELDS:
        if c not in panel.columns:
            panel[c] = np.nan

    panel.to_parquet(PANEL_OUT_PATH, index=False)

    # Recovery panel synced with enriched features
    rec = build_recovery_from_panel(panel)

    append_progress("Exploration V2: cloud ablation")
    cloud_fold, cloud_agg, cloud_coef = run_cloud_ablation(panel, rec)

    append_progress("Exploration V2: noise masking")
    mask_fold, mask_agg, mask_coef = run_noise_mask(panel, rec)

    append_progress("Exploration V2: urban-rural + population")
    urban_fold, urban_agg, urban_coef = run_urban_population(panel, rec)

    coef_all = pd.concat([cloud_coef, mask_coef, urban_coef], ignore_index=True)

    append_progress("Exploration V2: spatial diagnostics + contribution")
    moran, se_cmp, contrib, contrib_rank = run_spatial_and_contribution(panel, coef_all)

    append_progress("Exploration V2: extreme-event sensitivity")
    candidates, score = _compute_extreme_candidates()
    if score.empty:
        # safe fallback if baseline files unavailable
        pd.DataFrame(columns=["event_id", "high_shift_flag", "poor_perf_flag", "extreme_candidate"]).to_csv(EXTREME_CANDIDATE_PATH, index=False)
        pd.DataFrame(columns=["event_id", "extreme_score", "source_ref"]).to_csv(EXTREME_SCORE_PATH, index=False)
        extreme_fold = pd.DataFrame()
        extreme_agg = pd.DataFrame()
        decision = {"decision": "missing_baseline_files", "extreme_candidates": []}
        EXTREME_DECISION_PATH.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        extreme_fold, extreme_agg, decision = run_extreme_drop_sensitivity(panel, rec, score)

    append_progress("Exploration V2: write plan docs/report/index")
    _write_future_plan_files()
    _plot_experiment_summary(cloud_agg, mask_agg, urban_agg)
    _write_report(cloud_agg, mask_agg, urban_agg, moran, contrib, extreme_agg, decision, pop_quality)
    _update_index()

    save_issue_log(ctx)
    append_progress("Exploration V2 pipeline completed")

    return 0
# ----------------------------
# Unified CLI
# ----------------------------

import argparse


def cmd_run_v2() -> int:
    return _run_v2_impl()


def cmd_full_run() -> int:
    return _run_v2_impl()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Unified exploration/sensitivity entrypoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('run-v2', help='Run full exploration V2 pipeline')
    sub.add_parser('cloud-ablation', help='Run exploration V2 cloud ablation bundle')
    sub.add_parser('noise-mask', help='Run exploration V2 noise mask bundle')
    sub.add_parser('urban-rural', help='Run exploration V2 urban-rural bundle')
    sub.add_parser('spatial-diagnostics', help='Run exploration V2 spatial diagnostics bundle')
    sub.add_parser('extreme-event-sensitivity', help='Run exploration V2 extreme-event sensitivity bundle')
    sub.add_parser('full-run', help='Run full exploration V2 pipeline')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {'run-v2', 'cloud-ablation', 'noise-mask', 'urban-rural', 'spatial-diagnostics', 'extreme-event-sensitivity', 'full-run'}:
        return _run_v2_impl()
    parser.error(f'Unknown command: {args.command}')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
