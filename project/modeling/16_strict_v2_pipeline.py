#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

from pipeline_lib import (
    CONFIG_DEFAULTS,
    OUTPUT_DIR,
    PIXEL_DIR,
    REPORT_DIR,
    ROOT,
    RunContext,
    append_progress,
    build_recovery_panel,
    ensure_directories,
    init_tracking_files,
    load_json,
    log_issue,
    save_issue_log,
)


PANEL_FEATURE_PATH = PIXEL_DIR / "all_events_pixel_panel_v1_feature_upgrade.parquet"
SAMPLE_LOCK_PATH = PIXEL_DIR / "sample_lock_cohort_v1.parquet"

MANIFEST_PATH = OUTPUT_DIR / "feature_spec_manifest_v2_strict.json"
VIF_PATH = OUTPUT_DIR / "multicollinearity_vif_v2_strict.csv"
SUMMARY_PATH = OUTPUT_DIR / "model_summary_feature_upgrade_v2_strict.csv"
SAMPLE_AUDIT_PATH = OUTPUT_DIR / "sample_alignment_audit_v2_strict.csv"
COX_DIAG_PATH = OUTPUT_DIR / "cox_diagnostics_extended_v2_strict.csv"
LOGO_FOLD_PATH = OUTPUT_DIR / "logo_fold_metrics_v2_strict.csv"
LOGO_AGG_PATH = OUTPUT_DIR / "logo_aggregate_metrics_v2_strict.csv"
MISSING_FLAG_AUDIT_PATH = OUTPUT_DIR / "missing_flag_audit_v2_strict.csv"

OLS_RESULT_PATH = OUTPUT_DIR / "ols_results_feature_upgrade_v2_strict.csv"
MIXED_RESULT_PATH = OUTPUT_DIR / "mixedlm_results_feature_upgrade_v2_strict.csv"
LOGIT_RESULT_PATH = OUTPUT_DIR / "logit_results_feature_upgrade_v2_strict.csv"
COX_RESULT_PATH = OUTPUT_DIR / "cox_results_feature_upgrade_v2_strict.csv"

VARIANTS = [
    "baseline_locked_v2_strict",
    "nlcd_locked_v2_strict",
    "full_locked_v2_strict",
]
FULL_VARIANT = "full_locked_v2_strict"

RAW_FULL_FEATURES = [
    "osm_dist_any_m",
    "osm_power_count_1000m",
    "osm_medical_count_1000m",
    "pixel_cloud_proxy",
]
SCALED_BASE_FEATURES = ["pre_mean_ntl_centered"]
SCALED_FULL_FEATURES = [
    "pre_mean_ntl_centered",
    "osm_dist_any_m",
    "osm_power_count_1000m",
    "osm_medical_count_1000m",
    "pixel_cloud_proxy",
]
SCALED_SUFFIX = "_z"


def _coef_table_from_result(result, model_name: str, variant: str, kind: str = "linear") -> pd.DataFrame:
    ci = result.conf_int()
    table = pd.DataFrame(
        {
            "model": model_name,
            "variant": variant,
            "term": result.params.index,
            "coef": result.params.values,
            "std_err": getattr(result, "bse", pd.Series(np.nan, index=result.params.index)).values,
            "stat": getattr(
                result,
                "tvalues",
                getattr(result, "zvalues", pd.Series(np.nan, index=result.params.index)),
            ).values,
            "p_value": getattr(result, "pvalues", pd.Series(np.nan, index=result.params.index)).values,
            "ci_low": ci.iloc[:, 0].values,
            "ci_high": ci.iloc[:, 1].values,
            "kind": kind,
        }
    )
    return table


def _fit_logit_with_optimizers(formula: str, data: pd.DataFrame):
    last_error = None
    for method in ["newton", "lbfgs", "bfgs", "powell"]:
        try:
            return smf.logit(formula=formula, data=data).fit(disp=False, method=method, maxiter=400)
        except Exception as e_fit:
            last_error = e_fit
    raise RuntimeError(f"logit fit failed for all optimizers: {last_error}")


def _fit_mixed_with_optimizers(formula: str, data: pd.DataFrame):
    last_error = None
    md = smf.mixedlm(formula, data=data, groups=data["event_id"])
    for method in ["lbfgs", "powell"]:
        try:
            return md.fit(method=method, reml=False)
        except Exception as e_fit:
            last_error = e_fit
    raise RuntimeError(f"mixedlm fit failed for all optimizers: {last_error}")


def _safe_numeric(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if out.notna().any():
        return out.fillna(out.median())
    return out.fillna(0.0)


def _make_center_and_scale(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    out = df.copy()
    out["pre_mean_ntl_centered"] = out["pre_mean_ntl"] - out.groupby("event_id", observed=True)["pre_mean_ntl"].transform("mean")
    scale_stats: Dict[str, Dict[str, float]] = {}

    for col in SCALED_FULL_FEATURES:
        out[col] = _safe_numeric(out[col])
        mu = float(out[col].mean())
        sd = float(out[col].std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        zcol = f"{col}{SCALED_SUFFIX}"
        out[zcol] = (out[col] - mu) / sd
        scale_stats[col] = {"mean": mu, "std": sd}
    return out, scale_stats


def _build_variant_formula(model: str, variant: str, use_scaled: bool) -> str:
    pre_col = f"pre_mean_ntl_centered{SCALED_SUFFIX}" if use_scaled else "pre_mean_ntl_centered"
    full_cols = [f"{c}{SCALED_SUFFIX}" for c in RAW_FULL_FEATURES] if use_scaled else RAW_FULL_FEATURES

    if model == "OLS":
        terms = [f"in_buffer * {pre_col}", "C(event_id)"]
        if variant in {"nlcd_locked_v2_strict", "full_locked_v2_strict"}:
            terms.append("C(land_use_group)")
        if variant == "full_locked_v2_strict":
            terms.extend(full_cols)
        return "delta_ntl ~ " + " + ".join(terms)

    if model == "MixedLM":
        terms = [f"in_buffer * {pre_col}"]
        if variant in {"nlcd_locked_v2_strict", "full_locked_v2_strict"}:
            terms.append("C(land_use_group)")
        if variant == "full_locked_v2_strict":
            terms.extend(full_cols)
        return "delta_ntl ~ " + " + ".join(terms)

    if model == "Logit":
        terms = [f"in_buffer * {pre_col}", "C(event_id)"]
        if variant in {"nlcd_locked_v2_strict", "full_locked_v2_strict"}:
            terms.append("C(land_use_group)")
        if variant == "full_locked_v2_strict":
            terms.extend(full_cols)
        return "is_damaged ~ " + " + ".join(terms)

    raise ValueError(f"Unknown model: {model}")


def _build_transport_formula(model: str, use_scaled: bool) -> str:
    pre_col = f"pre_mean_ntl_centered{SCALED_SUFFIX}" if use_scaled else "pre_mean_ntl_centered"
    full_cols = [f"{c}{SCALED_SUFFIX}" for c in RAW_FULL_FEATURES] if use_scaled else RAW_FULL_FEATURES
    terms = [f"in_buffer * {pre_col}", "C(land_use_group)"] + full_cols
    if model in {"OLS", "MixedLM"}:
        return "delta_ntl ~ " + " + ".join(terms)
    if model == "Logit":
        return "is_damaged ~ " + " + ".join(terms)
    raise ValueError(f"Unknown model: {model}")


def _vif_gate(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["interaction"] = work["in_buffer"] * work["pre_mean_ntl_centered"]
    cols = [
        "in_buffer",
        "pre_mean_ntl_centered",
        "interaction",
        "osm_dist_any_m",
        "osm_power_count_1000m",
        "osm_medical_count_1000m",
        "pixel_cloud_proxy",
    ]
    X = work[cols].copy()
    for c in cols:
        X[c] = _safe_numeric(X[c])
    X = sm.add_constant(X, has_constant="add")

    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vif = float(variance_inflation_factor(X.values, i))
        rows.append({"term": col, "vif": vif, "pass_lt_10": int(vif < 10)})
    out = pd.DataFrame(rows).sort_values("vif", ascending=False)
    out.to_csv(VIF_PATH, index=False)
    return out


def _missing_flag_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["missing_osm_flag", "missing_cloud_flag"]:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False)
        for k, v in vc.items():
            rows.append(
                {
                    "flag_col": col,
                    "scope": "global",
                    "event_id": "all",
                    "value": str(k),
                    "count": int(v),
                    "share": float(v / len(df)),
                }
            )
        ev = df.groupby("event_id", observed=True)[col].mean().reset_index()
        for _, r in ev.iterrows():
            rows.append(
                {
                    "flag_col": col,
                    "scope": "event_mean",
                    "event_id": str(r["event_id"]),
                    "value": "mean",
                    "count": np.nan,
                    "share": float(r[col]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(MISSING_FLAG_AUDIT_PATH, index=False)
    return out


def _make_sample_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v in VARIANTS:
        rows.append(
            {
                "variant": v,
                "n_obs": int(len(df)),
                "n_unique_pixel": int(df["pixel_id"].nunique()),
                "n_event": int(df["event_id"].nunique()),
                "sample_lock_only": 1,
            }
        )
    out = pd.DataFrame(rows)
    out["matches_baseline_n_obs"] = (out["n_obs"] == out["n_obs"].iloc[0]).astype(int)
    out.to_csv(SAMPLE_AUDIT_PATH, index=False)
    return out


def _prepare_cox_design(rec: pd.DataFrame, variant: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    base_terms = ["in_buffer", f"pre_mean_ntl_centered{SCALED_SUFFIX}"]
    full_terms = [f"{c}{SCALED_SUFFIX}" for c in RAW_FULL_FEATURES]

    work = rec[["recovery_days", "event_observed", "event_id"] + base_terms + ["land_use_group"] + full_terms].copy()
    work["land_use_group"] = work["land_use_group"].fillna("unknown").astype(str)

    if variant == "baseline_locked_v2_strict":
        key_terms = base_terms.copy()
        design = work[["recovery_days", "event_observed", "event_id"] + key_terms].copy()
    elif variant == "nlcd_locked_v2_strict":
        key_terms = base_terms.copy()
        lu = pd.get_dummies(work["land_use_group"], prefix="lu", drop_first=True)
        design = pd.concat([work[["recovery_days", "event_observed", "event_id"] + key_terms], lu], axis=1)
    else:
        key_terms = base_terms + full_terms
        lu = pd.get_dummies(work["land_use_group"], prefix="lu", drop_first=True)
        design = pd.concat([work[["recovery_days", "event_observed", "event_id"] + key_terms], lu], axis=1)

    for c in design.columns:
        if c in {"recovery_days", "event_observed", "event_id"}:
            continue
        design[c] = _safe_numeric(design[c])
    return design, key_terms, base_terms + full_terms


def _fit_cox_variant_strict(rec: pd.DataFrame, variant: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    diag_rows: List[Dict[str, object]] = []
    design, key_terms, interaction_candidates = _prepare_cox_design(rec, variant)

    cph = CoxPHFitter()
    cph.fit(design, duration_col="recovery_days", event_col="event_observed", strata=["event_id"])
    ph = proportional_hazard_test(cph, design, time_transform="rank").summary.reset_index().rename(columns={"index": "covariate"})
    key_viol = int(ph[ph["covariate"].isin(key_terms)]["p"].lt(0.05).sum())
    diag_rows.append(
        {
            "variant": variant,
            "step": "strata_base",
            "n_obs": int(len(design)),
            "n_covariates": int(design.shape[1] - 3),
            "key_ph_violations": key_viol,
            "status": "ok",
        }
    )

    selected = cph
    selected_name = "strata_base"
    if key_viol > 1:
        design2 = design.copy()
        design2["log_time"] = np.log(np.clip(design2["recovery_days"].to_numpy(), 1.0, None))
        for c in interaction_candidates:
            if c in design2.columns:
                design2[f"{c}_x_log_time"] = design2[c] * design2["log_time"]

        cph2 = CoxPHFitter(penalizer=0.01)
        cph2.fit(design2, duration_col="recovery_days", event_col="event_observed", strata=["event_id"])
        ph2 = proportional_hazard_test(cph2, design2, time_transform="rank").summary.reset_index().rename(columns={"index": "covariate"})
        key_viol2 = int(ph2[ph2["covariate"].isin(key_terms)]["p"].lt(0.05).sum())
        diag_rows.append(
            {
                "variant": variant,
                "step": "strata_time_interaction",
                "n_obs": int(len(design2)),
                "n_covariates": int(design2.shape[1] - 3),
                "key_ph_violations": key_viol2,
                "status": "ok",
            }
        )
        selected = cph2
        selected_name = "strata_time_interaction"

    summary = selected.summary.reset_index().rename(columns={"index": "covariate"})
    summary["model"] = "Cox"
    summary["variant"] = variant
    summary["cox_spec"] = selected_name
    summary["hazard_ratio"] = np.exp(summary["coef"])

    return summary, pd.DataFrame(diag_rows)


def _summarize_for_report(
    ols_df: pd.DataFrame,
    mixed_df: pd.DataFrame,
    logit_df: pd.DataFrame,
    logit_auc: Dict[str, float],
    cox_df: pd.DataFrame,
    n_obs: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add_row(model: str, variant: str, metric: str, term: str, value, pval, lo, hi, notes: str):
        rows.append(
            {
                "model": model,
                "variant": variant,
                "key_metric": metric,
                "term": term,
                "value": value,
                "p_value": pval,
                "ci_low": lo,
                "ci_high": hi,
                "n_obs": n_obs,
                "notes": notes,
            }
        )

    for df, model in [(ols_df, "OLS"), (mixed_df, "MixedLM")]:
        sub = df[df["term"] == "in_buffer"]
        for _, r in sub.iterrows():
            add_row(model, r["variant"], "coef_in_buffer", "in_buffer", r["coef"], r["p_value"], r["ci_low"], r["ci_high"], "strict_v2")

    sub = logit_df[logit_df["term"] == "in_buffer"]
    for _, r in sub.iterrows():
        add_row(
            "Logit",
            r["variant"],
            "odds_ratio_in_buffer",
            "in_buffer",
            r["odds_ratio"],
            r["p_value"],
            r.get("or_ci_low", np.nan),
            r.get("or_ci_high", np.nan),
            "strict_v2",
        )
    for v, auc in logit_auc.items():
        add_row("Logit", v, "auc", "AUC", auc, np.nan, np.nan, np.nan, "strict_v2")

    csub = cox_df[cox_df["covariate"] == "in_buffer"]
    for _, r in csub.iterrows():
        add_row(
            "Cox",
            r["variant"],
            "hazard_ratio_in_buffer",
            "in_buffer",
            r["hazard_ratio"],
            r["p"],
            float(np.exp(r["coef lower 95%"])),
            float(np.exp(r["coef upper 95%"])),
            f"strict_v2:{r['cox_spec']}",
        )

    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_PATH, index=False)
    return out


def _build_logo(
    ctx: RunContext,
    df_raw: pd.DataFrame,
    df_scaled: pd.DataFrame,
    rec_scaled: pd.DataFrame,
    ref_sign: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    events = sorted(df_raw["event_id"].unique().tolist())

    full_ols_inference = _build_variant_formula("OLS", FULL_VARIANT, use_scaled=False)
    full_ols_transport = _build_transport_formula("OLS", use_scaled=False)
    full_mixed = _build_variant_formula("MixedLM", FULL_VARIANT, use_scaled=False)
    full_logit_inference = _build_variant_formula("Logit", FULL_VARIANT, use_scaled=True)
    full_logit_transport = _build_transport_formula("Logit", use_scaled=True)

    for test_event in events:
        tr_raw = df_raw[df_raw["event_id"] != test_event].copy()
        te_raw = df_raw[df_raw["event_id"] == test_event].copy()
        tr_scaled = df_scaled[df_scaled["event_id"] != test_event].copy()
        te_scaled = df_scaled[df_scaled["event_id"] == test_event].copy()
        tr_rec = rec_scaled[rec_scaled["event_id"] != test_event].copy()
        te_rec = rec_scaled[rec_scaled["event_id"] == test_event].copy()

        # OLS inference (train diagnostics, with event FE)
        ols_inf = smf.ols(full_ols_inference, data=tr_raw).fit(cov_type="HC1")
        pred_tr = ols_inf.predict(tr_raw)
        rows.append(
            {
                "fold_event": test_event,
                "spec": "inference",
                "model": "OLS",
                "rmse": float(math.sqrt(mean_squared_error(tr_raw["delta_ntl"], pred_tr))),
                "mae": float(mean_absolute_error(tr_raw["delta_ntl"], pred_tr)),
                "auc": np.nan,
                "brier": np.nan,
                "calibration_slope": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": float(ols_inf.params.get("in_buffer", np.nan)),
            }
        )

        # OLS transport (test diagnostics, no event FE)
        ols_tr = smf.ols(full_ols_transport, data=tr_raw).fit(cov_type="HC1")
        pred_te = ols_tr.predict(te_raw)
        rows.append(
            {
                "fold_event": test_event,
                "spec": "logo_transport",
                "model": "OLS",
                "rmse": float(math.sqrt(mean_squared_error(te_raw["delta_ntl"], pred_te))),
                "mae": float(mean_absolute_error(te_raw["delta_ntl"], pred_te)),
                "auc": np.nan,
                "brier": np.nan,
                "calibration_slope": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": float(ols_tr.params.get("in_buffer", np.nan)),
            }
        )

        # Mixed inference (train diagnostics)
        mx_inf = _fit_mixed_with_optimizers(full_mixed, tr_raw)
        pred_mx_tr = mx_inf.predict(tr_raw)
        rows.append(
            {
                "fold_event": test_event,
                "spec": "inference",
                "model": "MixedLM",
                "rmse": float(math.sqrt(mean_squared_error(tr_raw["delta_ntl"], pred_mx_tr))),
                "mae": float(mean_absolute_error(tr_raw["delta_ntl"], pred_mx_tr)),
                "auc": np.nan,
                "brier": np.nan,
                "calibration_slope": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": float(mx_inf.params.get("in_buffer", np.nan)),
            }
        )

        # Mixed transport (test diagnostics)
        pred_mx_te = mx_inf.predict(te_raw)
        rows.append(
            {
                "fold_event": test_event,
                "spec": "logo_transport",
                "model": "MixedLM",
                "rmse": float(math.sqrt(mean_squared_error(te_raw["delta_ntl"], pred_mx_te))),
                "mae": float(mean_absolute_error(te_raw["delta_ntl"], pred_mx_te)),
                "auc": np.nan,
                "brier": np.nan,
                "calibration_slope": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": float(mx_inf.params.get("in_buffer", np.nan)),
            }
        )

        # Logit inference (train diagnostics, with event FE)
        lg_inf = _fit_logit_with_optimizers(full_logit_inference, tr_scaled)
        prob_tr = np.asarray(lg_inf.predict(tr_scaled))
        auc_tr = float(roc_auc_score(tr_scaled["is_damaged"], prob_tr)) if tr_scaled["is_damaged"].nunique() > 1 else np.nan
        brier_tr = float(brier_score_loss(tr_scaled["is_damaged"], prob_tr))
        rows.append(
            {
                "fold_event": test_event,
                "spec": "inference",
                "model": "Logit",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": auc_tr,
                "brier": brier_tr,
                "calibration_slope": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": float(lg_inf.params.get("in_buffer", np.nan)),
            }
        )

        # Logit transport (test diagnostics, no event FE)
        lg_tr = _fit_logit_with_optimizers(full_logit_transport, tr_scaled)
        prob_te = np.asarray(lg_tr.predict(te_scaled))
        auc_te = float(roc_auc_score(te_scaled["is_damaged"], prob_te)) if te_scaled["is_damaged"].nunique() > 1 else np.nan
        brier_te = float(brier_score_loss(te_scaled["is_damaged"], prob_te))
        rows.append(
            {
                "fold_event": test_event,
                "spec": "logo_transport",
                "model": "Logit",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": auc_te,
                "brier": brier_te,
                "calibration_slope": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": float(lg_tr.params.get("in_buffer", np.nan)),
            }
        )

        # Cox inference (train diagnostics, with event strata)
        cox_inf, _, _ = _prepare_cox_design(tr_rec, FULL_VARIANT)
        cph_inf = CoxPHFitter(penalizer=0.01)
        cph_inf.fit(cox_inf, duration_col="recovery_days", event_col="event_observed", strata=["event_id"])
        risk_inf = cph_inf.predict_partial_hazard(cox_inf.drop(columns=["recovery_days", "event_observed", "event_id"]))
        cidx_inf = float(concordance_index(cox_inf["recovery_days"], -risk_inf.to_numpy().reshape(-1), cox_inf["event_observed"]))
        rows.append(
            {
                "fold_event": test_event,
                "spec": "inference",
                "model": "Cox",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": np.nan,
                "brier": np.nan,
                "calibration_slope": np.nan,
                "c_index": cidx_inf,
                "coef_in_buffer": float(cph_inf.params_.get("in_buffer", np.nan)),
            }
        )

        # Cox transport (test diagnostics, no event strata/no event FE)
        tr_cox = tr_rec.copy()
        te_cox = te_rec.copy()
        dtr, _, _ = _prepare_cox_design(tr_cox, FULL_VARIANT)
        dte, _, _ = _prepare_cox_design(te_cox, FULL_VARIANT)
        tr_x = dtr.drop(columns=["recovery_days", "event_observed", "event_id"])
        te_x = dte.drop(columns=["recovery_days", "event_observed", "event_id"]).reindex(columns=tr_x.columns, fill_value=0.0)
        fit_df = pd.concat([dtr[["recovery_days", "event_observed"]], tr_x], axis=1)
        cph_tr = CoxPHFitter(penalizer=0.01)
        cph_tr.fit(fit_df, duration_col="recovery_days", event_col="event_observed")
        risk_te = cph_tr.predict_partial_hazard(te_x)
        cidx_te = float(concordance_index(dte["recovery_days"], -risk_te.to_numpy().reshape(-1), dte["event_observed"]))
        rows.append(
            {
                "fold_event": test_event,
                "spec": "logo_transport",
                "model": "Cox",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": np.nan,
                "brier": np.nan,
                "calibration_slope": np.nan,
                "c_index": cidx_te,
                "coef_in_buffer": float(cph_tr.params_.get("in_buffer", np.nan)),
            }
        )

    fold = pd.DataFrame(rows)
    for model, sign in ref_sign.items():
        m = fold["model"] == model
        fold.loc[m, "sign_consistency"] = np.where(
            np.sign(fold.loc[m, "coef_in_buffer"]) == np.sign(sign),
            1.0,
            0.0,
        )
    fold.to_csv(LOGO_FOLD_PATH, index=False)

    agg = (
        fold.groupby(["spec", "model"], dropna=False)[
            ["rmse", "mae", "auc", "brier", "calibration_slope", "c_index", "sign_consistency"]
        ]
        .mean()
        .reset_index()
    )
    agg.to_csv(LOGO_AGG_PATH, index=False)
    return fold, agg


def _update_reports(
    summary: pd.DataFrame,
    vif_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    logo_agg: pd.DataFrame,
) -> None:
    def pick(model: str, variant: str, metric: str) -> Tuple[float, float]:
        sub = summary[
            (summary["model"] == model)
            & (summary["variant"] == variant)
            & (summary["key_metric"] == metric)
        ]
        if sub.empty:
            return np.nan, np.nan
        row = sub.iloc[0]
        return float(row["value"]), float(row["p_value"]) if pd.notna(row["p_value"]) else np.nan

    b = "baseline_locked_v2_strict"
    n = "nlcd_locked_v2_strict"
    f = "full_locked_v2_strict"

    lines = [
        "# Feature Upgrade Report / OSM+Cloud+Sample Lock 升级报告（Strict V2）",
        "",
        "## Objective",
        "本轮按 Strict V2 规则重训：`Proxy only` 云量、`dist_any only` 距离、`strict no fallback`。",
        "",
        "## Strict Spec",
        "- Full 主规格: `in_buffer * pre_mean_ntl_centered + C(event_id) + C(land_use_group) + osm_dist_any_m + osm_power_count_1000m + osm_medical_count_1000m + pixel_cloud_proxy`",
        "- `missing_osm_flag` 与 `missing_cloud_flag` 不入主模型，仅保留审计。",
        "",
        "## Quantitative Comparison",
    ]
    for model, metric in [
        ("OLS", "coef_in_buffer"),
        ("MixedLM", "coef_in_buffer"),
        ("Logit", "odds_ratio_in_buffer"),
        ("Cox", "hazard_ratio_in_buffer"),
    ]:
        vb, pb = pick(model, b, metric)
        vn, pn = pick(model, n, metric)
        vf, pf = pick(model, f, metric)
        lines.append(
            f"- {model} `{metric}`: baseline={vb:.4f} (p={pb:.4g}), nlcd={vn:.4f} (p={pn:.4g}), full={vf:.4f} (p={pf:.4g})"
        )
    auc_b, _ = pick("Logit", b, "auc")
    auc_n, _ = pick("Logit", n, "auc")
    auc_f, _ = pick("Logit", f, "auc")
    lines.append(f"- Logit `AUC`: baseline={auc_b:.4f}, nlcd={auc_n:.4f}, full={auc_f:.4f}")

    lines.extend(
        [
            "",
            "## Collinearity Gate",
            f"- Max VIF: {vif_df['vif'].max():.4f}",
            f"- Gate result (`VIF < 10`): {'PASS' if (vif_df['vif'] < 10).all() else 'FAIL'}",
            f"- Detail: `{VIF_PATH.relative_to(ROOT)}`",
            "",
            "## Missing-Flag Audit",
            f"- `missing_osm_flag` global distribution: `{MISSING_FLAG_AUDIT_PATH.relative_to(ROOT)}`",
            "- 当前数据下 `missing_osm_flag` 为全 0，说明 OSM 加载完整；作为常数项无信息量，已从主模型移除。",
            "",
            "## Outputs",
            f"- Summary: `{SUMMARY_PATH.relative_to(ROOT)}`",
            f"- Cox diagnostics: `{COX_DIAG_PATH.relative_to(ROOT)}`",
            f"- LOEO aggregate: `{LOGO_AGG_PATH.relative_to(ROOT)}`",
        ]
    )
    (REPORT_DIR / "06_feature_upgrade_report.md").write_text("\n".join(lines), encoding="utf-8")

    def lp(spec: str, model: str, col: str) -> str:
        sub = logo_agg[(logo_agg["spec"] == spec) & (logo_agg["model"] == model)]
        if sub.empty or pd.isna(sub.iloc[0][col]):
            return "N/A"
        return f"{float(sub.iloc[0][col]):.4f}"

    lines2 = [
        "# LOEO Validation Report / 按事件留一验证报告（Strict V2）",
        "",
        "## Objective",
        "在 `full_locked_v2_strict` 下执行 LOEO 双规格验证（inference vs transport）。",
        "",
        "## Aggregate Metrics",
        f"- OLS RMSE: inference={lp('inference','OLS','rmse')}, transport={lp('logo_transport','OLS','rmse')}",
        f"- MixedLM RMSE: inference={lp('inference','MixedLM','rmse')}, transport={lp('logo_transport','MixedLM','rmse')}",
        f"- Logit AUC: inference={lp('inference','Logit','auc')}, transport={lp('logo_transport','Logit','auc')}",
        f"- Logit Brier: inference={lp('inference','Logit','brier')}, transport={lp('logo_transport','Logit','brier')}",
        f"- Cox c-index: inference={lp('inference','Cox','c_index')}, transport={lp('logo_transport','Cox','c_index')}",
        "",
        "## Sign Consistency",
        f"- OLS: inference={lp('inference','OLS','sign_consistency')}, transport={lp('logo_transport','OLS','sign_consistency')}",
        f"- MixedLM: inference={lp('inference','MixedLM','sign_consistency')}, transport={lp('logo_transport','MixedLM','sign_consistency')}",
        f"- Logit: inference={lp('inference','Logit','sign_consistency')}, transport={lp('logo_transport','Logit','sign_consistency')}",
        f"- Cox: inference={lp('inference','Cox','sign_consistency')}, transport={lp('logo_transport','Cox','sign_consistency')}",
        "",
        f"Detail files: `{LOGO_FOLD_PATH.relative_to(ROOT)}`, `{LOGO_AGG_PATH.relative_to(ROOT)}`",
    ]
    (REPORT_DIR / "07_logo_validation_report.md").write_text("\n".join(lines2), encoding="utf-8")

    index_file = REPORT_DIR / "index.md"
    if index_file.exists():
        idx = index_file.read_text(encoding="utf-8")
    else:
        idx = "# Modeling Report Index\n\n## Deliverables\n"
    marker = "- `project/modeling_report/07_logo_validation_report.md`"
    if marker not in idx:
        idx = idx.rstrip() + "\n" + marker + "\n"
    strict_line = "- Strict V2 outputs: `project/modeling/output/model_summary_feature_upgrade_v2_strict.csv`, `project/modeling/output/logo_aggregate_metrics_v2_strict.csv`"
    if strict_line not in idx:
        idx = idx.rstrip() + "\n" + strict_line + "\n"
    index_file.write_text(idx, encoding="utf-8")


def main() -> None:
    ensure_directories()
    init_tracking_files()
    append_progress("Strict V2 pipeline started")
    ctx = RunContext(issues=[])

    defaults = load_json(CONFIG_DEFAULTS)
    rec_thr = float(defaults["recovery_threshold"])
    damage_thr = float(defaults["damage_threshold"])

    if not PANEL_FEATURE_PATH.exists():
        raise FileNotFoundError(f"Missing required panel: {PANEL_FEATURE_PATH}")
    df = pd.read_parquet(PANEL_FEATURE_PATH).copy()
    if SAMPLE_LOCK_PATH.exists():
        lock = pd.read_parquet(SAMPLE_LOCK_PATH)[["pixel_id", "sample_lock_flag", "lock_reason"]]
        df = df.drop(columns=["sample_lock_flag", "lock_reason"], errors="ignore").merge(lock, on="pixel_id", how="left")
    if "sample_lock_flag" not in df.columns:
        raise KeyError("sample_lock_flag missing; run feature-upgrade pipeline first.")

    df = df[df["sample_lock_flag"] == 1].copy()
    if df.empty:
        raise RuntimeError("No locked rows available.")
    df["land_use_group"] = df["land_use_group"].fillna("unknown").astype(str)
    df["is_damaged"] = (df["delta_ntl"] < damage_thr).astype(int)

    # Manifest and audits.
    missing_df = _missing_flag_audit(df)
    df_raw, scale_stats = _make_center_and_scale(df)
    _make_sample_audit(df_raw)

    manifest = {
        "version": "v2_strict",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "variants": VARIANTS,
        "full_feature_raw": RAW_FULL_FEATURES,
        "excluded_from_main_spec": [
            "missing_osm_flag",
            "missing_cloud_flag",
            "pixel_pre_valid_ratio",
            "pixel_post_valid_ratio",
            "osm_dist_power_m",
            "osm_any_count_750m",
        ],
        "scale_stats": scale_stats,
        "sample_n_obs": int(len(df_raw)),
        "events": sorted(df_raw["event_id"].unique().tolist()),
        "missing_osm_flag_all_zero": bool((df_raw["missing_osm_flag"] == 0).all()) if "missing_osm_flag" in df_raw.columns else None,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # VIF gate.
    vif_df = _vif_gate(df_raw)
    if not (vif_df["vif"] < 10).all():
        raise RuntimeError(f"VIF gate failed. See {VIF_PATH}")

    # Fit OLS / Mixed / Logit variants (strict, no formula fallback).
    ols_rows = []
    mixed_rows = []
    logit_rows = []
    logit_auc: Dict[str, float] = {}

    for v in VARIANTS:
        # OLS
        f_ols = _build_variant_formula("OLS", v, use_scaled=False)
        ols = smf.ols(f_ols, data=df_raw).fit(cov_type="HC1")
        ols_rows.append(_coef_table_from_result(ols, "OLS", v, kind="linear"))

        # MixedLM
        f_mx = _build_variant_formula("MixedLM", v, use_scaled=False)
        mx = _fit_mixed_with_optimizers(f_mx, df_raw)
        mixed_rows.append(_coef_table_from_result(mx, "MixedLM", v, kind="linear"))

        # Logit
        f_lg = _build_variant_formula("Logit", v, use_scaled=True)
        lg = _fit_logit_with_optimizers(f_lg, df_raw)
        coef = _coef_table_from_result(lg, "Logit", v, kind="logit")
        coef["odds_ratio"] = np.exp(coef["coef"])
        coef["or_ci_low"] = np.exp(coef["ci_low"])
        coef["or_ci_high"] = np.exp(coef["ci_high"])
        logit_rows.append(coef)
        prob = np.asarray(lg.predict(df_raw))
        logit_auc[v] = float(roc_auc_score(df_raw["is_damaged"], prob)) if df_raw["is_damaged"].nunique() > 1 else np.nan

    ols_df = pd.concat(ols_rows, ignore_index=True)
    mixed_df = pd.concat(mixed_rows, ignore_index=True)
    logit_df = pd.concat(logit_rows, ignore_index=True)
    ols_df.to_csv(OLS_RESULT_PATH, index=False)
    mixed_df.to_csv(MIXED_RESULT_PATH, index=False)
    logit_df.to_csv(LOGIT_RESULT_PATH, index=False)

    # Cox (strict, no downgrade to smaller spec).
    rec = build_recovery_panel(ctx=ctx, panel=df_raw, threshold=rec_thr, output_path=None)
    cox_rows = []
    cox_diag_rows = []
    for v in VARIANTS:
        cox_coef, cox_diag = _fit_cox_variant_strict(rec, v)
        cox_rows.append(cox_coef)
        cox_diag_rows.append(cox_diag)

    cox_df = pd.concat(cox_rows, ignore_index=True)
    cox_diag_df = pd.concat(cox_diag_rows, ignore_index=True)
    cox_df.to_csv(COX_RESULT_PATH, index=False)
    cox_diag_df.to_csv(COX_DIAG_PATH, index=False)

    # Strict fail-fast: full variant must exist in all four model outputs.
    checks = [
        ("OLS", not ols_df[ols_df["variant"] == FULL_VARIANT].empty),
        ("MixedLM", not mixed_df[mixed_df["variant"] == FULL_VARIANT].empty),
        ("Logit", not logit_df[logit_df["variant"] == FULL_VARIANT].empty),
        ("Cox", not cox_df[cox_df["variant"] == FULL_VARIANT].empty),
    ]
    fail_models = [m for m, ok in checks if not ok]
    if fail_models:
        raise RuntimeError(f"Fail-fast: full strict variant missing for models: {fail_models}")

    summary = _summarize_for_report(
        ols_df=ols_df,
        mixed_df=mixed_df,
        logit_df=logit_df,
        logit_auc=logit_auc,
        cox_df=cox_df,
        n_obs=len(df_raw),
    )

    # LOEO dual spec.
    ref_sign = {
        "OLS": float(ols_df[(ols_df["variant"] == FULL_VARIANT) & (ols_df["term"] == "in_buffer")]["coef"].iloc[0]),
        "MixedLM": float(mixed_df[(mixed_df["variant"] == FULL_VARIANT) & (mixed_df["term"] == "in_buffer")]["coef"].iloc[0]),
        "Logit": float(logit_df[(logit_df["variant"] == FULL_VARIANT) & (logit_df["term"] == "in_buffer")]["coef"].iloc[0]),
        "Cox": float(cox_df[(cox_df["variant"] == FULL_VARIANT) & (cox_df["covariate"] == "in_buffer")]["coef"].iloc[0]),
    }
    _, logo_agg = _build_logo(ctx, df_raw=df_raw, df_scaled=df_raw, rec_scaled=rec, ref_sign=ref_sign)

    # Reports
    _update_reports(summary=summary, vif_df=vif_df, missing_df=missing_df, logo_agg=logo_agg)
    save_issue_log(ctx)
    append_progress("Strict V2 pipeline finished")


if __name__ == "__main__":
    main()
