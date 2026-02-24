#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from patsy import build_design_matrices, dmatrices
from pandas.api.types import CategoricalDtype
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold

from pipeline_lib import (
    CONFIG_DEFAULTS,
    OUTPUT_DIR,
    PIXEL_DIR,
    REPORT_DIR,
    ROOT,
    RunContext,
    build_recovery_panel,
    ensure_directories,
    init_tracking_files,
    load_json,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

PANEL_V3_PATH = PIXEL_DIR / "all_events_pixel_panel_v1_cross_event_v3.parquet"
BASE_AGG_PATH = OUTPUT_DIR / "cross_event_aggregate_metrics_v3.csv"

ROUND_COMP_PATH = OUTPUT_DIR / "cross_event_round_comparison_v3x.csv"
STOP_DECISION_PATH = OUTPUT_DIR / "cross_event_stop_decision_v3x.json"
REPORT_PATH = REPORT_DIR / "09_cross_event_stabilization_report.md"
INDEX_PATH = REPORT_DIR / "index.md"

LOCAL_NUMERIC_BASE = [
    "osm_dist_any_m",
    "osm_power_count_1000m",
    "osm_medical_count_1000m",
    "pixel_cloud_proxy",
    "urban_share_1km",
    "water_share_1km",
    "developed_high_share_1km",
]
CAT_TERMS_BASE = ["land_use_group", "event_disaster_type"]

THRESH_AUC = 0.02
THRESH_SURV = 0.01
MAX_ROUNDS = 3


class StabilizationError(RuntimeError):
    pass


def _safe_numeric(s: pd.Series, default: float = 0.0) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if out.notna().any():
        return out.fillna(out.median())
    return out.fillna(default)


def _event_inverse_sqrt_weights(event_series: pd.Series) -> pd.Series:
    cnt = event_series.value_counts()
    return event_series.map(lambda e: 1.0 / math.sqrt(float(cnt.get(e, 1))))


def _fill_frame(
    df: pd.DataFrame,
    numeric_terms: Sequence[str],
    cat_terms: Sequence[str],
    cat_levels: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    out = df.copy()
    for c in ["pre_mean_ntl", "in_buffer", "delta_ntl", "is_damaged", "recovery_days", "event_observed"]:
        if c in out.columns:
            out[c] = _safe_numeric(out[c])

    for c in numeric_terms:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = _safe_numeric(out[c])
        out[c] = out.groupby("event_id", observed=True)[c].transform(lambda s: s.fillna(s.median()))
        out[c] = _safe_numeric(out[c])

    for c in cat_terms:
        if c not in out.columns:
            out[c] = "unknown"
        if isinstance(out[c].dtype, CategoricalDtype):
            if "unknown" not in out[c].cat.categories:
                out[c] = out[c].cat.add_categories(["unknown"])
        out[c] = out[c].fillna("unknown").astype(str)
        if cat_levels and c in cat_levels and len(cat_levels[c]) > 0:
            lv = list(cat_levels[c])
            if "unknown" not in lv:
                lv.append("unknown")
            out[c] = pd.Categorical(out[c], categories=lv)
    return out


def _build_formula(target: str, numeric_terms: Sequence[str], cat_terms: Sequence[str]) -> str:
    rhs = ["in_buffer * pre_mean_ntl"]
    rhs.extend([f"C({c})" for c in cat_terms])
    rhs.extend(list(numeric_terms))
    return f"{target} ~ " + " + ".join(rhs)


def _calibration_slope(y_true: np.ndarray, prob: np.ndarray) -> float:
    eps = 1e-6
    p = np.clip(prob, eps, 1 - eps)
    x = np.log(p / (1 - p))
    if np.allclose(np.std(x), 0) or len(np.unique(y_true)) < 2:
        return np.nan
    return float(np.polyfit(x, y_true, 1)[0])


def _select_penalty_cox(train_df: pd.DataFrame, penalties: Sequence[float]) -> float:
    if len(train_df) < 300:
        return 0.01
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    best_pen, best_score = 0.01, -np.inf
    for pen in penalties:
        scores = []
        for tr_idx, va_idx in kf.split(train_df):
            tr = train_df.iloc[tr_idx].copy()
            va = train_df.iloc[va_idx].copy()
            xva = va.drop(columns=["recovery_days", "event_observed"])
            try:
                cph = CoxPHFitter(penalizer=float(pen))
                cph.fit(tr, duration_col="recovery_days", event_col="event_observed")
                risk = cph.predict_partial_hazard(xva)
                cidx = float(concordance_index(va["recovery_days"], -risk.to_numpy().reshape(-1), va["event_observed"]))
                if np.isfinite(cidx):
                    scores.append(cidx)
            except Exception:
                continue
        if scores:
            s = float(np.mean(scores))
            if s > best_score:
                best_score, best_pen = s, float(pen)
    return best_pen


def _select_penalty_aft(train_df: pd.DataFrame, penalties: Sequence[float]) -> float:
    if len(train_df) < 300:
        return 0.01
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    best_pen, best_score = 0.01, -np.inf
    for pen in penalties:
        scores = []
        for tr_idx, va_idx in kf.split(train_df):
            tr = train_df.iloc[tr_idx].copy()
            va = train_df.iloc[va_idx].copy()
            xva = va.drop(columns=["recovery_days", "event_observed"])
            try:
                aft = WeibullAFTFitter(penalizer=float(pen))
                aft.fit(tr, duration_col="recovery_days", event_col="event_observed")
                med = aft.predict_median(xva)
                cidx = float(concordance_index(va["recovery_days"], -med.to_numpy().reshape(-1), va["event_observed"]))
                if np.isfinite(cidx):
                    scores.append(cidx)
            except Exception:
                continue
        if scores:
            s = float(np.mean(scores))
            if s > best_score:
                best_score, best_pen = s, float(pen)
    return best_pen


def _prepare_survival_design(
    rec_df: pd.DataFrame,
    numeric_terms: Sequence[str],
    cat_terms: Sequence[str],
) -> pd.DataFrame:
    cols = ["recovery_days", "event_observed", "event_id", "in_buffer", "pre_mean_ntl"] + list(numeric_terms) + list(cat_terms)
    cols = [c for c in cols if c in rec_df.columns]
    d = rec_df[cols].copy()

    for c in ["recovery_days", "event_observed", "in_buffer", "pre_mean_ntl"] + list(numeric_terms):
        if c in d.columns:
            d[c] = _safe_numeric(d[c])

    dummy_parts = []
    for c in cat_terms:
        if c in d.columns:
            dummy_parts.append(pd.get_dummies(d[c].fillna("unknown").astype(str), prefix=c, drop_first=True))
            d = d.drop(columns=[c])
    if dummy_parts:
        d = pd.concat([d.reset_index(drop=True)] + [x.reset_index(drop=True) for x in dummy_parts], axis=1)

    d = d.replace([np.inf, -np.inf], np.nan)
    for c in d.columns:
        if c in {"event_id"}:
            continue
        d[c] = _safe_numeric(d[c])
    return d


def _collect_coef_rows_statsmodels(
    result,
    round_id: str,
    fold_event: str,
    model: str,
) -> List[Dict[str, object]]:
    params = result.params
    pvals = getattr(result, "pvalues", pd.Series(np.nan, index=params.index))
    rows = []
    for term, coef in params.items():
        rows.append(
            {
                "round_id": round_id,
                "fold_event": fold_event,
                "model": model,
                "feature": str(term),
                "coef": float(coef),
                "p_value": float(pvals.get(term, np.nan)) if term in pvals.index else np.nan,
                "source": "interpretable_coef",
            }
        )
    return rows


def _collect_coef_rows_sklearn(
    model,
    feature_names: Sequence[str],
    round_id: str,
    fold_event: str,
    model_name: str,
) -> List[Dict[str, object]]:
    coefs = model.coef_.reshape(-1)
    rows = []
    for name, coef in zip(feature_names, coefs):
        rows.append(
            {
                "round_id": round_id,
                "fold_event": fold_event,
                "model": model_name,
                "feature": str(name),
                "coef": float(coef),
                "p_value": np.nan,
                "source": "interpretable_coef",
            }
        )
    return rows


def _run_loeo_round(
    round_id: str,
    panel: pd.DataFrame,
    recovery: pd.DataFrame,
    numeric_terms: Sequence[str],
    cat_terms: Sequence[str],
    damage_threshold: float,
    use_event_weights: bool,
    use_logit_calibration: bool,
    use_survival_grid: bool,
    cat_levels: Dict[str, List[str]],
    benchmark_selected_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = sorted(panel["event_id"].unique().tolist())
    fold_rows: List[Dict[str, object]] = []
    coef_rows: List[Dict[str, object]] = []
    perm_rows: List[Dict[str, object]] = []

    ols_formula = _build_formula("delta_ntl", numeric_terms, cat_terms)
    logit_formula = _build_formula("is_damaged", numeric_terms, cat_terms)

    for test_event in events:
        tr = panel[panel["event_id"] != test_event].copy()
        te = panel[panel["event_id"] == test_event].copy()
        tr["is_damaged"] = (tr["delta_ntl"] < damage_threshold).astype(int)
        te["is_damaged"] = (te["delta_ntl"] < damage_threshold).astype(int)

        tr = _fill_frame(tr, numeric_terms=numeric_terms, cat_terms=cat_terms, cat_levels=cat_levels)
        te = _fill_frame(te, numeric_terms=numeric_terms, cat_terms=cat_terms, cat_levels=cat_levels)

        w_tr = _event_inverse_sqrt_weights(tr["event_id"]).to_numpy() if use_event_weights else None

        # OLS
        try:
            if use_event_weights:
                ols = smf.wls(ols_formula, data=tr, weights=w_tr).fit(cov_type="HC1")
            else:
                ols = smf.ols(ols_formula, data=tr).fit(cov_type="HC1")
            pred = np.asarray(ols.predict(te))
            fold_rows.append(
                {
                    "round_id": round_id,
                    "fold_event": test_event,
                    "track": "interpretable",
                    "model": "OLS",
                    "rmse": float(math.sqrt(mean_squared_error(te["delta_ntl"], pred))),
                    "mae": float(mean_absolute_error(te["delta_ntl"], pred)),
                    "auc": np.nan,
                    "brier": np.nan,
                    "calibration_slope": np.nan,
                    "c_index": np.nan,
                    "coef_in_buffer": float(ols.params.get("in_buffer", np.nan)),
                    "notes": f"weights={int(use_event_weights)}",
                }
            )
            coef_rows.extend(_collect_coef_rows_statsmodels(ols, round_id, test_event, "OLS"))
        except Exception as e:
            raise StabilizationError(f"Round {round_id} OLS failed on fold {test_event}: {e}")

        # MixedLM
        try:
            mixed = None
            mixed_formula_used = ols_formula
            mixed_formulas = [ols_formula]
            # Stability fallback for folds with singular design (e.g., unseen category dominated fold).
            numeric_only_terms = [f"in_buffer * pre_mean_ntl"] + list(numeric_terms)
            mixed_formulas.append("delta_ntl ~ " + " + ".join(numeric_only_terms))
            mixed_formulas.append("delta_ntl ~ in_buffer * pre_mean_ntl")

            for fmx in mixed_formulas:
                for method in ["lbfgs", "powell", "cg", "nm"]:
                    try:
                        mixed = smf.mixedlm(fmx, data=tr, groups=tr["event_id"]).fit(method=method, reml=False)
                        mixed_formula_used = fmx
                        break
                    except Exception:
                        continue
                if mixed is not None:
                    break
            if mixed is None:
                raise RuntimeError("mixedlm fit failed for all optimizers")
            pred = np.asarray(mixed.predict(te))
            fold_rows.append(
                {
                    "round_id": round_id,
                    "fold_event": test_event,
                    "track": "interpretable",
                    "model": "MixedLM",
                    "rmse": float(math.sqrt(mean_squared_error(te["delta_ntl"], pred))),
                    "mae": float(mean_absolute_error(te["delta_ntl"], pred)),
                    "auc": np.nan,
                    "brier": np.nan,
                    "calibration_slope": np.nan,
                    "c_index": np.nan,
                    "coef_in_buffer": float(mixed.params.get("in_buffer", np.nan)),
                    "notes": f"mixedlm_formula={mixed_formula_used}",
                }
            )
            coef_rows.extend(_collect_coef_rows_statsmodels(mixed, round_id, test_event, "MixedLM"))
        except Exception as e:
            raise StabilizationError(f"Round {round_id} MixedLM failed on fold {test_event}: {e}")

        # Logit (L2)
        try:
            y_tr, X_tr = dmatrices(logit_formula, tr, return_type="dataframe")
            X_te = build_design_matrices([X_tr.design_info], te, return_type="dataframe")[0]
            y_tr_arr = y_tr.values.reshape(-1).astype(int)
            y_te_arr = te["is_damaged"].to_numpy().astype(int)

            base_clf = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=1200,
                fit_intercept=False,
                random_state=42,
            )
            base_clf.fit(X_tr, y_tr_arr, sample_weight=w_tr)

            if use_logit_calibration:
                cal = None
                for method in ["sigmoid", "isotonic"]:
                    try:
                        cal = CalibratedClassifierCV(
                            estimator=LogisticRegression(
                                penalty="l2",
                                C=1.0,
                                solver="lbfgs",
                                max_iter=1200,
                                fit_intercept=False,
                                random_state=42,
                            ),
                            method=method,
                            cv=3,
                        )
                        cal.fit(X_tr, y_tr_arr, sample_weight=w_tr)
                        prob = cal.predict_proba(X_te)[:, 1]
                        cal_used = method
                        break
                    except Exception:
                        cal = None
                if cal is None:
                    prob = base_clf.predict_proba(X_te)[:, 1]
                    cal_used = "none"
            else:
                prob = base_clf.predict_proba(X_te)[:, 1]
                cal_used = "none"

            auc = float(roc_auc_score(y_te_arr, prob)) if len(np.unique(y_te_arr)) > 1 else np.nan
            brier = float(brier_score_loss(y_te_arr, prob))
            slope = _calibration_slope(y_te_arr, prob)

            fold_rows.append(
                {
                    "round_id": round_id,
                    "fold_event": test_event,
                    "track": "interpretable",
                    "model": "Logit",
                    "rmse": np.nan,
                    "mae": np.nan,
                    "auc": auc,
                    "brier": brier,
                    "calibration_slope": slope,
                    "c_index": np.nan,
                    "coef_in_buffer": float(dict(zip(X_tr.columns, base_clf.coef_.reshape(-1))).get("in_buffer", np.nan)),
                    "notes": f"l2;cal={cal_used};weights={int(use_event_weights)}",
                }
            )
            coef_rows.extend(_collect_coef_rows_sklearn(base_clf, X_tr.columns, round_id, test_event, "Logit"))
        except Exception as e:
            raise StabilizationError(f"Round {round_id} Logit failed on fold {test_event}: {e}")

        # Survival (Cox + AFT, pick best later)
        try:
            tr_rec = recovery[recovery["event_id"] != test_event].copy()
            te_rec = recovery[recovery["event_id"] == test_event].copy()
            tr_rec = _fill_frame(tr_rec, numeric_terms=numeric_terms, cat_terms=cat_terms, cat_levels=cat_levels)
            te_rec = _fill_frame(te_rec, numeric_terms=numeric_terms, cat_terms=cat_terms, cat_levels=cat_levels)

            dtr = _prepare_survival_design(tr_rec, numeric_terms=numeric_terms, cat_terms=cat_terms)
            dte = _prepare_survival_design(te_rec, numeric_terms=numeric_terms, cat_terms=cat_terms)
            xtr = dtr.drop(columns=["recovery_days", "event_observed", "event_id"], errors="ignore")
            xte = dte.drop(columns=["recovery_days", "event_observed", "event_id"], errors="ignore")
            xte = xte.reindex(columns=xtr.columns, fill_value=0.0)

            fit_tr = pd.concat([dtr[["recovery_days", "event_observed"]].reset_index(drop=True), xtr.reset_index(drop=True)], axis=1)
            fit_te = pd.concat([dte[["recovery_days", "event_observed"]].reset_index(drop=True), xte.reset_index(drop=True)], axis=1)

            penalties = [0.001, 0.01, 0.05, 0.1]
            pen_cox = _select_penalty_cox(fit_tr, penalties) if use_survival_grid else 0.01
            pen_aft = _select_penalty_aft(fit_tr, penalties) if use_survival_grid else 0.01

            cph = CoxPHFitter(penalizer=float(pen_cox))
            cph.fit(fit_tr, duration_col="recovery_days", event_col="event_observed")
            risk = cph.predict_partial_hazard(fit_te.drop(columns=["recovery_days", "event_observed"]))
            cidx_cox = float(
                concordance_index(
                    fit_te["recovery_days"],
                    -risk.to_numpy().reshape(-1),
                    fit_te["event_observed"],
                )
            )

            aft = WeibullAFTFitter(penalizer=float(pen_aft))
            aft.fit(fit_tr, duration_col="recovery_days", event_col="event_observed")
            med = aft.predict_median(fit_te.drop(columns=["recovery_days", "event_observed"]))
            cidx_aft = float(
                concordance_index(
                    fit_te["recovery_days"],
                    -med.to_numpy().reshape(-1),
                    fit_te["event_observed"],
                )
            )

            fold_rows.append(
                {
                    "round_id": round_id,
                    "fold_event": test_event,
                    "track": "interpretable",
                    "model": "Cox",
                    "rmse": np.nan,
                    "mae": np.nan,
                    "auc": np.nan,
                    "brier": np.nan,
                    "calibration_slope": np.nan,
                    "c_index": cidx_cox,
                    "coef_in_buffer": float(cph.params_.get("in_buffer", np.nan)),
                    "notes": f"pen={pen_cox:.3g}",
                }
            )
            fold_rows.append(
                {
                    "round_id": round_id,
                    "fold_event": test_event,
                    "track": "interpretable",
                    "model": "AFT",
                    "rmse": np.nan,
                    "mae": np.nan,
                    "auc": np.nan,
                    "brier": np.nan,
                    "calibration_slope": np.nan,
                    "c_index": cidx_aft,
                    "coef_in_buffer": np.nan,
                    "notes": f"pen={pen_aft:.3g}",
                }
            )

            for feat, coef in cph.params_.items():
                pval = np.nan
                try:
                    if feat in cph.summary.index:
                        pval = float(cph.summary.loc[feat, "p"])
                except Exception:
                    pval = np.nan
                coef_rows.append(
                    {
                        "round_id": round_id,
                        "fold_event": test_event,
                        "model": "Cox",
                        "feature": str(feat),
                        "coef": float(coef),
                        "p_value": pval,
                        "source": "interpretable_coef",
                    }
                )
            try:
                aft_params = aft.params_.copy()
                for idx, val in aft_params.items():
                    coef_rows.append(
                        {
                            "round_id": round_id,
                            "fold_event": test_event,
                            "model": "AFT",
                            "feature": str(idx),
                            "coef": float(val),
                            "p_value": np.nan,
                            "source": "interpretable_coef",
                        }
                    )
            except Exception:
                pass
        except Exception as e:
            raise StabilizationError(f"Round {round_id} Survival failed on fold {test_event}: {e}")

        # Benchmark models
        try:
            xtr_b = pd.DataFrame({"in_buffer": _safe_numeric(tr["in_buffer"]), "pre_mean_ntl": _safe_numeric(tr["pre_mean_ntl"])})
            xte_b = pd.DataFrame({"in_buffer": _safe_numeric(te["in_buffer"]), "pre_mean_ntl": _safe_numeric(te["pre_mean_ntl"])})
            for c in numeric_terms:
                xtr_b[c] = _safe_numeric(tr[c])
                xte_b[c] = _safe_numeric(te[c])
            if cat_terms:
                tr_cat = pd.get_dummies(tr[list(cat_terms)].fillna("unknown").astype(str), drop_first=False)
                te_cat = pd.get_dummies(te[list(cat_terms)].fillna("unknown").astype(str), drop_first=False)
                te_cat = te_cat.reindex(columns=tr_cat.columns, fill_value=0)
                xtr_b = pd.concat([xtr_b.reset_index(drop=True), tr_cat.reset_index(drop=True)], axis=1)
                xte_b = pd.concat([xte_b.reset_index(drop=True), te_cat.reset_index(drop=True)], axis=1)

            if benchmark_selected_cols:
                use_cols = [c for c in benchmark_selected_cols if c in xtr_b.columns]
                if not use_cols:
                    use_cols = list(xtr_b.columns)
                xtr_b = xtr_b[use_cols].copy()
                xte_b = xte_b[use_cols].copy()

            ytr_reg = _safe_numeric(tr["delta_ntl"]).to_numpy()
            yte_reg = _safe_numeric(te["delta_ntl"]).to_numpy()
            ytr_cls = tr["is_damaged"].astype(int).to_numpy()
            yte_cls = te["is_damaged"].astype(int).to_numpy()

            hgb_r = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=250, random_state=42)
            hgb_r.fit(xtr_b, ytr_reg, sample_weight=w_tr)
            pred_r = hgb_r.predict(xte_b)
            fold_rows.append(
                {
                    "round_id": round_id,
                    "fold_event": test_event,
                    "track": "benchmark",
                    "model": "HGBRegressor",
                    "rmse": float(math.sqrt(mean_squared_error(yte_reg, pred_r))),
                    "mae": float(mean_absolute_error(yte_reg, pred_r)),
                    "auc": np.nan,
                    "brier": np.nan,
                    "calibration_slope": np.nan,
                    "c_index": np.nan,
                    "coef_in_buffer": np.nan,
                    "notes": f"weights={int(use_event_weights)}",
                }
            )

            hgb_c = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=250, random_state=42)
            hgb_c.fit(xtr_b, ytr_cls, sample_weight=w_tr)
            prob_c = hgb_c.predict_proba(xte_b)[:, 1]
            auc_c = float(roc_auc_score(yte_cls, prob_c)) if len(np.unique(yte_cls)) > 1 else np.nan
            brier_c = float(brier_score_loss(yte_cls, prob_c))
            fold_rows.append(
                {
                    "round_id": round_id,
                    "fold_event": test_event,
                    "track": "benchmark",
                    "model": "HGBClassifier",
                    "rmse": np.nan,
                    "mae": np.nan,
                    "auc": auc_c,
                    "brier": brier_c,
                    "calibration_slope": np.nan,
                    "c_index": np.nan,
                    "coef_in_buffer": np.nan,
                    "notes": f"weights={int(use_event_weights)}",
                }
            )

            if len(np.unique(yte_cls)) > 1:
                pi = permutation_importance(
                    hgb_c,
                    xte_b,
                    yte_cls,
                    n_repeats=5,
                    random_state=42,
                    scoring="roc_auc",
                )
                for name, m, s in zip(xte_b.columns, pi.importances_mean, pi.importances_std):
                    perm_rows.append(
                        {
                            "round_id": round_id,
                            "fold_event": test_event,
                            "model": "HGBClassifier",
                            "feature": str(name),
                            "source": "permutation",
                            "importance_mean": float(m),
                            "importance_std": float(s),
                        }
                    )
        except Exception as e:
            raise StabilizationError(f"Round {round_id} benchmark failed on fold {test_event}: {e}")

    fold_df = pd.DataFrame(fold_rows)
    for model in ["OLS", "MixedLM", "Logit"]:
        m = fold_df["model"] == model
        if m.any():
            ref = np.sign(np.nanmean(fold_df.loc[m, "coef_in_buffer"]))
            fold_df.loc[m, "sign_consistency"] = np.where(
                np.sign(fold_df.loc[m, "coef_in_buffer"]) == ref,
                1.0,
                0.0,
            )

    agg_cols = ["rmse", "mae", "auc", "brier", "calibration_slope", "c_index", "sign_consistency"]
    agg_df = fold_df.groupby(["round_id", "track", "model"], dropna=False)[agg_cols].mean().reset_index()

    coef_df = pd.DataFrame(coef_rows)
    if perm_rows:
        perm_df = pd.DataFrame(perm_rows)
        perm_agg = perm_df.groupby(["round_id", "model", "feature", "source"], dropna=False).agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_std", "mean"),
        ).reset_index()
    else:
        perm_agg = pd.DataFrame(columns=["round_id", "model", "feature", "source", "importance_mean", "importance_std"])

    if coef_df.empty:
        coef_part = pd.DataFrame(columns=["round_id", "model", "feature", "source", "coef", "p_value"])
    else:
        coef_part = coef_df.groupby(["round_id", "model", "feature", "source"], dropna=False).agg(
            coef=("coef", "mean"),
            p_value=("p_value", "mean"),
        ).reset_index()

    fi_df = coef_part.merge(perm_agg, on=["round_id", "model", "feature", "source"], how="outer")
    return fold_df, agg_df, fi_df


def _extract_round_metrics(agg_df: pd.DataFrame) -> Dict[str, float]:
    def pick(track: str, model: str, col: str) -> float:
        s = agg_df[(agg_df["track"] == track) & (agg_df["model"] == model)]
        if s.empty:
            return np.nan
        v = pd.to_numeric(pd.Series([s.iloc[0][col]]), errors="coerce").iloc[0]
        return float(v) if np.isfinite(v) else np.nan

    cox = pick("interpretable", "Cox", "c_index")
    aft = pick("interpretable", "AFT", "c_index")
    surv_best = np.nanmax([cox, aft]) if (np.isfinite(cox) or np.isfinite(aft)) else np.nan

    out = {
        "logit_auc": pick("interpretable", "Logit", "auc"),
        "logit_brier": pick("interpretable", "Logit", "brier"),
        "survival_best_c_index": float(surv_best) if np.isfinite(surv_best) else np.nan,
        "cox_c_index": cox,
        "aft_c_index": aft,
        "ols_rmse": pick("interpretable", "OLS", "rmse"),
        "mixedlm_rmse": pick("interpretable", "MixedLM", "rmse"),
        "hgb_auc": pick("benchmark", "HGBClassifier", "auc"),
        "hgb_rmse": pick("benchmark", "HGBRegressor", "rmse"),
    }
    return out


def _round_comparison_rows(
    round_id: str,
    cur: Dict[str, float],
    prev: Dict[str, float],
) -> Tuple[List[Dict[str, object]], bool]:
    rows = []

    d_auc = cur["logit_auc"] - prev["logit_auc"] if np.isfinite(cur["logit_auc"]) and np.isfinite(prev["logit_auc"]) else np.nan
    d_surv = (
        cur["survival_best_c_index"] - prev["survival_best_c_index"]
        if np.isfinite(cur["survival_best_c_index"]) and np.isfinite(prev["survival_best_c_index"])
        else np.nan
    )

    stop_flag = bool(np.isfinite(d_auc) and np.isfinite(d_surv) and (d_auc < THRESH_AUC) and (d_surv < THRESH_SURV))

    metric_defs = [
        ("classification", "logit_auc", THRESH_AUC),
        ("survival", "survival_best_c_index", THRESH_SURV),
        ("classification", "logit_brier", np.nan),
        ("regression", "ols_rmse", np.nan),
        ("regression", "mixedlm_rmse", np.nan),
        ("benchmark", "hgb_auc", np.nan),
    ]

    for group, name, thr in metric_defs:
        v = cur.get(name, np.nan)
        pv = prev.get(name, np.nan)
        dv = (v - pv) if np.isfinite(v) and np.isfinite(pv) else np.nan
        if np.isfinite(thr):
            pass_flag = int(np.isfinite(dv) and dv >= thr)
            notes = f"target_delta>={thr:.3f}"
        else:
            pass_flag = np.nan
            notes = "tracking"
        rows.append(
            {
                "round_id": round_id,
                "metric_group": group,
                "metric_name": name,
                "value": v,
                "prev_value": pv,
                "delta": dv,
                "threshold": thr,
                "pass_flag": pass_flag,
                "stop_flag": int(stop_flag),
                "notes": notes,
            }
        )
    return rows, stop_flag


def _build_round_summary(round_id: str, metrics: Dict[str, float], prev_metrics: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for metric in ["logit_auc", "logit_brier", "survival_best_c_index", "cox_c_index", "aft_c_index", "ols_rmse", "mixedlm_rmse", "hgb_auc", "hgb_rmse"]:
        v = metrics.get(metric, np.nan)
        pv = prev_metrics.get(metric, np.nan)
        dv = (v - pv) if np.isfinite(v) and np.isfinite(pv) else np.nan
        rows.append(
            {
                "round_id": round_id,
                "metric_name": metric,
                "value": v,
                "prev_value": pv,
                "delta_vs_prev": dv,
            }
        )
    return pd.DataFrame(rows)


def _pick_round3_numeric(r2_coef_df: pd.DataFrame, base_terms: Sequence[str]) -> List[str]:
    if r2_coef_df.empty:
        return list(base_terms)
    stable_terms: List[str] = []
    for t in base_terms:
        sub = r2_coef_df[
            (r2_coef_df["model"].isin(["OLS", "MixedLM", "Logit"]))
            & (r2_coef_df["feature"] == t)
            & pd.to_numeric(r2_coef_df["coef"], errors="coerce").notna()
        ]
        if sub.empty:
            continue
        signs = np.sign(pd.to_numeric(sub["coef"], errors="coerce").to_numpy())
        pos = int((signs > 0).sum())
        neg = int((signs < 0).sum())
        if max(pos, neg) >= 4:
            stable_terms.append(t)
    if not stable_terms:
        return list(base_terms)
    return stable_terms


def _pick_round3_benchmark_cols(r2_fi_df: pd.DataFrame, top_k: int = 10) -> Optional[List[str]]:
    if r2_fi_df.empty:
        return None
    sub = r2_fi_df[(r2_fi_df["source"] == "permutation") & (r2_fi_df["model"] == "HGBClassifier")].copy()
    if sub.empty:
        return None
    sub = sub.sort_values("importance_mean", ascending=False).head(top_k)
    cols = [str(x) for x in sub["feature"].tolist() if pd.notna(x)]
    return cols or None


def _update_index() -> None:
    line = "- `project/modeling_report/09_cross_event_stabilization_report.md`"
    note = "- V3 stabilization outputs: `project/modeling/output/cross_event_round_comparison_v3x.csv`, `project/modeling/output/cross_event_stop_decision_v3x.json`"

    if INDEX_PATH.exists():
        text = INDEX_PATH.read_text(encoding="utf-8")
    else:
        text = "# Modeling Report Index\n\n## Deliverables\n"

    if line not in text:
        text = text.rstrip() + "\n" + line + "\n"
    if note not in text:
        text = text.rstrip() + "\n" + note + "\n"
    INDEX_PATH.write_text(text, encoding="utf-8")


def _write_report(
    round_rows: pd.DataFrame,
    stop_decision: Dict[str, object],
    round_metrics_map: Dict[str, Dict[str, float]],
) -> None:
    def fmt(v) -> str:
        return f"{float(v):.4f}" if pd.notna(v) and np.isfinite(float(v)) else "N/A"

    lines = [
        "# Cross-Event Stabilization Report / 跨事件稳定化修正报告（V3.x）",
        "",
        "## Objective",
        "在最多 3 轮内验证稳定化修正是否能把 V3 从边际改进推进到可接受改进；若仍边际改进则按规则收工。",
        "",
        "## Stop Rule",
        f"- Balanced: `ΔLogit AUC < {THRESH_AUC:.2f}` 且 `ΔSurvival(best c-index) < {THRESH_SURV:.2f}` -> 停止。",
        f"- 最终停止轮次: `{stop_decision.get('stopped_at_round')}`",
        f"- 停止原因: `{stop_decision.get('reason')}`",
        "",
        "## Round Metrics",
        "| round | logit_auc | logit_brier | survival_best_c_index | cox_c_index | aft_c_index | ols_rmse | mixedlm_rmse | hgb_auc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for rid in sorted(round_metrics_map.keys(), key=lambda x: int(x.replace("r", "")) if x.startswith("r") else -1):
        m = round_metrics_map[rid]
        lines.append(
            f"| {rid} | {fmt(m.get('logit_auc'))} | {fmt(m.get('logit_brier'))} | {fmt(m.get('survival_best_c_index'))} | {fmt(m.get('cox_c_index'))} | {fmt(m.get('aft_c_index'))} | {fmt(m.get('ols_rmse'))} | {fmt(m.get('mixedlm_rmse'))} | {fmt(m.get('hgb_auc'))} |"
        )

    lines.extend(
        [
            "",
            "## Stop Decision Evidence",
            f"- Comparison file: `{ROUND_COMP_PATH.relative_to(ROOT)}`",
            f"- Decision file: `{STOP_DECISION_PATH.relative_to(ROOT)}`",
            "",
            "## Output Files",
            "- Round fold metrics: `project/modeling/output/cross_event_fold_metrics_v3r*.csv`",
            "- Round aggregates: `project/modeling/output/cross_event_aggregate_metrics_v3r*.csv`",
            "- Round feature importance: `project/modeling/output/cross_event_feature_importance_v3r*.csv`",
            "- Round summaries: `project/modeling/output/model_summary_cross_event_v3r*.csv`",
            "",
            "## Conclusion",
        ]
    )

    if bool(stop_decision.get("marginal_improvement_stop", False)):
        lines.append("- 触发边际改进停止规则：今天按计划收工。")
    else:
        lines.append("- 未触发边际改进停止，但已达到轮次预算上限：今天按计划收工。")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_directories()
    init_tracking_files()

    if not PANEL_V3_PATH.exists():
        raise FileNotFoundError(f"Missing panel: {PANEL_V3_PATH}")
    if not BASE_AGG_PATH.exists():
        raise FileNotFoundError(f"Missing V3 aggregate anchor: {BASE_AGG_PATH}")

    defaults = load_json(CONFIG_DEFAULTS)
    damage_thr = float(defaults["damage_threshold"])
    rec_thr = float(defaults["recovery_threshold"])

    panel = pd.read_parquet(PANEL_V3_PATH).copy()
    if "sample_lock_flag" in panel.columns:
        panel = panel[panel["sample_lock_flag"] == 1].copy()
    if panel.empty:
        raise RuntimeError("No rows available after sample lock filter.")

    # Base event-level constant features are intentionally excluded from interpretable numeric list.
    required_cols = ["event_id", "delta_ntl", "pre_mean_ntl", "in_buffer", "land_use_group", "event_disaster_type"] + LOCAL_NUMERIC_BASE
    missing = [c for c in required_cols if c not in panel.columns]
    if missing:
        raise KeyError(f"Missing required columns in V3 panel: {missing}")

    cat_levels = {
        c: sorted(panel[c].fillna("unknown").astype(str).unique().tolist())
        for c in CAT_TERMS_BASE
        if c in panel.columns
    }
    panel = _fill_frame(panel, numeric_terms=LOCAL_NUMERIC_BASE, cat_terms=CAT_TERMS_BASE, cat_levels=cat_levels)
    rec_ctx = RunContext(issues=[])
    recovery = build_recovery_panel(ctx=rec_ctx, panel=panel, threshold=rec_thr, output_path=None)
    recovery = _fill_frame(recovery, numeric_terms=LOCAL_NUMERIC_BASE, cat_terms=CAT_TERMS_BASE, cat_levels=cat_levels)

    # Round0 anchor from existing V3 aggregate.
    agg0 = pd.read_csv(BASE_AGG_PATH)
    base_metrics = {
        "logit_auc": float(agg0[(agg0["track"] == "interpretable") & (agg0["model"] == "Logit")]["auc"].iloc[0]),
        "logit_brier": float(agg0[(agg0["track"] == "interpretable") & (agg0["model"] == "Logit")]["brier"].iloc[0]),
        "cox_c_index": float(agg0[(agg0["track"] == "interpretable") & (agg0["model"] == "Cox")]["c_index"].iloc[0]),
        "aft_c_index": float(agg0[(agg0["track"] == "interpretable") & (agg0["model"] == "AFT")]["c_index"].iloc[0]),
        "ols_rmse": float(agg0[(agg0["track"] == "interpretable") & (agg0["model"] == "OLS")]["rmse"].iloc[0]),
        "mixedlm_rmse": float(agg0[(agg0["track"] == "interpretable") & (agg0["model"] == "MixedLM")]["rmse"].iloc[0]),
        "hgb_auc": float(agg0[(agg0["track"] == "benchmark") & (agg0["model"] == "HGBClassifier")]["auc"].iloc[0]),
        "hgb_rmse": float(agg0[(agg0["track"] == "benchmark") & (agg0["model"] == "HGBRegressor")]["rmse"].iloc[0]),
    }
    base_metrics["survival_best_c_index"] = float(max(base_metrics["cox_c_index"], base_metrics["aft_c_index"]))

    round_metrics_map: Dict[str, Dict[str, float]] = {"r0": base_metrics}
    comp_rows: List[Dict[str, object]] = []

    stopped = False
    stop_reason = ""
    stopped_round = "r0"
    prev_metrics = base_metrics

    r2_coef_store = pd.DataFrame()
    r2_fi_store = pd.DataFrame()

    for rid in range(1, MAX_ROUNDS + 1):
        round_name = f"r{rid}"

        if rid == 1:
            numeric_terms = list(LOCAL_NUMERIC_BASE)
            benchmark_cols = None
            use_weights = False
            use_cal = False
            use_surv_grid = False
        elif rid == 2:
            numeric_terms = list(LOCAL_NUMERIC_BASE)
            benchmark_cols = None
            use_weights = True
            use_cal = True
            use_surv_grid = True
        else:
            numeric_terms = _pick_round3_numeric(r2_coef_store, LOCAL_NUMERIC_BASE)
            benchmark_cols = _pick_round3_benchmark_cols(r2_fi_store, top_k=10)
            use_weights = True
            use_cal = True
            use_surv_grid = True

        fold_df, agg_df, fi_df = _run_loeo_round(
            round_id=round_name,
            panel=panel,
            recovery=recovery,
            numeric_terms=numeric_terms,
            cat_terms=CAT_TERMS_BASE,
            damage_threshold=damage_thr,
            use_event_weights=use_weights,
            use_logit_calibration=use_cal,
            use_survival_grid=use_surv_grid,
            cat_levels=cat_levels,
            benchmark_selected_cols=benchmark_cols,
        )

        # Save round outputs.
        fold_path = OUTPUT_DIR / f"cross_event_fold_metrics_v3{round_name}.csv"
        agg_path = OUTPUT_DIR / f"cross_event_aggregate_metrics_v3{round_name}.csv"
        fi_path = OUTPUT_DIR / f"cross_event_feature_importance_v3{round_name}.csv"
        summary_path = OUTPUT_DIR / f"model_summary_cross_event_v3{round_name}.csv"

        fold_df.to_csv(fold_path, index=False)
        agg_df.to_csv(agg_path, index=False)
        fi_df.to_csv(fi_path, index=False)

        cur_metrics = _extract_round_metrics(agg_df)
        # Hard stability guard: if mandatory metrics are NaN, stop immediately.
        mandatory = [cur_metrics.get("logit_auc", np.nan), cur_metrics.get("survival_best_c_index", np.nan)]
        if any([not np.isfinite(v) for v in mandatory]):
            raise StabilizationError(f"Round {round_name} produced invalid mandatory metrics: {mandatory}")

        summary_df = _build_round_summary(round_name, cur_metrics, prev_metrics)
        summary_df.to_csv(summary_path, index=False)

        round_metrics_map[round_name] = cur_metrics
        rows, stop_flag = _round_comparison_rows(round_name, cur_metrics, prev_metrics)
        comp_rows.extend(rows)

        if rid == 2:
            r2_coef_store = fi_df[fi_df["source"] == "interpretable_coef"].copy()
            r2_fi_store = fi_df.copy()

        if stop_flag:
            stopped = True
            stop_reason = "marginal_improvement"
            stopped_round = round_name
            break

        prev_metrics = cur_metrics
        stopped_round = round_name

    if not stopped:
        stop_reason = "max_round_reached"

    comp_df = pd.DataFrame(comp_rows)
    comp_df = comp_df[
        [
            "round_id",
            "metric_group",
            "metric_name",
            "value",
            "prev_value",
            "delta",
            "threshold",
            "pass_flag",
            "stop_flag",
            "notes",
        ]
    ]
    comp_df.to_csv(ROUND_COMP_PATH, index=False)

    final_metrics = round_metrics_map.get(stopped_round, {})
    stop_json = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Balanced",
        "thresholds": {"delta_logit_auc": THRESH_AUC, "delta_survival_best_c_index": THRESH_SURV},
        "max_rounds": MAX_ROUNDS,
        "stopped_at_round": stopped_round,
        "reason": stop_reason,
        "marginal_improvement_stop": bool(stop_reason == "marginal_improvement"),
        "baseline_round": "r0",
        "baseline_metrics": round_metrics_map.get("r0", {}),
        "final_metrics": final_metrics,
        "executed_rounds": sorted([k for k in round_metrics_map.keys() if k.startswith("r") and k != "r0"], key=lambda x: int(x[1:])),
    }
    STOP_DECISION_PATH.write_text(json.dumps(stop_json, indent=2), encoding="utf-8")

    _write_report(round_rows=comp_df, stop_decision=stop_json, round_metrics_map=round_metrics_map)
    _update_index()


if __name__ == "__main__":
    main()
