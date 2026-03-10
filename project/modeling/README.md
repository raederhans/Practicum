# Modeling README

## Scope

This directory now reflects the post-fix modeling pipeline as of 2026-03-09. The current workflow is organized around:

1. A strict in-sample / LOEO mainline with fold-local preprocessing only.
2. A cross-event transport pipeline that is being rerun under the same no-leakage rules.
3. Exploration and sensitivity bundles that stay separate from headline claims.
4. Appendix-only post-hoc analyses for quality adjustment and hazard/exposure transport.

Historical code and pre-fix docs are archived under `project/modeling/legacy/archive_model_fix_20260309/`.

## Entry Points

- `project/modeling/run_pipeline.py`
  - Compatibility entrypoint. Now delegates to the real pipeline sequence in `01_in_sample_pipeline.py`.
- `project/modeling/pipelines/01_in_sample_pipeline.py`
  - Baseline build, NLCD attach, feature-upgrade, strict-v2, reports, figures.
- `project/modeling/pipelines/02_cross_event_pipeline.py`
  - Cross-event V3 build and stabilization.
- `project/modeling/pipelines/03_exploration_pipeline.py`
  - Exploration V2, appendix bundles, and event-expansion utilities.

## Environment

- Working interpreter for this rerun: `.venv_modeling`
- Frozen package snapshot: `project/modeling/requirements-modeling.txt`
- Core libraries confirmed during rerun: `pandas`, `pyarrow`, `rasterio`, `statsmodels`, `lifelines`, `scikit-learn`

## Data Sources And Gates

The current mainline uses the existing pixel panel plus attached land-use / cloud features, but the model-fitting rules are tighter than before:

- `sample_lock_flag` is treated as a hard cohort gate.
- Cloud-quality flags are enforced before mainline fitting instead of being left as a purely downstream covariate.
- Recovery panels are rebuilt from the current panel and threshold config, not backfilled from stale cached recovery artifacts.
- Missing target columns fail fast in exploration bundles instead of being silently zero-filled.
- Fold-level preprocessing is now train-only:
  - fill rules are fit on the training fold
  - scaling is fit on the training fold
  - held-out events only receive those learned rules
- Mainline commands no longer fall back to stale cached artifacts after build failures.

## Mainline Flow

1. `build-baseline`
   - Builds the baseline panel and sample lock cohort.
2. `attach-nlcd`
   - Adds land-use context.
3. `feature-upgrade`
   - Produces refreshed in-sample and LOEO outputs for the feature-upgrade branch.
4. `strict-v2`
   - Produces the strict ex-ante locked-panel results and transport diagnostics.
5. `build-v3` / `stabilize-v3`
   - Produces cross-event transport artifacts under the same fold-local preprocessing discipline.
6. `run-v2`
   - Runs the exploration mainline bundles only.
7. `quality-matched-v1` and `hazard-mainline-v1`
   - Run appendix-only analyses. These do not contribute to headline claims.

## Rerun Commands

Use the modeling environment first:

```bash
source .venv_modeling/bin/activate
python project/modeling/pipelines/01_in_sample_pipeline.py build-baseline
python project/modeling/pipelines/01_in_sample_pipeline.py attach-nlcd
python project/modeling/pipelines/01_in_sample_pipeline.py feature-upgrade
python project/modeling/pipelines/01_in_sample_pipeline.py strict-v2
python project/modeling/pipelines/02_cross_event_pipeline.py build-v3
python project/modeling/pipelines/02_cross_event_pipeline.py stabilize-v3
python project/modeling/pipelines/03_exploration_pipeline.py run-v2
python project/modeling/pipelines/03_exploration_pipeline.py quality-matched-v1
python project/modeling/pipelines/03_exploration_pipeline.py hazard-mainline-v1
```

Independent exploration bundle commands now execute only their own bundle:

```bash
python project/modeling/pipelines/03_exploration_pipeline.py cloud-ablation
python project/modeling/pipelines/03_exploration_pipeline.py noise-mask
python project/modeling/pipelines/03_exploration_pipeline.py urban-rural
python project/modeling/pipelines/03_exploration_pipeline.py spatial-diagnostics
python project/modeling/pipelines/03_exploration_pipeline.py extreme-event-sensitivity
```

## Artifact Map

Mainline:

- `project/modeling/output/model_summary_feature_upgrade_v2_strict.csv`
- `project/modeling/output/logo_aggregate_metrics_v2_strict.csv`
- `project/modeling/output/multicollinearity_vif_v2_strict.csv`
- `project/modeling/output/model_summary_cross_event_v3.csv`
- `project/modeling/output/cross_event_aggregate_metrics_v3.csv`
- `project/modeling/output/cross_event_round_comparison_v3x.csv`
- `project/modeling/output/cross_event_stop_decision_v3x.json`

Exploration mainline:

- `project/modeling/output/cloud_ablation_aggregate_metrics.csv`
- `project/modeling/output/cloud_ablation_coefficients.csv`
- `project/modeling/output/noise_mask_experiment_metrics.csv`
- `project/modeling/output/noise_mask_coefficients.csv`
- `project/modeling/output/urban_rural_model_comparison.csv`
- `project/modeling/output/urban_rural_coefficients.csv`
- `project/modeling/output/spatial_autocorr_morans_i.csv`
- `project/modeling/output/extreme_event_drop_aggregate_v1.csv`

Appendix-only:

- `project/modeling/output/quality_transport_aggregate_metrics_v1.csv`
- `project/modeling/output/spatial_block_cv_metrics_v1.csv`
- `project/modeling/output/facility_centered_model_summary.csv`
- `project/modeling/output/hazard_transport_aggregate_metrics_v1.csv`
- `project/modeling/output/hazard_transport_feature_summary_v1.csv`
- `project/modeling/output/event_selection_scorecard_v1.csv`

## Headline Results

### Strict V2 Mainline

On the full locked panel (`n = 10,306`), the strict full specification currently reports:

- OLS `coef_in_buffer = 0.0269`, `p = 0.0544`
- MixedLM `coef_in_buffer = 0.0269`, `p = 0.00943`
- Logit `odds_ratio_in_buffer = 0.7503`, `AUC = 0.7299`, `p = 1.83e-06`
- Cox `hazard_ratio_in_buffer = 1.3319`, `p = 6.15e-15`

Strict LOEO transport remains materially harder than the in-sample fits:

- OLS `RMSE = 0.4132`, `MAE = 0.2622`
- MixedLM `RMSE = 0.4512`, `MAE = 0.3237`
- Logit `AUC = 0.4549`, `Brier = 0.2542`, `calibration_slope = -0.0305`
- Cox `c_index = 0.5200`

Interpretation:

- The in-sample sign pattern remains detectable under the stricter preprocessing rules.
- The out-of-event transport signal is weak and unstable, especially for the damage-classification branch.
- Headline claims should therefore rely on the strict mainline plus cross-event transport, not on the appendix bundles.

### Cross-Event Mainline

The corrected V3 rerun now uses the same fold-local preprocessing discipline as the strict mainline.

V3 transport (`build-v3`):

- AFT `c_index = 0.5354`
- Cox `c_index = 0.4641`
- Logit `AUC = 0.4897`, `Brier = 0.3110`
- OLS `RMSE = 0.5243`, `MAE = 0.3820`
- MixedLM `RMSE = 0.6010`, `MAE = 0.4842`

Coefficient snapshot from `model_summary_cross_event_v3.csv`:

- Logit `coef_in_buffer = -0.2327`, `p = 0.00233`
- Cox `coef_in_buffer = 0.0900`, `p = 0.0978`
- MixedLM `coef_in_buffer = 0.0468`, `p = 0.0881`
- OLS `coef_in_buffer = 0.0462`, `p = 0.1286`

Stabilization (`stabilize-v3`) stopped after round `r1` with `reason = marginal_improvement`:

- final Logit `AUC = 0.4814`
- final Logit `Brier = 0.2567`
- final survival-best `c_index = 0.5213`
- final OLS `RMSE = 0.4142`
- final MixedLM `RMSE = 0.4683`

Interpretation:

- Relative to strict LOEO, V3 lifts Logit AUC by about `+0.0349`, but it still stays below `0.50`.
- The strongest survival result is the V3 AFT branch (`0.5354`), while the stabilized round settles at `0.5213`.
- Stabilization improves the linear error metrics materially, but the stop rule correctly classifies the round as marginal rather than decisive.
- Cross-event evidence is therefore usable as a transport diagnostic, but still not strong enough to claim stable out-of-event classification performance.

### Exploration Mainline

The exploration rerun is now separated from appendix analyses.

- Cloud ablation:
  - best refreshed Logit AUC among the saved specs is `0.4925`
  - best refreshed Cox `c_index` is `0.5216`
  - linear-model coefficients remain small and sign-fragile across the cloud variants
- Noise-mask sensitivity:
  - OLS `RMSE` improves from `0.4142` in `M0` to `0.3465` in `M2`
  - Logit AUC remains modest (`0.4762` in `M2`)
- Urban/rural split:
  - full-sample `UR_full` outperforms the urban-only slice on Logit AUC (`0.4749` vs `0.4252`)
  - this indicates sample fragmentation and event heterogeneity rather than a clean urban-only gain
- Spatial diagnostics:
  - Moran's I is positive for all six baseline events (`0.274` to `0.592`)
  - spatial dependence remains a real modeling concern
- Extreme-event sensitivity:
  - dropping the two most extreme events raises Logit AUC to `0.5001` and Cox `c_index` to `0.5295`
  - this should be read as sensitivity analysis, not as a replacement headline result

## Appendix-Only Analyses

The following outputs are intentionally retained as appendix analyses only.

### Post-Hoc Quality Adjustment

- `quality_transport_aggregate_metrics_v1.csv`
  - Logit `AUC = 0.4973`
  - Cox `c_index = 0.5174`
  - OLS `RMSE = 0.3955`
- `spatial_block_cv_metrics_v1.csv`
  - Logit `AUC = 0.5949`
  - Cox `c_index = 0.6448`
  - OLS `RMSE = 0.3580`
- `facility_centered_model_summary.csv`
  - facility-matched OLS `coef_in_buffer = 0.0213`, `p = 0.0908`
  - facility-matched Logit `odds_ratio_in_buffer = 0.7308`, `p = 0.00741`

These outputs remain useful as robustness and design diagnostics, but they are not used as cross-event headline evidence because the quality-adjusted labels are post-hoc by construction.

### Hazard / Exposure Appendix

- `hazard_transport_aggregate_metrics_v1.csv`
  - Logit `AUC = 0.6025`
  - Cox `c_index = 0.5341`
  - OLS `RMSE = 0.4163`

These models use event-level hazard/exposure summaries that are informative for explanation, but they are not treated as ex-ante transport evidence in the mainline narrative.

## Archived Files

Pre-fix snapshots for the overwritten code and docs are stored in:

- `project/modeling/legacy/archive_model_fix_20260309/SNAPSHOT.md`
- `project/modeling/legacy/archive_model_fix_20260309/project/modeling/pipelines/01_in_sample_pipeline.py`
- `project/modeling/legacy/archive_model_fix_20260309/project/modeling/pipelines/02_cross_event_pipeline.py`
- `project/modeling/legacy/archive_model_fix_20260309/project/modeling/pipelines/03_exploration_pipeline.py`
- `project/modeling/legacy/archive_model_fix_20260309/project/modeling/pipeline_lib.py`
- `project/modeling/legacy/archive_model_fix_20260309/project/modeling/run_pipeline.py`
- `project/modeling/legacy/archive_model_fix_20260309/project/modeling/pipelines/README.md`
- `project/modeling/legacy/archive_model_fix_20260309/project/modeling_report/index.md`

## Known Limits

- Cross-event benchmark branches were disabled for the 2026-03 rerun so the corrected interpretable LOEO pipeline could complete in a reasonable time. Headline interpretation is unaffected because the README now relies on the interpretable track only.
- MixedLM and Cox branches still emit numerical convergence warnings on some folds; these are tracked but no longer masked by train-score fallbacks.
- Event-expansion / international acquisition utilities still depend on online sources and are not part of the headline rerun path documented here.
