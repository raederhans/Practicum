# Modeling Report Index

## Canonical Documentation

- Main workflow and result summary: `project/modeling/README.md`
- Active pipeline scripts:
  - `project/modeling/pipelines/01_in_sample_pipeline.py`
  - `project/modeling/pipelines/02_cross_event_pipeline.py`
  - `project/modeling/pipelines/03_exploration_pipeline.py`

## Deliverables

- `project/modeling_report/01_ols_report.md`
- `project/modeling_report/02_mixedlm_report.md`
- `project/modeling_report/03_logit_report.md`
- `project/modeling_report/04_cox_report.md`
- `project/modeling_report/05_iteration_summary.md`
- `project/modeling_report/06_feature_upgrade_report.md`
- `project/modeling_report/07_logo_validation_report.md`
- `project/modeling_report/08_cross_event_model_report.md`
- `project/modeling_report/09_cross_event_stabilization_report.md`
- `project/modeling_report/10_exploration_upgrade_report.md`
- `project/modeling_report/11_quality_matched_report.md`
- `project/modeling_report/12_hazard_exposure_transport_report.md`
- `project/modeling_report/13_event_increment_report.md`
- `project/modeling_report/14_intl_stage_repair_report.md`

## Current Mainline Artifacts

Strict mainline:

- `project/modeling/output/model_summary_feature_upgrade_v2_strict.csv`
- `project/modeling/output/logo_aggregate_metrics_v2_strict.csv`
- `project/modeling/output/multicollinearity_vif_v2_strict.csv`

Cross-event mainline:

- `project/modeling/output/model_summary_cross_event_v3.csv`
- `project/modeling/output/cross_event_aggregate_metrics_v3.csv`
- `project/modeling/output/cross_event_round_comparison_v3x.csv`
- `project/modeling/output/cross_event_stop_decision_v3x.json`

Exploration mainline:

- `project/modeling/output/cloud_ablation_aggregate_metrics.csv`
- `project/modeling/output/noise_mask_experiment_metrics.csv`
- `project/modeling/output/urban_rural_model_comparison.csv`
- `project/modeling/output/spatial_autocorr_morans_i.csv`
- `project/modeling/output/extreme_event_drop_aggregate_v1.csv`

Appendix-only:

- `project/modeling/output/quality_transport_aggregate_metrics_v1.csv`
- `project/modeling/output/spatial_block_cv_metrics_v1.csv`
- `project/modeling/output/facility_centered_model_summary.csv`
- `project/modeling/output/hazard_transport_aggregate_metrics_v1.csv`
- `project/modeling/output/event_selection_scorecard_v1.csv`

## Strict Mainline Snapshot

Full strict specification (`n = 10,306`):

- OLS `coef_in_buffer = 0.0269`, `p = 0.0544`
- MixedLM `coef_in_buffer = 0.0269`, `p = 0.00943`
- Logit `odds_ratio_in_buffer = 0.7503`, `AUC = 0.7299`
- Cox `hazard_ratio_in_buffer = 1.3319`, `p = 6.15e-15`

Strict LOEO transport:

- OLS `RMSE = 0.4132`
- MixedLM `RMSE = 0.4512`
- Logit `AUC = 0.4549`, `Brier = 0.2542`
- Cox `c_index = 0.5200`

## Cross-Event Status

Corrected V3 rerun:

- AFT `c_index = 0.5354`
- Cox `c_index = 0.4641`
- Logit `AUC = 0.4897`, `Brier = 0.3110`
- OLS `RMSE = 0.5243`
- MixedLM `RMSE = 0.6010`

Stabilization V3x:

- stopped at `r1`
- stop reason: `marginal_improvement`
- final Logit `AUC = 0.4814`
- final survival-best `c_index = 0.5213`
- final OLS `RMSE = 0.4142`
- final MixedLM `RMSE = 0.4683`

Interpretation:

- V3 improves Logit AUC versus strict LOEO but still remains below `0.50`.
- Stabilization helps the linear branches but does not produce a decisive transport gain under the configured stop rule.
- Benchmark outputs are intentionally omitted from this rerun summary.

## Exploration Snapshot

- Cloud ablation saved variants reach Logit AUC up to `0.4925` and Cox `c_index` up to `0.5216`.
- Noise-mask sensitivity improves OLS RMSE down to `0.3465` in `M2`, but classification performance remains modest.
- Urban-only slicing reduces Logit AUC from `0.4749` (`UR_full`) to `0.4252` (`UR_urban`).
- Moran's I remains positive for every baseline event (`0.274` to `0.592`).
- Dropping the two most extreme events raises Logit AUC to `0.5001` and Cox `c_index` to `0.5295`.

## Appendix Snapshot

- Post-hoc quality adjustment:
  - transport Logit `AUC = 0.4973`
  - spatial-block Logit `AUC = 0.5949`
  - facility-matched Logit `odds_ratio_in_buffer = 0.7308`, `p = 0.00741`
- Hazard / exposure appendix:
  - Logit `AUC = 0.6025`
  - Cox `c_index = 0.5341`

These appendix outputs are not headline evidence for ex-ante transport.

## Archive

Pre-fix snapshots for overwritten code and docs are stored in:

- `project/modeling/legacy/archive_model_fix_20260309/`
- V3 cross-event outputs: `project/modeling/output/model_summary_cross_event_v3.csv`, `project/modeling/output/cross_event_aggregate_metrics_v3.csv`
- V3 stabilization outputs: `project/modeling/output/cross_event_round_comparison_v3x.csv`, `project/modeling/output/cross_event_stop_decision_v3x.json`
- `project/modeling_report/bug_transport_report.md`
- Appendix BUG-aware outputs: `project/modeling/output/bug_transport_aggregate_metrics_v1.csv`, `project/modeling/output/bug_transport_feature_audit_v1.csv`
- `project/modeling_report/hazard_transport_readiness_report_v1.md`
- Appendix readiness-filtered hazard outputs: `project/modeling/output/hazard_transport_readiness_aggregate_metrics_v1.csv`, `project/modeling/output/hazard_transport_readiness_events_v1.csv`
- `project/modeling_report/bug2_pr_pilot_report.md`
- Appendix BUG2 pilot outputs: `project/modeling/output/bug2_pilot_acquisition_backlog_v1.csv`, `project/modeling/output/bug2_pr_pilot_qa_v1.csv`
- `project/modeling_report/bug_detectability_transport_report.md`
- Appendix BUG-detectability outputs: `project/modeling/output/bug_detectability_transport_aggregate_metrics_v1.csv`, `project/modeling/output/bug_detectability_transport_feature_audit_v1.csv`
- `project/modeling_report/bug2_pr_acquisition_memo_v1.md`
- Appendix BUG2 pilot outputs: `project/modeling/output/bug2_pilot_acquisition_backlog_v1.csv`, `project/modeling/output/bug2_pr_pilot_qa_v1.csv`, `project/modeling/output/bug2_pr_proxy_overlay_v1.csv`
