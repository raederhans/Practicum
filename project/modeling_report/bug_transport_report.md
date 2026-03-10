# BUG-aware Transport Report

## Objective
- Keep the existing mainline untouched and test whether BUG-aware proxy features improve damage transport stability.

## Compared Specs
- `BUG0`: quality-adjusted transport baseline rerun under the BUG family
- `BUG1A`: baseline plus prior-weighted BUG features
- `BUG1B`: replace binary `in_buffer` with `high_conf_bug_buffer` and keep baseline controls
- `BUG1C`: BUG prior features plus quality guards, without the legacy baseline spatial context block

## Metric Comparison
- QT1 Logit AUC: 0.4973
- HZ1 Logit AUC: 0.6025
- BUG0 Logit AUC: 0.4973
- BUG1A Logit AUC: 0.4965
- BUG1A vs BUG0 AUC delta: -0.0008
- BUG1A Logit Brier: 0.2392
- BUG1A vs BUG0 Brier delta: -0.0017

## Feature Audit
- earthquake_sanjuan: poi_status=ok, poi_count=434, high_conf_share=0.31, high_conf_bug_buffer_share=0.431
- ida_neworleans: poi_status=ok, poi_count=239, high_conf_share=0.38, high_conf_bug_buffer_share=0.232
- irma_miami: poi_status=ok, poi_count=328, high_conf_share=0.35, high_conf_bug_buffer_share=0.303
- laura_lakecharles: poi_status=ok, poi_count=145, high_conf_share=0.70, high_conf_bug_buffer_share=0.320
- maria_sanjuan: poi_status=ok, poi_count=434, high_conf_share=0.31, high_conf_bug_buffer_share=0.429
- michael_panamacity: poi_status=ok, poi_count=77, high_conf_share=0.40, high_conf_bug_buffer_share=0.145

## Top BUG Features (Logit)
- BUG1A | bug_prior_hours_proxy_1km: mean_coef=-0.1556, mean_abs=0.1556, sign_consistency=1.00
- BUG1A | in_buffer: mean_coef=0.0325, mean_abs=0.1210, sign_consistency=0.83
- BUG1A | bug_prior_count_750m: mean_coef=0.1003, mean_abs=0.1003, sign_consistency=1.00
- BUG1A | bug_prior_count_1250m: mean_coef=0.0134, mean_abs=0.0562, sign_consistency=0.67
- BUG1A | bug_prior_capacity_proxy_1km: mean_coef=-0.0246, mean_abs=0.0274, sign_consistency=0.83
- BUG1A | bug_prior_min_dist_m: mean_coef=-0.0000, mean_abs=0.0000, sign_consistency=0.83
- BUG1B | in_buffer: mean_coef=0.1987, mean_abs=0.1987, sign_consistency=1.00
- BUG1B | bug_prior_hours_proxy_1km: mean_coef=-0.1434, mean_abs=0.1434, sign_consistency=1.00

## Recommendation
- BUG1A does not improve transport enough to justify moving to official inventory work yet.
- Keep `strict-v2`, `quality_transport`, and `hazard_transport` unchanged; this report is parallel-only evidence.

## Outputs
- `project/modeling/output/bug_transport_fold_metrics_v1.csv`
- `project/modeling/output/bug_transport_aggregate_metrics_v1.csv`
- `project/modeling/output/bug_transport_feature_summary_v1.csv`
- `project/modeling/output/bug_transport_feature_audit_v1.csv`
- `project/modeling_report/figures/exploration_v2/bug_transport_compare_v1.png`
