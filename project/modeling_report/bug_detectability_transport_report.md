# BUG Detectability Transport Report

## Objective
- Re-test the BUG mechanism using detectability-aware priors that encode likely diesel dominance, size, and runtime rather than facility presence alone.

## Compared Specs
- `BD0`: quality-adjusted transport baseline rerun under the detectability family
- `BD1A`: baseline plus detectability-aware BUG features
- `BD1B`: replace binary `in_buffer` with `high_detect_bug_buffer` and keep baseline controls
- `BD1C`: detectability features plus quality guards, without the legacy baseline spatial context block

## Metric Comparison
- QT1 Logit AUC: 0.4973
- HZ1 Logit AUC: 0.6025
- BUG1A Logit AUC: 0.4965
- BD0 Logit AUC: 0.4973
- BD1A Logit AUC: 0.4931
- BD1A vs BD0 AUC delta: -0.0042
- BD1A Logit Brier: 0.2422
- BD1A vs BD0 Brier delta: 0.0013

## Feature Audit
- earthquake_sanjuan: poi_status=ok, poi_count=434, high_detect_share=0.29, high_detect_bug_buffer_share=0.426
- ida_neworleans: poi_status=ok, poi_count=239, high_detect_share=0.35, high_detect_bug_buffer_share=0.223
- irma_miami: poi_status=ok, poi_count=328, high_detect_share=0.35, high_detect_bug_buffer_share=0.302
- laura_lakecharles: poi_status=ok, poi_count=145, high_detect_share=0.63, high_detect_bug_buffer_share=0.314
- maria_sanjuan: poi_status=ok, poi_count=434, high_detect_share=0.29, high_detect_bug_buffer_share=0.424
- michael_panamacity: poi_status=ok, poi_count=77, high_detect_share=0.40, high_detect_bug_buffer_share=0.145

## Top Detectability Features (Logit)
- BD1A | bug_detect_diesel_proxy_1km: mean_coef=-0.1223, mean_abs=0.1608, sign_consistency=0.83
- BD1A | in_buffer: mean_coef=0.0343, mean_abs=0.1233, sign_consistency=0.83
- BD1A | bug_detect_count_750m: mean_coef=0.0973, mean_abs=0.0973, sign_consistency=1.00
- BD1A | bug_detect_capacity_proxy_1km: mean_coef=-0.0086, mean_abs=0.0796, sign_consistency=0.50
- BD1A | bug_detect_hours_proxy_1km: mean_coef=-0.0261, mean_abs=0.0574, sign_consistency=0.50
- BD1A | bug_detect_count_1250m: mean_coef=0.0036, mean_abs=0.0476, sign_consistency=0.67
- BD1A | bug_detect_score_1km: mean_coef=-0.0158, mean_abs=0.0433, sign_consistency=0.67
- BD1A | bug_detect_min_dist_m: mean_coef=-0.0000, mean_abs=0.0000, sign_consistency=0.83

## Recommendation
- BD1A does not improve transport enough to justify more proxy-only tuning.
- Treat BUG-detectability as the strongest remaining proxy-enhancement test.
- If it still fails to add ranking value, stop expanding proxy-only families and keep BUG2 focused on official inventory validation.

## Outputs
- `project/modeling/output/bug_detectability_transport_fold_metrics_v1.csv`
- `project/modeling/output/bug_detectability_transport_aggregate_metrics_v1.csv`
- `project/modeling/output/bug_detectability_transport_feature_summary_v1.csv`
- `project/modeling/output/bug_detectability_transport_feature_audit_v1.csv`
- `project/modeling_report/figures/exploration_v2/bug_detectability_transport_compare_v1.png`
