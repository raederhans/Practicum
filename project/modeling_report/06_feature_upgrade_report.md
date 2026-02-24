# Feature Upgrade Report / OSM+Cloud+Sample Lock 升级报告（Strict V2）

## Objective
本轮按 Strict V2 规则重训：`Proxy only` 云量、`dist_any only` 距离、`strict no fallback`。

## Strict Spec
- Full 主规格: `in_buffer * pre_mean_ntl_centered + C(event_id) + C(land_use_group) + osm_dist_any_m + osm_power_count_1000m + osm_medical_count_1000m + pixel_cloud_proxy`
- `missing_osm_flag` 与 `missing_cloud_flag` 不入主模型，仅保留审计。

## Quantitative Comparison
- OLS `coef_in_buffer`: baseline=0.0137 (p=0.3357), nlcd=0.0155 (p=0.2637), full=0.0254 (p=0.09753)
- MixedLM `coef_in_buffer`: baseline=0.0137 (p=0.1568), nlcd=0.0155 (p=0.1078), full=0.0254 (p=0.01494)
- Logit `odds_ratio_in_buffer`: baseline=0.7855 (p=1.887e-05), nlcd=0.7887 (p=2.739e-05), full=0.7823 (p=7.732e-05)
- Cox `hazard_ratio_in_buffer`: baseline=1.2551 (p=2.059e-11), nlcd=1.2543 (p=3.124e-11), full=1.3274 (p=1.293e-14)
- Logit `AUC`: baseline=0.7212, nlcd=0.7211, full=0.7302

## Collinearity Gate
- Max VIF: 8.7133
- Gate result (`VIF < 10`): PASS
- Detail: `project/modeling/output/multicollinearity_vif_v2_strict.csv`

## Missing-Flag Audit
- `missing_osm_flag` global distribution: `project/modeling/output/missing_flag_audit_v2_strict.csv`
- 当前数据下 `missing_osm_flag` 为全 0，说明 OSM 加载完整；作为常数项无信息量，已从主模型移除。

## Outputs
- Summary: `project/modeling/output/model_summary_feature_upgrade_v2_strict.csv`
- Cox diagnostics: `project/modeling/output/cox_diagnostics_extended_v2_strict.csv`
- LOEO aggregate: `project/modeling/output/logo_aggregate_metrics_v2_strict.csv`