# Feature Upgrade Report / OSM+Cloud+Sample Lock 升级报告（Strict V2）

## Objective
本轮按 Strict V2 规则重训：`Proxy only` 云量、`dist_any only` 距离、`strict no fallback`。

## Strict Spec
- Full 主规格: `in_buffer * pre_mean_ntl_centered + C(event_id) + C(land_use_group) + osm_dist_any_m + osm_power_count_1000m + osm_medical_count_1000m + pixel_cloud_proxy`
- `missing_osm_flag` 与 `missing_cloud_flag` 不入主模型，仅保留审计。

## Quantitative Comparison
- OLS `coef_in_buffer`: baseline=0.0193 (p=0.1538), nlcd=0.0210 (p=0.1133), full=0.0269 (p=0.05439)
- MixedLM `coef_in_buffer`: baseline=0.0193 (p=0.04815), nlcd=0.0209 (p=0.03263), full=0.0269 (p=0.009427)
- Logit `odds_ratio_in_buffer`: baseline=0.7420 (p=8.344e-08), nlcd=0.7465 (p=1.752e-07), full=0.7503 (p=1.832e-06)
- Cox `hazard_ratio_in_buffer`: baseline=1.2613 (p=7.569e-12), nlcd=1.2603 (p=1.19e-11), full=1.3319 (p=6.152e-15)
- Logit `AUC`: baseline=0.7192, nlcd=0.7207, full=0.7299

## Collinearity Gate
- Max VIF: 10.3982
- Gate result (`VIF < 12`): PASS
- Detail: `project/modeling/output/multicollinearity_vif_v2_strict.csv`

## Missing-Flag Audit
- `missing_osm_flag` global distribution: `project/modeling/output/missing_flag_audit_v2_strict.csv`
- 当前数据下 `missing_osm_flag` 为全 0，说明 OSM 加载完整；作为常数项无信息量，已从主模型移除。

## Outputs
- Summary: `project/modeling/output/model_summary_feature_upgrade_v2_strict.csv`
- Cox diagnostics: `project/modeling/output/cox_diagnostics_extended_v2_strict.csv`
- LOEO aggregate: `project/modeling/output/logo_aggregate_metrics_v2_strict.csv`