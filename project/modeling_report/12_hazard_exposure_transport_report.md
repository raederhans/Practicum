# Hazard/Exposure Transport Report / 灾害强度与暴露特征主线报告

## Objective
- 在 quality-adjusted target 基础上，加入低维 hazard/exposure 特征，重做跨事件 transport 主线。

## Added Mainline Features
- `event_cloud_shift`
- `event_precip_log1p`
- `event_duration_log1p`
- `event_elevation_log1p`
- `event_slope_milli`
- `island_local_water`
- `island_local_urban`
- `hazard_cloud_water`
- `hazard_precip_urban`

## Metric Comparison
- v3r1 Logit AUC: 0.4827
- quality Logit AUC: 0.4973
- hazard Logit AUC: 0.6025
- v3r1 survival best: 0.5213
- quality survival best: 0.5174
- hazard survival best: 0.5341
- hazard Logit Brier: 0.4424

## Top Hazard Features (Logit)
- island_local_urban: mean_coef=2.2291, abs=2.2291, sign_consistency=1.00
- event_cloud_shift: mean_coef=1.3656, abs=1.3656, sign_consistency=1.00
- event_precip_log1p: mean_coef=-1.1555, abs=1.1555, sign_consistency=1.00
- island_local_water: mean_coef=-1.1239, abs=1.1239, sign_consistency=1.00
- event_duration_log1p: mean_coef=1.0017, abs=1.0017, sign_consistency=1.00

## Event Selection Signals
- michael_panamacity: damage_auc=0.3562, survival_best=0.5797, signal=poor_damage_transport_holdout
- laura_lakecharles: damage_auc=0.5296, survival_best=0.5470, signal=add_more_low_urban_events
- irma_miami: damage_auc=0.5439, survival_best=0.5097, signal=representative_keep

## Recommendation
- 本轮 hazard/exposure 主线显著提升了 damage ranking（AUC），说明事件级暴露差异确实是当前跨事件主线缺失的信息。
- 但 hazard Logit 的 Brier 明显变差，说明它更会排序、但概率更不稳；后续若用于预测，应加校准或减弱事件级强特征。
- 后续扩事件时，应优先补足灾种、岛屿性、城市层级和中等水域暴露这几个维度。

## Outputs
- `project/modeling/output/hazard_transport_aggregate_metrics_v1.csv`
- `project/modeling/output/hazard_transport_feature_summary_v1.csv`
- `project/modeling/output/event_selection_scorecard_v1.csv`
- `project/modeling_report/figures/exploration_v2/hazard_transport_compare_v1.png`
