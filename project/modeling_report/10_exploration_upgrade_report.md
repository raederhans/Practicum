# Exploration Upgrade Report (V2) / 六事件探索增强报告（V2）

## Objective
在不扩事件集合前提下，验证云量、噪声 mask、城乡/人口、空间依赖与极端事件敏感性是否能解释并改善跨事件泛化表现。

## Experiment Matrix
- Cloud: C0/C1/C2/C3
- Noise Mask: M0/M1/M2
- Urban-Rural: UR_full/UR_urban/UR_rural
- Spatial: Moran's I + cluster SE
- Extreme sensitivity: drop-1/drop-2/control-drop

## Cloud Importance Findings
- Logit AUC: C0=0.4679, C1=0.4913, C2=0.4925
- 判定逻辑：若 AUC 提升但 Brier 恶化，则归类为有代价提升。

## Noise Mask Findings
- Logit AUC: M0=0.4913, M2=0.4762
- M1/M2 通过去除高噪声地物类别检验图像背景噪声对可泛化性的影响。

## Urban-Rural + Population Findings
- UR_full Logit AUC=0.4749
- Population source quality: status=ok, coverage=1.000
- 城乡与人口密度用于检验发电机相关韧性是否存在结构性分层。

## Spatial Autocorrelation Findings
- Significant Moran's I events: 6/6
- 已输出 HC1 vs spatial cluster SE 对照，避免显著性高估。

## Indicator Contribution Verdict
### Positive contribution (top)
- Cox | pixel_post_valid_ratio | effect=1.8885 | stability=1.00
- Logit | pixel_cloud_proxy | effect=1.5998 | stability=1.00
- Cox | water_share_1km | effect=1.4301 | stability=1.00
- Cox | land_use_group_unknown | effect=1.1416 | stability=1.00
- Cox | pixel_pre_valid_ratio | effect=1.0074 | stability=1.00

### Negative contribution (top)
- Logit | water_share_1km | effect=-1.8283 | stability=0.98
- Logit | urban_share_1km | effect=-1.3511 | stability=0.98
- Logit | developed_high_share_1km | effect=-1.2045 | stability=1.00
- Cox | event_disaster_type_hurricane | effect=-0.8600 | stability=1.00
- Cox | pixel_cloud_proxy | effect=-0.6532 | stability=0.88

## Extreme-event Identification & Drop Sensitivity
- Decision: structural_shift_not_single_event
- Extreme candidates: maria_sanjuan, earthquake_sanjuan
- 该结论仅用于敏感性，不改变主规格口径。

## Key Outputs
- `project/modeling/output/cloud_ablation_aggregate_metrics.csv`
- `project/modeling/output/noise_mask_experiment_metrics.csv`
- `project/modeling/output/urban_rural_model_comparison.csv`
- `project/modeling/output/spatial_autocorr_morans_i.csv`
- `project/modeling/output/feature_contribution_scorecard.csv`
- `project/modeling/output/extreme_event_drop_aggregate_v1.csv`

## Figures
- `project/modeling_report/figures/exploration_v2/exploration_auc_compare.png`
