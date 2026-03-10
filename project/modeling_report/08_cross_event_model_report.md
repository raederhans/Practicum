# Cross-Event Model Report / 跨事件预测模型报告（V3）

## Objective
目标是提升 LOEO/transport 外推能力，同时解释为何 strict-v2 在跨事件预测上偏弱。

## Why LOEO was low
核心原因是事件间协变量漂移（domain shift）显著：灾害类型、局地土地利用结构、云量有效观测代理和设施密度分布存在系统差异。

## Data & Event Profiles
- Event profile: `project/modeling/pixel_data/event_profile_v1.csv`
- Enriched panel: `project/modeling/pixel_data/all_events_pixel_panel_v1_cross_event_v3.parquet`
- Shift diagnostics: `project/modeling/output/cross_event_shift_diagnostics_v3.csv`

### Event Profile Snapshot
| event_id | disaster_type | event_duration_days | storm_precip_7d | quality_flag |
|---|---|---|---|---|
| earthquake_sanjuan | earthquake | 38 | 13.7000 | ok;ok;cloud_summary_fallback |
| ida_neworleans | hurricane | 59 | 252.7000 | ok;ok;cloud_summary_fallback |
| irma_miami | hurricane | 49 | 182.4000 | ok;ok;cloud_summary_fallback |
| laura_lakecharles | hurricane | 89 | 177.8000 | ok;ok;cloud_summary_fallback |
| maria_sanjuan | hurricane | 63 | 14.9000 | ok;ok;cloud_summary_fallback |
| michael_panamacity | hurricane | 76 | 147.6000 | ok;ok;cloud_summary_fallback |

## Specs
- Interpretable track: OLS, MixedLM, Logit, Cox, AFT（统一 transport 口径，不使用 event FE 作为主预测特征）
- Benchmark track: HistGradientBoostingRegressor / HistGradientBoostingClassifier
- Validation: LOEO (6 folds), each fold trains on 5 events and tests on 1 unseen event.

## Fold Results
- Logit AUC: 0.4897 (vs strict-v2 0.4549, Δ=+0.0349)
- Logit Brier: 0.3110 (vs strict-v2 0.2542, Δ=+0.0568)
- Cox c-index: 0.4641 (vs strict-v2 0.5200, Δ=-0.0559)
- OLS RMSE: 0.5243 (vs strict-v2 0.4132, Δ=+0.1111)
- MixedLM RMSE: 0.6010 (vs strict-v2 0.4512, Δ=+0.1498)
- HGBClassifier AUC: N/A

## Improvement Verdict
- Verdict: **Partially Improved**
- 判定口径：Logit AUC 是否提升到 >=0.50 或较 strict-v2 至少 +0.03；Cox c-index 是否不降并接近/超过 0.52。

## Failure Cases
下列事件表现为主要外推压力源（按 `smd_mean` 排序）：
- earthquake_sanjuan: smd_mean=0.762
- laura_lakecharles: smd_mean=0.760

## Figures
- Pairwise shift heatmap: `project/modeling_report/figures/cross_event/shift_pairwise_smd_heatmap_v3.png`
- Transport metrics compare: `project/modeling_report/figures/cross_event/transport_metrics_compare_v3.png`
- HGB permutation importance: `project/modeling_report/figures/cross_event/hgb_permutation_importance_v3.png`

## Next Actions
- 固定同样本下，增加事件强度变量（例如最大风速/震级/降水极值）并做分层 LOEO。
- 对高漂移事件启用 group-reweight（GroupDRO 思路）并与当前 ERM 结果对照。
- 保留 interpretable 主结果，benchmark 仅作为性能上界参考。

## References
- [Invariant Risk Minimization (Arjovsky et al., 2019)](https://arxiv.org/abs/1907.02893)
- [DomainBed: In Search of Lost Domain Generalization (Gulrajani & Lopez-Paz, 2020)](https://arxiv.org/abs/2007.01434)
- [Group DRO (Sagawa et al., 2019)](https://arxiv.org/abs/1911.08731)
- [Spatiotemporal distribution of power outages with climate events (Do et al., 2023)](https://www.nature.com/articles/s41467-023-38084-6)
- [Antecedent rainfall and outage risk (Manning et al., 2025)](https://www.nature.com/articles/s43247-025-02176-6)