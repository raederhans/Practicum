# Modeling Report Index

## Deliverables
- `project/modeling_report/01_ols_report.md`
- `project/modeling_report/02_mixedlm_report.md`
- `project/modeling_report/03_logit_report.md`
- `project/modeling_report/04_cox_report.md`
- `project/modeling_report/05_iteration_summary.md`
- `project/modeling_report/06_feature_upgrade_report.md`
- `project/modeling_report/07_logo_validation_report.md`

## Reuse-first Upgrade (Current Round)
- Reuse outputs:
  - `project/modeling/output/teammate_reuse_manifest.csv`
  - `project/modeling/output/teammate_reuse_gap.csv`
  - `project/modeling/output/teammate_reuse_sync_log.csv`
  - `project/modeling/output/input_data_gate_report.csv`
- Cloud feature outputs:
  - `project/modeling/output/cloud_feature_summary.csv`
  - `project/modeling/pixel_data/all_events_pixel_panel_v1.parquet` (cloud columns attached)
- Upgrade figures:
  - `project/modeling_report/figures/feature_upgrade/model_compare_before_after.png`
  - `project/modeling_report/figures/feature_upgrade/cloud_usable_ratio_by_event.png`
  - `project/modeling_report/figures/feature_upgrade/teammate_sync_summary.png`

## Cross-model key points
- OLS (no_nlcd): coef_in_buffer = 0.0283, p=0.07015
- OLS (with_nlcd): coef_in_buffer = -0.0238, p=0.122
- MixedLM (no_nlcd): coef_in_buffer = 0.0283, p=0.01961
- MixedLM (with_nlcd): coef_in_buffer = -0.0238, p=0.04531
- Logit (no_nlcd): odds_ratio_in_buffer = 0.6831, p=1.022e-07
- Logit (with_nlcd): odds_ratio_in_buffer = 1.1777, p=0.1049
- Logit (no_nlcd): auc = 0.7192
- Logit (with_nlcd): auc = 0.7490
- Cox (no_nlcd): hazard_ratio_in_buffer = 1.1261, p=4.527e-07
- Cox (with_nlcd): hazard_ratio_in_buffer = 1.0536, p=0.06921

## Land-use Control Delta (with_nlcd - no_nlcd)
- OLS `coef_in_buffer`: no_nlcd=0.0283, with_nlcd=-0.0238, delta=-0.0521
- MixedLM `coef_in_buffer`: no_nlcd=0.0283, with_nlcd=-0.0238, delta=-0.0521
- Logit `odds_ratio_in_buffer`: no_nlcd=0.6831, with_nlcd=1.1777, delta=0.4946
- Cox `hazard_ratio_in_buffer`: no_nlcd=1.1261, with_nlcd=1.0536, delta=-0.0726

## Consistency & Conflict
- 若 OLS/MixedLM 与 Logit/Cox 对 in_buffer 的方向一致，则支持备用发电机韧性信号存在。
- 若方向冲突，优先检查阈值敏感性、样本构成和删失结构。

## Citation-ready statement
- 在六事件统一像素框架下，控制基线亮度与事件异质性后，关键设施缓冲区在夜光恢复/损伤概率上展现出可检验的韧性差异。
- Strict V2 outputs: `project/modeling/output/model_summary_feature_upgrade_v2_strict.csv`, `project/modeling/output/logo_aggregate_metrics_v2_strict.csv`
- `project/modeling_report/08_cross_event_model_report.md`
- V3 cross-event outputs: `project/modeling/output/model_summary_cross_event_v3.csv`, `project/modeling/output/cross_event_aggregate_metrics_v3.csv`
- `project/modeling_report/09_cross_event_stabilization_report.md`
- V3 stabilization outputs: `project/modeling/output/cross_event_round_comparison_v3x.csv`, `project/modeling/output/cross_event_stop_decision_v3x.json`
