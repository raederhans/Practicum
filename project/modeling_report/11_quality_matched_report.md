# Quality + Matched Upgrade Report / 质量控制与设施匹配升级报告

## Objective
- 在不扩事件的前提下，优先提升 target/recovery 质量、空间约束和 buffer 对照可比性。

## What Was Added
- `delta_ntl_obs_adjusted`：低观测质量像素向事件高质量参考值收缩。
- `recovery_daily_panel_v2.parquet`：显式记录 valid days、first valid day、first threshold hit 和 censoring risk。
- `facility_context_panel_v1.parquet`：按最近关键设施生成局地 matched design。
- `spatial_block_cv_metrics_v1.csv`：把空间依赖从诊断升级为 block-level 验证。

## Target Quality Audit
- Mean high-censoring share: 0.003
- Mean absolute delta adjustment: 0.0095
- Worst observed-rate event: maria_sanjuan (0.623)

## Appendix: Post-Hoc Quality Adjustment
- Logit AUC: 0.4973
- Logit Brier: 0.2409
- AFT c-index: 0.4807
- Cox c-index: 0.5174
- OLS RMSE: 0.3955
- MixedLM RMSE: 0.5257

## Spatial Block Validation
- Logit AUC (block CV): 0.5949
- AFT c-index (block CV): 0.3557
- 若 block CV 明显低于 LOEO，则说明原模型仍受空间近邻泄漏影响。

## Facility-Centered Matched Results
- Matched OLS coef(in_buffer): 0.0213, p=0.09077
- Matched Logit OR(in_buffer): 0.7308, p=0.007412
- Paired ATT (treated-control delta_ntl): -0.0004
- Mean matched pairs per facility: 2.68

## Appendix Interpretation
- 如果 quality-aware transport 比当前 v3r1 更稳，说明瓶颈的一部分来自 target/recovery 噪声。
- 如果 matched 结果仍保持 buffer 正向信号，说明关键设施局地韧性并非完全由土地利用与城市密度混淆驱动。
- 如果 spatial block CV 下分数明显回落，说明后续论文口径必须强调空间依赖修正。

## Output Files
- `project/modeling/output/target_quality_audit.csv`
- `project/modeling/output/quality_transport_aggregate_metrics_v1.csv`
- `project/modeling/output/spatial_block_cv_metrics_v1.csv`
- `project/modeling/output/facility_centered_model_summary.csv`
- `project/modeling/output/model_role_matrix_v1.csv`

## Figure
- `project/modeling_report/figures/exploration_v2/quality_matched_compare_v1.png`
