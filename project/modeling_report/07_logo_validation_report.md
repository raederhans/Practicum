# LOEO Validation Report / 按事件留一验证报告（Strict V2）

## Objective
在 `full_locked_v2_strict` 下执行 LOEO 双规格验证（inference vs transport）。

## Aggregate Metrics
- OLS RMSE: transport=0.4132
- MixedLM RMSE: transport=0.4512
- Logit AUC: transport=0.4549
- Logit Brier: transport=0.2542
- Cox c-index: transport=0.5200

## Sign Consistency
- OLS: transport=0.5000
- MixedLM: transport=0.8333
- Logit: transport=1.0000
- Cox: transport=0.8333

Detail files: `project/modeling/output/logo_fold_metrics_v2_strict.csv`, `project/modeling/output/logo_aggregate_metrics_v2_strict.csv`