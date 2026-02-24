# LOEO Validation Report / 按事件留一验证报告（Strict V2）

## Objective
在 `full_locked_v2_strict` 下执行 LOEO 双规格验证（inference vs transport）。

## Aggregate Metrics
- OLS RMSE: inference=0.3695, transport=0.4243
- MixedLM RMSE: inference=0.4354, transport=0.4437
- Logit AUC: inference=0.7300, transport=0.4654
- Logit Brier: inference=0.1982, transport=0.2578
- Cox c-index: inference=0.5315, transport=0.5121

## Sign Consistency
- OLS: inference=0.8333, transport=1.0000
- MixedLM: inference=0.8333, transport=0.8333
- Logit: inference=1.0000, transport=1.0000
- Cox: inference=1.0000, transport=1.0000

Detail files: `project/modeling/output/logo_fold_metrics_v2_strict.csv`, `project/modeling/output/logo_aggregate_metrics_v2_strict.csv`