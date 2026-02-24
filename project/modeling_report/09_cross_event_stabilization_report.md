# Cross-Event Stabilization Report / 跨事件稳定化修正报告（V3.x）

## Objective
在最多 3 轮内验证稳定化修正是否能把 V3 从边际改进推进到可接受改进；若仍边际改进则按规则收工。

## Stop Rule
- Balanced: `ΔLogit AUC < 0.02` 且 `ΔSurvival(best c-index) < 0.01` -> 停止。
- 最终停止轮次: `r1`
- 停止原因: `marginal_improvement`

## Round Metrics
| round | logit_auc | logit_brier | survival_best_c_index | cox_c_index | aft_c_index | ols_rmse | mixedlm_rmse | hgb_auc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r0 | 0.4897 | 0.3110 | 0.5354 | 0.4641 | 0.5354 | 0.5243 | 0.4842 | 0.5688 |
| r1 | 0.4827 | 0.2658 | 0.5213 | 0.5213 | 0.4773 | 0.4119 | 0.4683 | 0.5377 |

## Stop Decision Evidence
- Comparison file: `project/modeling/output/cross_event_round_comparison_v3x.csv`
- Decision file: `project/modeling/output/cross_event_stop_decision_v3x.json`

## Output Files
- Round fold metrics: `project/modeling/output/cross_event_fold_metrics_v3r*.csv`
- Round aggregates: `project/modeling/output/cross_event_aggregate_metrics_v3r*.csv`
- Round feature importance: `project/modeling/output/cross_event_feature_importance_v3r*.csv`
- Round summaries: `project/modeling/output/model_summary_cross_event_v3r*.csv`

## Conclusion
- 触发边际改进停止规则：今天按计划收工。