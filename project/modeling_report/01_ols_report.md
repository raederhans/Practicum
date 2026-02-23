# OLS Model Report / OLS 模型报告

## Objective
评估在控制基线亮度与事件差异后，缓冲区像素是否表现出更高韧性 (resilience).

## Data & Features
使用 `all_events_pixel_panel_v1.parquet`，核心变量为 `delta_ntl`, `in_buffer`, `pre_mean_ntl`, `event_id`。

## Model Spec
`delta_ntl ~ in_buffer * pre_mean_ntl + C(event_id)` (HC1 robust SE)

## Problems & Fixes
No critical issue observed.

Residual risk: sample composition and unobserved confounders may still influence effect size; this risk is tracked in robustness outputs.

## Results
核心系数 `in_buffer` = 0.0283, p-value = 0.07015 (no_nlcd baseline).

## Figures
- 系数图 Coefficient Plot: `project/modeling_report/figures/ols/ols_coefficients.png`
- 预测-实际 Predicted vs Actual: `project/modeling_report/figures/ols/ols_pred_vs_actual.png`
- 残差诊断 Residual Diagnostic: `project/modeling_report/figures/ols/ols_residual_diagnostic.png`

## Interpretation
若 `in_buffer` 为正且显著，说明在同等基线下，设施缓冲区像素夜光下降更少。

## Limitations
OLS 假设独立同分布；像素空间相关可能造成标准误低估。

## Next Step
与 MixedLM 对照随机效应后确认结论稳健性。
