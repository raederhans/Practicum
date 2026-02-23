# Cox PH Model Report / Cox 生存模型报告

## Objective
比较缓冲区与非缓冲区像素恢复速度 (recovery speed) 差异。

## Data & Features
使用 `recovery_days` 与 `event_observed`（右删失处理）。

## Model Spec
Cox proportional hazards on in_buffer + baseline + event dummies.

## Problems & Fixes
No critical issue observed.

Residual risk: sample composition and unobserved confounders may still influence effect size; this risk is tracked in robustness outputs.

## Results
`in_buffer` hazard ratio = 1.1261, p-value = 4.527e-07 (threshold 90%).

## Figures
- Kaplan-Meier 曲线: `project/modeling_report/figures/cox/cox_km_curve.png`
- 风险比图 Hazard Ratio Plot: `project/modeling_report/figures/cox/cox_hazard_ratio.png`
- PH 检验图 Proportional Hazard Test: `project/modeling_report/figures/cox/cox_ph_test.png`

## Interpretation
HR>1 表示更快达到恢复阈值；若显著则支持缓冲区恢复优势。

## Limitations
恢复定义受阈值影响；观测窗口长度不一致会影响删失比例。

## Next Step
执行 80/90/95% 阈值敏感性并与主结果对照。
