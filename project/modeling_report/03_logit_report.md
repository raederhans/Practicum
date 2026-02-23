# Logistic Model Report / Logit 模型报告

## Objective
评估缓冲区位置是否降低像素受损概率 (damage probability).

## Data & Features
因变量 `is_damaged = 1(delta_ntl < threshold)`，阈值基线 -10%。

## Model Spec
`is_damaged ~ in_buffer * pre_mean_ntl + C(event_id)`.

## Problems & Fixes
No critical issue observed.

Residual risk: sample composition and unobserved confounders may still influence effect size; this risk is tracked in robustness outputs.

## Results
`in_buffer` odds ratio = 0.6831, p-value = 1.022e-07; ROC AUC = 0.7192.

## Figures
- 优势比图 Odds Ratio Plot: `project/modeling_report/figures/logit/logit_odds_ratio.png`
- ROC 曲线 ROC Curve: `project/modeling_report/figures/logit/logit_roc_curve.png`
- 校准图 Calibration: `project/modeling_report/figures/logit/logit_calibration.png`

## Interpretation
OR<1 表示缓冲区像素受损 odds 更低；AUC 反映分类区分能力。

## Limitations
损害阈值定义会影响绝对概率，需结合阈值敏感性解释。

## Next Step
在 robustness 中比较 -5/-10/-15/-20% 阈值结果一致性。
