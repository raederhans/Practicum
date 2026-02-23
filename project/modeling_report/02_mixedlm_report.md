# Mixed-Effects Model Report / 混合效应模型报告

## Objective
在事件层级随机截距下估计缓冲区效应，处理像素嵌套结构。

## Data & Features
同 OLS 数据，但模型引入 `event_id` random intercept。

## Model Spec
`delta_ntl ~ in_buffer * pre_mean_ntl` with `groups=event_id`.

## Problems & Fixes
- 症状 Symptom: Cannot predict random effects from singular covariance structure.
  原因 Cause: random_effect_extraction_failed
  修复 Fix: keep fixed effects and predictions; skip random-intercept export
  影响 Impact: random-effect chart unavailable for this variant
  状态 Status: monitor

## Results
核心系数 `in_buffer` = 0.0283, p-value = 0.01961 (no_nlcd baseline).

## Figures
- 固定效应系数 Fixed Effects: `project/modeling_report/figures/mixedlm/mixedlm_fixed_effects.png`
- 随机截距 Random Intercepts: `project/modeling_report/figures/mixedlm/mixedlm_random_effects.png`
- 组内拟合 Group-level Fit: `project/modeling_report/figures/mixedlm/mixedlm_group_fit.png`

## Interpretation
若 MixedLM 与 OLS 的 `in_buffer` 方向一致，说明结论不依赖单一方差假设。

## Limitations
仅建模事件随机截距，未引入空间随机场。

## Next Step
在 NLCD 接入后重跑 mixed model 对比系数变化。
