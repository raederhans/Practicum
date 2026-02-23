# Modeling Report Index

## Deliverables
- `project/modeling_report/01_ols_report.md`
- `project/modeling_report/02_mixedlm_report.md`
- `project/modeling_report/03_logit_report.md`
- `project/modeling_report/04_cox_report.md`

## Cross-model key points
- OLS (no_nlcd): coef_in_buffer = 0.0283, p=0.07015
- MixedLM (no_nlcd): coef_in_buffer = 0.0283, p=0.01961
- Logit (no_nlcd): odds_ratio_in_buffer = 0.6831, p=1.022e-07
- Logit (no_nlcd): auc = 0.7192
- Cox (no_nlcd): hazard_ratio_in_buffer = 1.1261, p=4.527e-07

## Consistency & Conflict
- 若 OLS/MixedLM 与 Logit/Cox 对 in_buffer 的方向一致，则支持备用发电机韧性信号存在。
- 若方向冲突，优先检查阈值敏感性、样本构成和删失结构。

## Citation-ready statement
- 在六事件统一像素框架下，控制基线亮度与事件异质性后，关键设施缓冲区在夜光恢复/损伤概率上展现出可检验的韧性差异。