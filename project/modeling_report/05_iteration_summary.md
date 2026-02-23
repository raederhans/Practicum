# Iteration Summary Report / 三阶段迭代总结报告

## Objective
本报告对比三次建模阶段（Stage A Baseline -> Stage B Initial NLCD -> Stage C Corrected NLCD），并回答核心问题：**这一次是否有改进（Did we improve）**。

## Version Timeline
- **Stage A / Baseline (no_nlcd)**: 第一轮主链路结果，无 land-use 控制。
- **Stage B / Initial NLCD (reconstructed)**: 首次接入 NLCD，存在覆盖率不足与规格冲突信号。
- **Stage C / Corrected NLCD (current)**: 修正 NLCD 覆盖后当前结果，`nlcd_coverage` 最小值提升到 `0.9973`。

## What Changed Each Iteration
- **Stage A**
  - 仅 baseline controls（`in_buffer * pre_mean_ntl + event controls`）。
  - 结果方向较一致：MixedLM 与 Cox 支持 buffer 韧性信号。
- **Stage B**
  - 引入 NLCD 控制，但 Michael 事件覆盖不足（min coverage 约 `0.6949`）。
  - with_nlcd 核心系数接近 0，冲突信号开始出现。
- **Stage C**
  - 完成覆盖修正（all events > 0.9，当前 min `0.9973`）。
  - 四模型产物完整，但跨模型方向仍不一致（MixedLM 与 Cox 分化）。

## Quantitative Comparison Table
Data source: `project/modeling/output/iteration_comparison_snapshot.csv`

| Model | Metric | Stage A Baseline | Stage B Initial NLCD (reconstructed) | Stage C Corrected NLCD |
|---|---|---:|---:|---:|
| OLS | coef_in_buffer | 0.0283 (p=0.0701) | 0.0075 (p=0.5982) | -0.0233 (p=0.1292) |
| MixedLM | coef_in_buffer | 0.0283 (p=0.0196) | 0.0074 (p=0.5225) | -0.0233 (p=0.0496) |
| Logit | odds_ratio_in_buffer | 0.6831 (p=1.02e-07) | 0.9659 (p=0.7454) | 1.1676 (p=0.1243) |
| Logit | AUC | 0.7192 | 0.7590 | 0.7488 |
| Cox | hazard_ratio_in_buffer | 1.1261 (p=4.53e-07) | 1.1083 (p=1.94e-05) | 1.1075 (p=2.00e-05) |

Coverage comparison:
- Stage A: N/A (no NLCD)
- Stage B: min coverage = `0.6949`
- Stage C: min coverage = `0.9973`

## Consistency Check Across Models
- **Stage A**: MixedLM positive, Cox HR>1, Logit OR<1，整体韧性方向相对一致。
- **Stage B**: OLS/MixedLM 接近 0，Logit 接近 1，Cox 仍为正，出现结构性分歧。
- **Stage C**: 工程与数据质量显著提升，但统计方向冲突仍在（MixedLM negative vs Cox positive）。

## Did We Improve? (Verdict)
**Verdict: `Partially Improved`**

判定依据（按预设规则逐条映射）：
1. **工程与数据层改进**: 是。NLCD 覆盖率从不足提升到 >0.9（当前 min 0.9973），且四模型输出齐全。
2. **统计一致性改进**: 否（未完全达成）。当前规格下 MixedLM 与 Cox 的方向未收敛为同向稳定。
3. **冲突可解释性改进**: 是。冲突来源可定位：样本重定义、land-use 控制规格差异、Logit 稳定性问题。

因此总体结论为 **Partially Improved**，不是 Not Improved，也不是 Fully Improved。

## Problems Still Open
- `with_nlcd` 当前仍采用开发用地子样本（21/22/23/24），造成样本构成变化，对跨阶段可比性不利。
- Logit 在细分类 land-use 控制下存在 separation / 收敛不稳风险。
- 原始六事件 post-event daily tifs 缺失，导致 recovery robustness（80/90/95%）仍不完整。
- MixedLM 随机效应奇异问题仍在（已监控但未根治）。

## Next Priorities
1. 固定样本主规格收敛：`no_nlcd` vs `with_nlcd` 使用同样本，仅新增 land-use 控制。
2. 固化论文主口径：MixedLM + Cox 双主结果，并给出一致性裁决规则。
3. Logit 稳定化：正则化/降维 land-use，规避 separation。
4. 空间相关控制：Moran's I + spatial cluster SE。
5. 补齐原始 post-event tif 并重建 recovery panel（下一里程碑，不阻塞本次总结交付）。

## Audit Note
- Stage B 指标来自历史执行记录重建，不是当前重跑产物。
- 重建方法与局限详见：`project/modeling_tracking/progress_record/03_iteration_snapshot_method.md`。
