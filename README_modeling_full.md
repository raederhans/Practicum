# 六事件建模总览 README（Modeling Full Summary）

## 1) 项目目标与建模口径
本项目围绕“关键设施缓冲区（buffer）是否体现更强韧性（resilience）”展开，使用六事件像素级夜光（NTL）数据进行多模型建模与对照验证。

核心口径：
- 解释口径（in-sample / event-aware）：OLS、MixedLM、Logit、Cox
- 外推口径（cross-event / LOEO transport）：去事件固定效应或不依赖事件哑变量，验证跨事件泛化
- 主线数据：六事件统一像素样本（`sample_lock_flag=1`）

---

## 2) 数据与特征演进（每轮加了什么）

### Round A: Baseline（no_nlcd）
新增/使用要素：
- 核心像素字段：`pre_mean_ntl`, `post_mean_ntl`, `delta_ntl`, `in_buffer`
- 设施距离与密度：`distance_to_nearest`, `n_facilities_in_buffer`
- 事件控制：`C(event_id)`（线性/分类模型）

目标：
- 建立第一版可复现主链路，验证 `in_buffer` 是否有方向性信号。

---

### Round B: Initial NLCD（历史重建）
新增要素：
- `land_use`/`land_use_group`（NLCD 控制）

问题：
- 首次 NLCD 覆盖不完整（历史记录 min coverage ~0.6949），统计结果出现不稳定。

---

### Round C: Corrected NLCD（coverage 修正）
新增/修正要素：
- NLCD 覆盖修正后重跑（min coverage ~0.9973）

结果：
- 工程与数据质量显著提升，但模型间方向冲突未完全消失。

---

### Round D: Feature Upgrade（OSM + Cloud + Sample Lock）
新增要素：
- 样本锁定：`sample_lock_cohort_v1.parquet`
- OSM 特征（Track A）：`osm_any_count_750m`, `osm_power_count_1000m`, `osm_medical_count_1000m`, `osm_dist_any_m`, `osm_dist_power_m`
- 像素级云量代理：`pixel_pre_valid_ratio`, `pixel_post_valid_ratio`, `pixel_cloud_proxy`
- 缺失审计标记：`missing_osm_flag`, `missing_cloud_flag`

目标：
- 修复“样本重定义”和“事件级云量常数不可识别”问题，提升机制解释力。

---

### Round E: Strict V2（严格规格）
新增/收紧规则：
- 主规格固定：`Proxy-only` 云量（`pixel_cloud_proxy`），`dist_any-only` 距离（`osm_dist_any_m`）
- 严格排除主模型变量：`missing_*` 等仅用于审计
- VIF 门禁：`VIF < 10`
- Fail-fast：关键模型缺失即失败

目的：
- 通过严格规格和可识别性约束提高结果可信度。

---

### Round F: Cross-event V3（跨事件预测）
新增要素：
- 事件画像：`event_profile_v1.csv`
- 局地土地利用结构占比：`urban_share_1km`, `water_share_1km`, `developed_high_share_1km`
- 事件级特征注入：`event_disaster_type`, `event_*`
- LOEO（leave-one-event-out）transport 评估
- Benchmark：`HGBClassifier`/`HGBRegressor`

目的：
- 从“事件内解释”转向“跨事件泛化”。

---

### Round G: V3 Stabilization（稳定化，最多3轮）
策略：
- Stability-first（移除事件级常数数值特征在 interpretable 主规格中的主作用）
- Logit L2 正则
- Survival 采用 Cox + AFT 双轨，主指标取 best c-index
- 边际改进停止规则（Balanced）

结果：
- 在 r1 即触发停止，不继续 r2/r3。

---

## 3) 每类模型如何构建（设计流程）

### 3.1 OLS
目标：
- 解释连续响应 `delta_ntl` 的平均线性效应。

典型规格：
- Baseline: `delta_ntl ~ in_buffer * pre_mean_ntl + C(event_id)`
- NLCD/Strict: 再加入 `C(land_use_group)` 与选定数值控制项
- Transport: 去 `C(event_id)`，只保留可迁移特征

评估：
- 系数（`coef_in_buffer`）、`p-value`
- 预测误差：RMSE / MAE

---

### 3.2 MixedLM
目标：
- 在事件层随机效应下估计固定效应，降低事件间未观测异质性干扰。

典型规格：
- `delta_ntl ~ ...`，groups=`event_id`

评估：
- 固定效应系数（特别是 `in_buffer`）
- RMSE / MAE
- 随机效应稳定性（奇异协方差告警）

---

### 3.3 Logit
目标：
- 预测/解释损伤概率（`is_damaged`）。

典型规格：
- `is_damaged ~ in_buffer * pre_mean_ntl + controls`
- 稳定化轮次引入 L2 正则与（可选）校准

评估：
- `odds_ratio_in_buffer`
- AUC, Brier, calibration slope

---

### 3.4 Cox / AFT（生存分析）
目标：
- 建模恢复时间（`recovery_days`）与恢复事件（`event_observed`）。

典型规格：
- CoxPH（含 penalizer、必要时 time interaction 或 strata）
- Weibull AFT 作为对照或外推主候选

评估：
- Cox: hazard ratio + c-index
- AFT: c-index（外推口径下常作为稳健补充）

---

### 3.5 Benchmark（HGB）
目标：
- 提供“预测性能上界参考”，不替代解释模型。

模型：
- `HistGradientBoostingClassifier`（AUC/Brier）
- `HistGradientBoostingRegressor`（RMSE/MAE）

---

## 4) 关键轮次结果与改进总结

### 4.1 Baseline 与 NLCD阶段（A/B/C）
来自 `project/modeling/output/model_summary_for_report.csv` 与 `project/modeling/output/iteration_comparison_snapshot.csv`：
- Baseline（no_nlcd）：
  - OLS `coef_in_buffer = 0.0283`（p=0.0701）
  - MixedLM `coef_in_buffer = 0.0283`（p=0.0196）
  - Logit `OR = 0.6831`, AUC=0.7192
  - Cox `HR = 1.1261`（p<1e-6）
- Corrected NLCD（with_nlcd）：
  - OLS / MixedLM 系数转负（约 -0.0233）
  - Logit OR > 1, AUC ~0.7488
  - Cox HR 仍 > 1

结论：
- NLCD 控制显著影响了系数方向和解释口径，说明土地利用混淆真实存在。

---

### 4.2 Feature Upgrade 与 Strict V2（D/E）
来自 `project/modeling/output/model_summary_feature_upgrade_v2_strict.csv`：
- Strict V2 full：
  - OLS `coef_in_buffer = 0.0254`（p=0.0975）
  - MixedLM `coef_in_buffer = 0.0254`（p=0.0149，显著）
  - Logit `OR = 0.7823`, AUC=0.7302
  - Cox `HR = 1.3274`（p≈1.29e-14）

结论：
- 在事件内解释口径下，Strict V2 是目前最稳定、最可解释的一版。

---

### 4.3 Cross-event V3（F）
来自 `project/modeling/output/model_summary_cross_event_v3.csv`：
- Logit AUC: 0.4897（相对 strict-v2 transport +0.0242）
- Cox c-index: 0.4641（相对 strict-v2 transport -0.0479）
- HGBClassifier AUC: 0.5688（高于 interpretable Logit）

结论：
- 跨事件泛化仅“部分改善”，尤其生存模型外推退化明显。

---

### 4.4 Stabilization（G）
来自 `project/modeling/output/cross_event_aggregate_metrics_v3r1.csv` 与 `project/modeling/output/cross_event_stop_decision_v3x.json`：
- r1:
  - Logit AUC: 0.4827（较 r0 下降）
  - Survival best c-index: 0.5213（较 r0 下降）
- 满足边际改进停止条件，停止于 `r1`。

结论：
- 本轮稳定化未带来跨事件关键指标提升，按规则收工是正确决策。

---

## 5) 每轮改进思路（方法论演进）
1. **先跑通主链路**：Baseline 四模型统一框架。
2. **先修数据再修统计**：NLCD 覆盖问题先解决，再谈系数含义。
3. **引入机制变量**：OSM 与像素云量增强解释。
4. **收紧规格**：Strict V2 防止“隐性回退”与共线性误判。
5. **转向泛化目标**：LOEO + benchmark 定位 domain shift。
6. **设置止损机制**：Stabilization + stop rule，避免低收益反复迭代。

---

## 6) 最后一轮遇到的问题、原因与解决路径

### 6.1 直接问题
- 在 LOEO 折中，`event_disaster_type` 出现“训练集未见类别”导致公式模型预测失败（尤其留出 earthquake 折）。
- MixedLM 在高异质折中出现奇异协方差与优化失败。
- Survival 在跨事件 transport 下不稳定（Cox 与 AFT折间波动明显）。

### 6.2 我们判断的原因
- **根因是跨事件 domain shift**：事件类型、地理环境、恢复机制差异过大。
- 事件级常数特征容易成为“事件替代变量”，对外推并不稳健。
- 六事件样本下，复杂层级模型（MixedLM/Cox）在某些折本身就数值脆弱。

### 6.3 已采取与建议的解决方向
已做：
- 全局类别空间对齐（避免 unseen category 报错）
- MixedLM 回退公式以保证可运行
- Balanced stop rule 早停

下一轮建议（按优先级）：
1. 事件分组重采样 / GroupDRO（优先应对 high-shift 事件）
2. 明确“事件级常数特征”仅作为分层变量，不作为主效应
3. Survival 以 AFT 作为外推主口径，Cox 保留解释性对照
4. 空间相关修正（spatial cluster SE / Moran’s I）降低乐观显著性
5. 若目标是强泛化，需要增加事件数量而非仅调参

---

## 7) 当前项目状态（结论）
- **事件内解释模型**：已达到高可复核水平（Strict V2 最成熟）。
- **跨事件泛化模型**：目前为“部分有效”，仍不足以给出强外推结论。
- **工程流程**：已具备规范化迭代、审计、停止机制，后续可持续演进。

---

## 8) 主要产物索引
- 迭代总结：`project/modeling_report/05_iteration_summary.md`
- 特征升级：`project/modeling_report/06_feature_upgrade_report.md`
- LOEO验证：`project/modeling_report/07_logo_validation_report.md`
- 跨事件V3：`project/modeling_report/08_cross_event_model_report.md`
- 稳定化收尾：`project/modeling_report/09_cross_event_stabilization_report.md`

关键输出：
- `project/modeling/output/model_summary_for_report.csv`
- `project/modeling/output/model_summary_feature_upgrade_v2_strict.csv`
- `project/modeling/output/model_summary_cross_event_v3.csv`
- `project/modeling/output/cross_event_round_comparison_v3x.csv`
- `project/modeling/output/cross_event_stop_decision_v3x.json`
