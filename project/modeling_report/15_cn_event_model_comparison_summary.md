# 新增事件前后模型表现中文总结

## 1. 这份总结是干什么的
这份文件用于给组内同步目前建模工作的阶段性结论，重点回答三个问题：

1. 原来的 6 事件模型表现如何。
2. 直接把新事件加进训练后，模型发生了什么变化。
3. 在修补国际事件人口协变量并重构更瘦的 HZ2 规格后，模型有没有回升，以及这说明了什么。

这里主要比较三条线：

- `strict-v2`：解释型主线，偏事件内/同分布分析。
- `hazard-mainline (HZ1)`：跨事件 transport 主线，偏 LOEO 泛化。
- `quality-matched`：局地 matched design，偏解释增强。

另外，最后还比较了新的独立实验线：

- `HZ2`：只针对 Stage 9/10 国际事件修补后重跑的更瘦 transport 规格。

---

## 2. 6 事件基线模型表现

### 2.1 strict-v2
6 事件基线下，解释型模型整体比较稳定：

- OLS: `coef(in_buffer) = 0.0254`, `p = 0.0975`
- MixedLM: `coef(in_buffer) = 0.0254`, `p = 0.0149`
- Logit: `OR(in_buffer) = 0.7823`
- Logit AUC: `0.7302`
- Cox: `HR(in_buffer) = 1.3274`

解释：

- `in_buffer` 的方向稳定，没有翻转。
- MixedLM 是这套基线里最稳的解释模型，显著性最好。
- Logit 显示 buffer 内 damage odds 更低，说明关键设施周边更有韧性的信号是存在的。

### 2.2 hazard-mainline (HZ1)
跨事件预测主线在 6 事件时的表现是当前所有版本里最好的：

- Logit AUC: `0.6001`
- Logit Brier: `0.4406`
- Cox c-index: `0.5341`
- AFT c-index: `0.4640`
- survival best: `0.5341`

解释：

- 这说明 6 事件集合本身是“可学”的。
- 但这个表现不能简单理解为模型已经具备很强的泛化能力，因为 6 个事件的结构异质性还不够大。

### 2.3 quality-matched
6 事件 matched 设计下：

- Matched OLS: `coef(in_buffer) = 0.0213`, `p = 0.0908`
- Matched Logit: `OR(in_buffer) = 0.7308`, `p = 0.0074`

解释：

- 即使在局地可比对照下，buffer 内仍然表现为更低的 damage odds。
- 这支持“关键设施周边更 resilient”这条论文叙事。

---

## 3. 直接增加 4 个新事件后发生了什么

新增顺序为：

1. `ian_fortmyers`
2. `ian_charlotteharbor`
3. `earthquake_hatay`
4. `dorian_freeport`

最终事件数从 6 个增到 10 个。

### 3.1 跨事件预测主线明显变差

`hazard-mainline` 的 Logit AUC 变化如下：

- 6 事件 baseline: `0.6001`
- Stage 7 (`ian_fortmyers`): `0.4914`
- Stage 8 (`ian_charlotteharbor`): `0.4856`
- Stage 9 (`earthquake_hatay`): `0.4726`
- Stage 10 (`dorian_freeport`): `0.4762`

`survival best` 变化如下：

- baseline: `0.5341`
- Stage 7: `0.5312`
- Stage 8: `0.5125`
- Stage 9: `0.5219`
- Stage 10: `0.5161`

结论：

- 直接加事件后，跨事件 transport 并没有提升，反而明显下降。
- 尤其是 damage 分类，从 `0.6001` 掉到 `0.47-0.49`，下降非常明显。

### 3.2 strict-v2 解释线没有一起崩

以 full 规格为例：

- Stage 9 MixedLM `coef(in_buffer) = 0.0167`, `p = 0.0488`
- Stage 10 MixedLM `coef(in_buffer) = 0.0215`, `p = 0.0094`
- Stage 10 strict-v2 Logit AUC: `0.7490`

结论：

- 加事件后，解释型模型整体没有失效。
- `in_buffer` 的方向始终没翻。
- MixedLM 仍然保持稳定，是当前最可信的解释模型之一。

### 3.3 matched 解释线方向保住了，但强度变弱

6 事件 baseline:

- Matched Logit OR: `0.7308`
- Matched OLS coef: `0.0213`

10 事件 final:

- Matched Logit OR: `0.8170`
- Matched OLS coef: `0.0078`

结论：

- `OR < 1` 仍然成立，所以“buffer 更 resilient”这条解释没有消失。
- 但 OR 更接近 1，说明效应被稀释了。
- Matched OLS 基本失去显著性，说明连续值层面的局地信号变弱。

---

## 4. 为什么直接加事件会掉分

这里最重要的结论是：

**问题不是“加事件”本身，而是新事件带来的结构异质性大于当前特征体系可以解释的范围。**

主要原因有四个。

### 4.1 事件异质性突然变大
新增事件同时引入了：

- 美国沿海飓风
- 非美国地震
- 非 Puerto Rico 的 island-like hurricane

这会同时扩大：

- `US vs non-US`
- `island vs non-island`
- `hurricane vs earthquake`
- `mid-urban vs low-urban`
- `观测质量差异`

旧的 HZ1 规格没有足够强的 covariate 去解释这些差异。

### 4.2 国际人口协变量当时是错误的
在旧 Stage 9/10 面板中：

- `earthquake_hatay`
- `dorian_freeport`

这两个事件的 `pop_density_per_km2` 都是同一个常数 `883.7286...`
但 `missing_pop_flag` 仍然等于 `1`。

这意味着：

- 它看起来“有值”
- 实际上依然是缺失
- 而且完全没有空间变化

这会直接污染国际事件的 transport 学习。

### 4.3 recovery 比 damage 更依赖观测结构
以 `earthquake_hatay` 为例：

- `post_tif_n = 8`

这意味着 recovery 面板很浅，删失结构更敏感。  
所以 survival 模型更容易受到新增事件观测深度差异的影响。

### 4.4 解释模型和 transport 模型本来就在吃不同的信息

- `strict-v2` 更像事件内解释模型，吃的是样本量和局地信号。
- `hazard-mainline` 更像跨事件预测模型，吃的是机制可迁移性。

所以加事件以后出现“解释线还稳、预测线掉分”的现象，是合理的，不是自相矛盾。

---

## 5. 修补国际人口协变量 + HZ2 之后的结果

这一轮没有重跑全部 10 事件，只针对：

- Stage 9 `earthquake_hatay`
- Stage 10 `dorian_freeport`

做了两件事：

1. 用 WorldPop 2020 栅格逐像素采样，替换原先错误的国际人口常数值
2. 用一个更瘦的 HZ2 transport 规格重跑

### 5.1 协变量修补本身是成功的

#### `earthquake_hatay`
- 旧问题：常数人口值 + `missing_pop_flag = 1`
- 新结果：
  - `v2_missing_pop_flag_mean = 0.0006`
  - `v2_unique_nonmissing = 1571`
  - 说明人口层基本修好

#### `dorian_freeport`
- 新结果：
  - `v2_missing_pop_flag_mean = 0.0587`
  - `v2_unique_nonmissing = 353`
  - 说明人口层大部分修好，但缺失比例略高于我们设的 `0.05` 门槛

### 5.2 HZ2 对 damage transport 有帮助

#### Stage 9
- HZ1 Logit AUC: `0.4726`
- HZ2 Logit AUC: `0.5048`
- 改善：`+0.0323`

#### Stage 10
- HZ1 Logit AUC: `0.4762`
- HZ2 Logit AUC: `0.4994`
- 改善：`+0.0232`

结论：

- 国际人口层修补 + 更瘦的 HZ2 规格，确实让 damage ranking 回升了。
- 这说明旧国际人口 covariate 是真实问题，不只是小噪声。

### 5.3 但 recovery transport 仍然没有改善

#### Stage 9
- survival best: `0.5219 -> 0.4995`

#### Stage 10
- survival best: `0.5161 -> 0.5000`

结论：

- recovery/survival 的主问题不只是人口层。
- 更可能仍然受以下因素影响：
  - post 观测深度
  - censoring
  - 事件恢复轨迹差异

### 5.4 HZ2 的代价：概率校准更差

#### Stage 9
- Brier: `0.3808 -> 0.4820`

#### Stage 10
- Brier: `0.3748 -> 0.5198`

结论：

- HZ2 更擅长“排序”
- 但不擅长输出稳定概率

换句话说：

- 如果把 HZ2 当 risk ranking model，它是改进的
- 如果把 HZ2 当 calibrated probability model，它不是改进的

---

## 6. 现阶段应该如何理解这些结果

### 6.1 直接加事件的效果

可以概括为：

- **预测上：变差**
- **解释上：大体稳定**
- **matched 解释上：方向保住，但强度变弱**

所以，不能简单说“新增事件失败了”，更准确的说法是：

**新增事件增加了覆盖，但在旧特征框架下把 transport 任务变难了。**

### 6.2 修补后再看新增国际事件

可以概括为：

- **damage transport：部分恢复**
- **recovery transport：仍未恢复**
- **国际人口层：数据质量问题已被确认并大部分修复**

所以修补是有价值的，但还不够把国际事件直接拉回主训练。

---

## 7. 目前最合理的事件使用建议

基于 `event_readiness_score_v1.csv` 和 `event_training_decision_v1.csv`，当前建议如下。

### 适合进入主训练的事件

- `ian_charlotteharbor`
- `earthquake_sanjuan`
- `ida_neworleans`
- `irma_miami`
- `laura_lakecharles`

这些事件的共同特点是：

- 观测质量更稳定
- covariate 更完整
- readiness score 更高

### 只建议做 sensitivity 的事件

- `ian_fortmyers`

它的数据质量本身没有问题，但从增量效果看，对 transport 仍然是负贡献或至少不明显正贡献。

### 需要先修再考虑是否纳入主训练的事件

- `dorian_freeport`
- `earthquake_hatay`
- `maria_sanjuan`
- `michael_panamacity`

其中：

- `earthquake_hatay` 的主要问题是 post 栈太浅，`post_tif_n = 8`
- `dorian_freeport` 的主要问题是国际人口层虽已修补，但 `missing_pop_flag_v2` 仍略高

---

## 8. 最终一句话结论

目前最稳妥的结论是：

**新增事件让模型覆盖更广，但在旧特征体系下显著拉低了跨事件泛化；在修补国际人口协变量并改用更瘦的 HZ2 规格后，Stage 9/10 的 damage 排序能力有明显回升，但 recovery/survival 仍然没有恢复。因此，当前更合理的策略不是把所有新事件直接并入主训练，而是优先保留 readiness 高、增量影响较稳的事件，把国际新增事件先作为 sensitivity 或 repair-first 对象。**

---

## 9. 对同学的直接建议

如果下一步继续推进，建议顺序是：

1. 不要继续盲目加事件。
2. 先把国际事件的 covariate 完整性和 recovery 观测深度问题继续补齐。
3. 主训练集优先保留 readiness 高且增量不明显伤害 transport 的事件。
4. 国际新增事件先保留在独立实验线或 sensitivity 线中，不直接覆盖主线结论。

