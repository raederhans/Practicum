# Plan

## Goal

从基线 `ca8292040a402eae1d2e461708a4cc912867efcb` 出发，把 full-upstream 的七个来源 blocker 尽可能转换为小型、可审计、可重复验证的来源 receipt；无法安全完成的分支必须收窄为精确阻断和唯一最小补证动作。

## Scope

- NASA VNP46A2 与 NLCD：官方 Earth Engine 标识、无泄密认证预检、受控导出计划和 receipt 接口。
- OSM：完整分析事件范围的不可变 snapshot 方案、查询与 checksum receipt。
- TIGER/Line：2020 ZCTA/County 官方 archive 的隔离下载、size/SHA-256 验证与本地 cache receipt。
- Miami-Dade：从仓库历史线索和官方 ArcGIS 元数据确定准确 item/service，或保留精确选择阻断。
- WorldPop：锁定 TUR/BHS 2020 的精确 variant、immutable URL、license、size 与 checksum。
- EAGLE-I：核对公开上游版本、52 个 tracked derivatives 的结构与可证明变换线索；不猜测缺失 lineage。
- 仅写入用户授权的 acquisition、manifest、reproducibility、测试和本任务记录路径。

## Sources of truth

- 用户委托中的安全、权限、来源和 ready-for-integration 交付要求。
- `project/data/manifests/reproducibility_inputs_v1.json` 与 `source_manifest_v1.json`。
- 当前 checkout 的来源脚本、事件配置、Git 历史和 52 个 tracked EAGLE-I derivatives。
- NASA、USGS、Census、OSM/Geofabrik、Miami-Dade ArcGIS、WorldPop、DOE/ORNL/Figshare、Google Earth Engine 的官方或一手页面/API。

## Stages

- [x] Stage 1: 确认基线、工作树状态、适用规则和既有七 blocker。
- [x] Stage 2: 完成无泄密 Earth Engine/Google Cloud 认证预检与官方来源研究。
- [x] Stage 3: 在隔离 cache/output 中执行有上限的公开下载或 snapshot，并生成 receipt；OSM 按 429 停止且未伪造 receipt。
- [x] Stage 4: 实现默认 dry-run、fail-closed、可恢复的 acquisition/receipt 接口与测试。
- [x] Stage 5: 新增七边界 acquisition manifest，并保持 full-upstream fail closed。
- [x] Stage 6: 运行最小充分验证、review 和第一性原理复核，整理 ready-for-integration 包。

## Acceptance criteria

- 七个 blocker 均有“已证事实 / 缺口 / 最小补证路径”或真实 verified receipt。
- 任何下载前记录 license、immutable identifier、空间/时间范围、预计大小、配额/费用风险和停止条件。
- Earth Engine 仅报告认证“存在/可用/需交互”，不泄露 token、私钥、账号或 credential 内容。
- raw/cache/output 位于任务专属 ignored cache 或 OS 临时目录，不覆盖现有数据，不进入 Git diff。
- 新脚本默认只 preflight/dry-run；执行必须显式 opt-in 和显式 output directory；异常时不留下伪 verified receipt。
- `reviewed-modeling` 仍通过；`full-upstream` 只有全部必要输入真实通过才允许 ready，否则保持 blocked 并列出剩余 blocker。
- 最终交付包含确切文件、diff、命令结果、认证状态、资产 checksum/license/size/位置、风险和临时产物状态。

## Non-goals

- 不运行完整科学模型，不宣称新的科学验证或 full-upstream ready。
- 不修改云端计费、配额、IAM、项目选择或服务开通状态。
- 不提交 credential、私有资产、受限数据、大体积 raw/derived 文件。
- 不触碰公开站点、dashboard、CI workflow、registry、其他 worktree 或用户 personal-project WIP。
- 不执行 `git add/commit/push/merge/rebase/cherry-pick/worktree` 等 Git 状态变更。

## Risks and constraints

- Earth Engine 可能需要浏览器登录、选择 GCP project、接受条款或开通计费；这些是硬交互闸门。
- OSM live API 可变且有公平使用限制；优先不可变官方/Geofabrik snapshot，任何 live 请求必须有小范围和停止条件。
- TIGER/Line 两个 archive 合计约 609 MB；必须隔离下载并验证既有 size/SHA-256。
- EAGLE-I 官方 2014–2023 原始文件合计约 9 GB，不能为本任务无上限下载；局部 lineage 证明优先于全量取回。
- WorldPop 许可可能随具体 dataset/输入派生而变化，不能用站点一般说明替代具体文件 receipt。
