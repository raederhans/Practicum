# Plan

## Goal

在精确基线 `6b3de4ee97c5391084538bec84db3b1a1f4e05ed` 上，为 Nightlight Public App 依次交付三个可独立回滚的能力边界：本地且明确 opt-in 的研究 analytics、版本化 freshness/error contract spike、Atlas route/mounted behavior seam。交付必须保持 Public static/local-assets-only/aggregate-only 边界、科学声明边界和 fail-closed missingness 语义，并形成 committed ready-for-integration 候选。

## Scope

- `project/nightlight-public/**` 的 Public runtime、domain/lib、Atlas view、必要的 App/route/focus integration、Public tests、政策/安全说明、source/dist verifier、release manifest 与 exact allowlist。
- 本目录内唯一一组任务留档。
- 用户委托中的 Phase 1 `NL-A01`、Phase 2 `NL-F01`、Phase 3 `NL-ATLAS-01/02`，严格串行实施。

## Sources of truth

- 用户委托与验收标准，来源任务 `019febfc-6511-7051-89c9-60970113a4ea`。
- 数据产品与科学指标探索最终报告 `019fef1c-15fd-7ff0-a1c6-92e907b9c496`。
- 技术架构与性能探索最终报告 `019fef1c-349e-7ee2-992e-405a8ce4e562`。
- 当前代码、测试、verifier、release manifest 与 fresh validation output；实际代码库优先于历史记录。

## Stages

- [x] Stage 1 — `NL-A01` local/explicit opt-in analytics contract。
  - 预定义研究问题、allowlisted event/property schema、consent version/state。
  - 默认关闭、opt-in 前不记录、不发送网络；session-only storage，无 cookie/跨会话 identifier。
  - 本地查看、JSON export、一键 clear；禁止 free text、model inputs、精确位置和 restricted fields。
  - DATA_POLICY、SECURITY、UI copy 与 negative tests 同步。
- [x] Stage 2 — `NL-F01` runtime freshness/version/error contract spike。
  - `source/version/effective/retrieved/validated` 与 `value/null/reasonCode` 的版本化契约。
  - 区分 stale/offline/rate_limited/auth_required/source_failure/validation_failure；失败不变零。
  - 只建立 pure adapter seam、ADR/trigger 和负向测试；不启用 external fetch、不改 CSP、不建 API/server/database。
- [x] Stage 3 — `NL-ATLAS-01/02` route-state and mounted behavior seams。
  - 先提取 pure route-state helper 并锁定 hydrate/canonicalize/push/replace/back-forward/cross-seed/selection/focus/schema failure。
  - 行为稳定后再按停止条件评估 `ComparisonPanel`、`EvidencePassport`；无明确 parent responsibility 减少则保留 helper-only。
- [ ] Stage 4 — 单 owner 的完整验证与交付审计。
  - CI 等价 `VITE_BASE_PATH=/Practicum/ npm run validate`、source/dist verifier、manifest/CSP/network negative tests。
  - Atlas route/query/focus 与 320/768/reflow/forced-colors browser gate。
  - analytics default-off/opt-in/clear/export/local-only 与 zero runtime network。
  - `git diff --check`、secret/large/generated-file 审查、commit 范围核验。

## Acceptance criteria

- 三阶段严格顺序完成，每阶段通过与职责匹配的测试并形成可独立回滚的 Lore commit。
- Analytics 无第三方 SDK、server send、cookie、persistent cross-session identifier、任意 custom payload 或自由文本通道。
- Freshness contract 对 unavailable/failure/stale 显式建模，合法零只允许在 `available` 值状态；最大 stale age 和展示条件明确。
- Public CSP 保持 `connect-src 'none'`，runtime network verifier 与 browser evidence 均为零请求能力。
- Atlas URL/focus/selection 行为有真实运行时测试；no-score/no-rank/no-outcome 与 R²/AUC/Passport 语义不变。
- 所有新 source/test/config path 逐项进入 exact allowlist，不采用 wildcard。
- release manifest 验证实际 path/bytes/hash/base/build contract，不依赖固定 11 文件断言。
- 完整 Public validate 与 browser gate 有 fresh evidence；未运行或环境受限的 gate 明确报告。

## Non-goals

- 不修改 modeling/data/dashboard，不实施新科学指标，不把当前 R²/AUC/Passport 改名或重新解释。
- 不接入第三方 analytics，不发送服务端，不启用真实 external fetch，不创建 API/server/database。
- 不放宽 CSP `connect-src`、runtime dependency boundary 或 source exact allowlist。
- 不修改 main，不 merge/rebase/cherry-pick/push/deploy，不改变 worktree topology，不修改 `_worktree_registry.md`。
- 不读取或输出 credential，不读取主工作区未跟踪研究目录。

## Risks and constraints

- 当前 Public verifier 会扫描所有 runtime source 的 network-shaped token；本地 export 必须使用无网络 API 且保持 scanner 可判定。
- `AtlasView.vue` 目前是 router/query/focus/state 唯一 owner；拆分不得产生第二 owner。
- 新模块和测试会改变 exact source allowlist，构建可能改变 dist 文件数量；验证以 manifest 内容为准。
- 浏览器测试可能需要新增测试依赖；优先使用现有 Vitest + browser-native harness，确需依赖时必须单独准入且不得加入 analytics SDK。
- 并行 Modeling/Data 与 Dashboard 任务存在；本 worktree 只拥有 Public App，不吸收或回退其他线的改动。
