# Context

## Current truth

- Worktree: `C:\Users\raede\.codex\worktrees\c631\Practicum`。
- 准入基线：`HEAD = 6b3de4ee97c5391084538bec84db3b1a1f4e05ed`，2026-08-11 首次核对时 worktree 干净。
- 仓库内未发现更具体的 `AGENTS.md`；采用委托中提供的顶层约束。
- 已完整读取两份探索任务最终报告。共同结论：Public 保持 static/local-only；analytics 必须是预定义问题驱动且明确 opt-in；freshness 优先静态或 build-time，当前阶段仅建契约；Atlas 先 mounted seam 后评估中等模块化。
- 当前 Public verifier 使用逐文件 exact allowlist，拒绝 runtime fetch/XHR/WebSocket/EventSource/sendBeacon 等网络能力，并要求 CSP `connect-src 'none'`。
- 当前 release manifest 已验证实际文件的 path/bytes/SHA-256/base/build contract；不需要固定文件数量。

## Decisions and deviations

| Time | Evidence or decision | Impact |
| --- | --- | --- |
| 2026-08-11 | 基线精确匹配用户指定 SHA，worktree 无未归属改动 | 允许写入；若后续出现边界外改动则停止并审计 owner |
| 2026-08-11 | Analytics MVP 采用 session-only local research log，默认关闭、明确 opt-in、本地 view/export/clear | 保持 server/network/cookie/cross-session identifier 全部缺席 |
| 2026-08-11 | Freshness 仅实现 pure versioned contract + bundled adapter seam | CSP、release contract 与 runtime external request 保持不变 |
| 2026-08-11 | Atlas 先抽 pure route-state helper并补 mounted behavior harness | 组件拆分取决于 parent responsibility 与测试 seam 的实际改善 |
| 2026-08-11 | Stage 1 使用 fixed methods `recordSurfaceViewed` / `recordAtlasModeSelected`，内部 schema 严格拒绝 unknown event/extra property | 不提供任意 custom payload API；研究事件只对应两项已声明问题 |
| 2026-08-11 | consent 与 event log 只在当前 tab session storage；无存储时退回当前页面内存 | 默认关闭；版本漂移/损坏 envelope 删除并回到 not_granted |
| 2026-08-11 | Stage 1 commit `2560fb0bf6e5b03a5ed8006152d094eaea6b1c5b` | Analytics 契约可独立回滚；完整 browser/full validate 仍保留到最终 gate |
| 2026-08-11 | Freshness source envelope 与 aggregate value schema 分离 | source/transport failure 不改变既有科学值状态或 R²/AUC/Passport 语义 |
| 2026-08-11 | stale display 使用 effective date 与显式 evaluation time，最大 30 天 | day 31、validation failure、缺 metadata 或未知原因一律 fail closed |
| 2026-08-11 | Phase 2 targeted suite 4 files / 91 tests passed；source verifier 在该阶段实现后通过 | 证明 pure contract、adapter fail-closed 与 exact source boundary；未证明 live source 或 API |
| 2026-08-11 | Stage 2 commit `adb2cc3da501945a25a20acefb85796d8a0bc280` | Freshness contract 可独立回滚；未启用任何 runtime source |
| 2026-08-11 | 为真实 mounted Vue/router/DOM tests 准入 dev-only `@vue/test-utils@2.4.11` 与 `happy-dom@20.11.2` | exact versions、MIT notices、Node >=20 和 verifier dependency allowlist 同步；runtime dependencies 不变 |
| 2026-08-11 | `AtlasView.vue` 从探索报告的 840 行降到 788 行；route parse/hydrate/serialize/match 由 pure codec 单独负责 | URL owner 仍只有 Atlas parent；route-state 可独立 unit test，mounted behavior 覆盖 push/replace/history/cross-seed/focus/schema failure |
| 2026-08-11 | Conditional split 评估为 helper-only STOP | 788 行低于探索建议的约 1,000 行门槛；无重复 router owner、持续 merge conflicts 或并行 feature-owner 证据；拆 `ComparisonPanel`/`EvidencePassport` 当前不会再改善 seam，故不扩大 template/CSS/allowlist diff |
| 2026-08-11 | Stage 3 commit `260f92d706ce63ff2dbc23950ab2eb599ebf4215` | Atlas route-state/mounted behavior seam 可独立回滚；未拆分展示组件，未改变科学指标 |
| 2026-08-11 | CI base-path validation 揭示 manifest unit test 隐式继承 `VITE_BASE_PATH` | 测试 fixture 显式传入 `/`，使 unit contract 与环境隔离；另有 `/Practicum/` 专项 fixture 验证 repository base |
| 2026-08-11 | 首次 preview 未继承 `/Practicum/`，入口 200 但 JS asset 404 | 该 browser session 全部作废并关闭；只停止 PID `599880`，随后在同一构建 base 下重启并验证入口和 JS asset 均为 200 |

## Live process ownership

| Process | Owner | Log path | State |
| --- | --- | --- | --- |
| Public CI-equivalent validate/build | 当前执行线程 `019fef3b-e7bc-75b2-b678-9b04987de081` | `C:\Users\raede\AppData\Local\Temp\nightlight-public-c631-20260811\validate.stdout.log` 与 `validate.stderr.log` | Complete；PID `612336` exit 0，output=`dist` |
| Public preview/browser gate | 当前执行线程 `019fef3b-e7bc-75b2-b678-9b04987de081` | 同一临时目录中的 `preview.*.log` 与 `.playwright-cli` artifacts | Stopped；accepted preview PID `133248` 已停止，browser session 已关闭，`41783` 已释放，无残留 browser |

Success：validate exit 0、source/dist verifier 与 manifest contract 通过；preview listener PID 与记录一致；browser route/query/focus、320/768/reflow/forced-colors、analytics opt-in/clear/export/zero-network 通过。Failure：同一确定性失败复现后停止重试并保留日志。Stop：关闭 Playwright session、停止且仅停止已记录 PID、确认 `41783` 无 listener；其他 agent 只读日志，不启动、轮询或停止该 lane。

## Handoff

- 本线是 `project/nightlight-public/**` 的唯一 implementation owner。
- Integration owner 可在最终交付后按 commit 顺序集成；本线不得自行修改 main、推送或部署。
- Modeling/Data lane 提供未来 source/metric semantics，本线 Phase 2 不依赖其未交付代码，只保留 contract integration seam。

## Next step

Integration owner 按 `2560fb0b` → `adb2cc3d` → `260f92d7` → 最终 verification record 的顺序整合；Modeling/Data lane 未来只有在 source schema 和准入条件满足时才接入 freshness adapter seam。
