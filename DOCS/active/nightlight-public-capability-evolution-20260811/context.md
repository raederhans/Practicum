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

## Live process ownership

| Process | Owner | Log path | State |
| --- | --- | --- | --- |
| Public CI-equivalent validate/build/preview/browser gate | 当前执行线程 `019fef3b-e7bc-75b2-b678-9b04987de081` | 待启动前写入 task-owned log path | Not started；唯一 owner，其他 agent 不得启动/轮询/停止 |

## Handoff

- 本线是 `project/nightlight-public/**` 的唯一 implementation owner。
- Integration owner 可在最终交付后按 commit 顺序集成；本线不得自行修改 main、推送或部署。
- Modeling/Data lane 提供未来 source/metric semantics，本线 Phase 2 不依赖其未交付代码，只保留 contract integration seam。

## Next step

创建 Stage 1 Lore commit，然后实施 Stage 2 freshness/version/error pure contract 与 bundled adapter seam。
