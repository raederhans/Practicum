# Task

## Current status

`IN_PROGRESS` — Stage 1 analytics 已实现并完成 targeted validation；正在准备独立 Lore commit，Stage 2 尚未开始。

## Checklist

- [x] 核对 `HEAD` 精确等于 `6b3de4ee97c5391084538bec84db3b1a1f4e05ed` 且 worktree 干净。
- [x] 读取适用指令和 `$manage-task-records`、`$orchestrate-live-tests`、`$write-lore-commits`。
- [x] 完整读取数据产品/科学指标和技术架构/性能探索任务最终报告。
- [x] 建立唯一 `plan.md`、`context.md`、`task.md`。
- [x] Stage 1 analytics contract / UI / policy / tests。
- [ ] Stage 1 targeted validation and Lore commit（targeted validation 已完成，commit 待创建）。
- [ ] Stage 2 freshness/error contract / adapter seam / ADR / tests。
- [ ] Stage 2 targeted validation and Lore commit。
- [ ] Stage 3 Atlas route-state/mounted seam。
- [ ] 评估并记录 helper-only 或 conditional component split 的停止决定。
- [ ] Stage 3 targeted validation and Lore commit(s)。
- [ ] 固化 live-process owner、命令、端口、日志、输出与 stop conditions。
- [ ] 运行 CI 等价 full validate、source/dist verifier、manifest/CSP/network negative tests。
- [ ] 运行 Atlas 320/768/reflow/forced-colors browser gate 与 analytics zero-network gate。
- [ ] 完成 diff/status/secret/large/generated-file 审查并更新交付记录。

## Validation evidence

| Command or check | Result |
| --- | --- |
| `git rev-parse HEAD` | PASS — `6b3de4ee97c5391084538bec84db3b1a1f4e05ed` |
| `git status --short` | PASS — 初始状态无输出 |
| 两份探索任务 final report | PASS — 从对应任务的完整 final response 读取 |
| `npm ci` | PASS — 锁文件安装 73 packages，audit 0 vulnerabilities；仅有 allow-scripts 配置提示 |
| `npm test -- tests/local-analytics.test.js tests/static-shell.test.js tests/routes.test.js tests/release-manifest.test.js tests/platform-boundary.test.js tests/public-boundary.test.js` | PASS — 6 files，84 tests |
| `npm run verify:public` | PASS — `Public source boundary verified.` |

## Open risks and remaining work

- Analytics UI 已放在 global shell 与 footer 之间的默认折叠 disclosure；Credits/Policy 同步提供可扫描的 opt-in、tab-only、zero-transport 说明。
- Mounted Atlas behavior test 需要核对现有依赖能否真实 mount Vue/router；若现有工具不足，优先建立 browser-native harness，新增依赖仅在确实必要时准入。
- 完整 validate 和 browser gate 尚未运行；历史 `180/180` 与 Pages 证据不能代替本任务 fresh validation。
