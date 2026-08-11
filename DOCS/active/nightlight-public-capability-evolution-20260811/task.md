# Task

## Current status

`IN_PROGRESS` — Stage 1 与 Stage 2 已提交；Stage 3 route-state/mounted seam 已实现并通过 targeted validation，准备 helper-only 独立提交。

## Checklist

- [x] 核对 `HEAD` 精确等于 `6b3de4ee97c5391084538bec84db3b1a1f4e05ed` 且 worktree 干净。
- [x] 读取适用指令和 `$manage-task-records`、`$orchestrate-live-tests`、`$write-lore-commits`。
- [x] 完整读取数据产品/科学指标和技术架构/性能探索任务最终报告。
- [x] 建立唯一 `plan.md`、`context.md`、`task.md`。
- [x] Stage 1 analytics contract / UI / policy / tests。
- [x] Stage 1 targeted validation and Lore commit — `2560fb0bf6e5b03a5ed8006152d094eaea6b1c5b`。
- [x] Stage 2 freshness/error contract / adapter seam / ADR / tests。
- [x] Stage 2 targeted validation and Lore commit — `adb2cc3da501945a25a20acefb85796d8a0bc280`。
- [x] Stage 3 Atlas route-state/mounted seam。
- [x] 评估并记录 helper-only 或 conditional component split 的停止决定 — 停止于 helper-only。
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
| `npm test -- tests/source-freshness-contract.test.js tests/error-contract.test.js tests/platform-boundary.test.js tests/public-boundary.test.js` | PASS — 4 files，91 tests（最终 targeted run） |
| `npm view @vue/test-utils ...` / `npm view happy-dom ...` | PASS — exact `2.4.11` / `20.11.2`，均 MIT；Happy DOM 要求 Node >=20 |
| `npm install --package-lock-only --ignore-scripts` + `npm ci` | PASS — 137 packages，audit 0；存在 transitive `glob@10.5.0` deprecation warning，未发现 audit vulnerability |
| `npm test -- tests/atlas-route-state.test.js tests/atlas-mounted.test.js tests/atlas-schema-mounted.test.js tests/compare-events.test.js tests/static-shell.test.js tests/routes.test.js tests/platform-boundary.test.js tests/public-boundary.test.js` | PASS — 8 files，114 tests |
| `npm run verify:public` after Atlas seam | PASS — exact source/dependency boundary verified |

## Open risks and remaining work

- Analytics UI 已放在 global shell 与 footer 之间的默认折叠 disclosure；Credits/Policy 同步提供可扫描的 opt-in、tab-only、zero-transport 说明。
- Freshness contract 目前没有 production consumer；这是刻意的 contract spike，不代表 runtime data 已启用或有 live freshness 保证。
- Mounted Atlas tests 使用 dev-only `@vue/test-utils` 与 `happy-dom`；二者不会进入 runtime bundle，仍需最终 build/manifest gate确认。
- 完整 validate 和 browser gate 尚未运行；历史 `180/180` 与 Pages 证据不能代替本任务 fresh validation。
