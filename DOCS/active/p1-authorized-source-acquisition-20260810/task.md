# Task

## Current status

`ready-for-integration` — 小型 acquisition/receipt 实现、七边界审计 manifest、测试和任务证据已完成；未声称 full-upstream ready。

## Checklist

- [x] 确认指定 base SHA、detached 状态和干净工作树。
- [x] 读取适用规则、registry、现有 reproducibility/source manifests、验证器和相关测试。
- [x] 建立单一 `plan.md` / `context.md` / `task.md` 留档组。
- [x] 无泄密检查 official Earth Engine Python API/CLI、Google Cloud ADC 和相关环境变量是否存在/可用。
- [x] 为七个 blocker 收集官方 source/license/identifier/范围/size/风险证据。
- [x] 在受控 ignored cache 中完成允许的下载；OSM snapshot 按 429 停止并清理 partial。
- [x] 实现和测试默认 dry-run/fail-closed acquisition/receipt 接口。
- [x] 新增七边界审计 manifest，并保持未满足边界 fail closed。
- [x] 运行 targeted tests、reproducibility scopes、静态 review 和 Git diff 检查。
- [x] 形成 ready-for-integration 交付包并记录临时产物清理状态。

## Validation evidence

| Command or check | Result |
| --- | --- |
| `git rev-parse HEAD` | `ca8292040a402eae1d2e461708a4cc912867efcb` |
| `git status --short` | 开始时为空 |
| `rg --files -g AGENTS.md -g lessons learned.md` | 仓库内无匹配文件 |
| Manifest/source/test 静态检查 | 七 blocker 与有限 reviewed-modeling claim 均存在 |
| `py -m pytest project/tests/test_authorized_source_acquisition.py project/tests/test_source_acquisitions.py project/tests/test_reproducibility_inputs.py -q` | `30 passed in 7.44s`（最终 fresh） |
| Python compile + 四份 JSON parse | `compile-and-json-ok` |
| `py project/modeling/reproducibility.py --scope reviewed-modeling` | `status=ready`, 16 receipts，保留三项有限 claim |
| `py project/modeling/reproducibility.py --scope full-upstream` | 预期 `status=blocked`, exit 1；七边界已更新为精确 auth/project/date、lineage、HTTP 429、owner-cache、历史服务 access blocker |
| TIGER/Line downloads | 2/2 size 与 SHA-256 verified，合计 608,640,344 bytes |
| WorldPop downloads | 2/2 exact variant/size 与 SHA-256 verified，合计 549,074,631 bytes |
| OSM bounded snapshot | HTTP 429 后停止；0 verified receipt，0 partial retained |

## Open risks and remaining work

- Earth Engine 本地认证/Cloud project 不存在或不可用；VNP46A2 还缺 cloud-screening date receipt，NLCD 还缺 export task/output receipt。
- OSM 仍缺 26/26 immutable snapshot；当前官方 Overpass endpoint 返回 429，不应无界重试。
- TIGER/WorldPop 只在此 worktree 的 ignored task cache 真实存在，尚未进入 clean-checkout reproducibility contract。
- Miami-Dade 精确历史 item/service 已找到，但匿名访问受限；当前公开两年 table 不等价。
- EAGLE-I 的 parent-to-derivative deterministic lineage 尚未证明，且约 9 GB 官方 parent 不应在缺 transform 证据时全量下载。
