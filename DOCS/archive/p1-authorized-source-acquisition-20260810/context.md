# Context

## Current truth

- Worktree: `C:\Users\raede\.codex\worktrees\aefe\Practicum`。
- Git HEAD: `ca8292040a402eae1d2e461708a4cc912867efcb`，detached HEAD，开始时工作树干净。
- 仓库根目录没有实际 `AGENTS.md` 或 `lessons learned.md`；本任务附带的 `AGENTS.md` 是当前规则来源。
- 当前 `reviewed-modeling` 是有限的 committed-output consistency 检查；full-upstream 仍有七个 blocker，不能等同于科学复现。
- 本 agent 是本阶段 Earth Engine/API/download/raw cache/output 的唯一 live-process owner；所有 live process 已退出，当前无运行中下载或导出。
- 任务专属 ignored 根目录是 `cache/p1-authorized-source-acquisition-20260810/`；启动前已解析为当前 worktree 内的绝对路径，磁盘可用空间约 2.22 TB。
- 官方 `earthengine-api==1.7.38` 仅安装在任务专属虚拟环境；没有修改全局 Python，也没有启动认证浏览器或 Earth Engine 导出。
- 无泄密预检结果：全局 `earthengine`/`gcloud` 命令均不存在；相关 Cloud 环境值、常见 ADC 文件与常见 Earth Engine credential 文件均不存在。只检查存在性，没有读取内容。
- `reviewed-modeling` fresh 验证仍为 `ready`；`full-upstream` fresh 验证仍以七个既有边界为 `blocked`。
- 既有 `source_manifest_v1.json` 与 `reproducibility_inputs_v1.json` 已同步新证据：full-upstream 不再输出 Miami catalog-selection 或 WorldPop variant-selection 旧 blocker，但仍因 access/owner-cache 等精确边界 fail closed。

## Decisions and deviations

| Time | Evidence or decision | Impact |
| --- | --- | --- |
| 2026-08-10 | 基线 SHA 与用户指定值完全一致；`git status --short` 为空。 | 可以在当前 worktree 内开始只属于本任务的改动。 |
| 2026-08-10 | 既有 manifest 明确列出七个 blocker；TIGER receipts 已有历史 checksum，但当前 worktree 无 cache。 | 不重复发明 source catalog；以补齐本地可验证 receipt 和改进 fail-closed contract 为主。 |
| 2026-08-10 | EAGLE-I 52-file tracked-tree receipt 只证明当前 bytes，官方公开 release 与本地 derivatives 的 parent/transform 仍未建立。 | 公开 license 不能自动证明本地 52 文件可再生成。 |
| 2026-08-10 | 不调用 `$integrate-worktrees` 或 `$write-lore-commits`。 | 本任务只产出 unstaged diff 与 ready-for-integration 证据。 |
| 2026-08-10 | NASA VNP46A2 的官方 collection/DOI/band 已锁定；仓库缺少用于选择每日导出日期的 `project/script/data/*cloud_screening.csv`。 | 即使完成认证也不能安全枚举有界 daily export；不得运行 `stage3_export_all.py` 或声称 export receipt 已完成。 |
| 2026-08-10 | NLCD 的 CONUS 2021 与 Puerto Rico 2016 Earth Engine asset 已锁定，但没有可用认证、Cloud project、task ID、输出文件或 checksum。 | 来源身份已证，导出 receipt 仍为交互认证硬阻断。 |
| 2026-08-10 | TIGER/Line 两个 2020 national archive 下载成功，大小与既有 SHA-256 完全匹配。 | 本任务 cache 内可验证；由于 cache ignored，干净 checkout 仍需转移 cache 或重复 acquisition。 |
| 2026-08-10 | WorldPop 官方 REST catalog 精确选择 TUR id 6443 和 BHS id 6483，均为 2020 unconstrained 100m count、not UN-adjusted；下载后锁定 SHA-256。 | 消除了“first API file”歧义；URL 不是 content-addressed，因此本地 SHA-256 是字节不可变锚点。 |
| 2026-08-10 | OSM 26-event attic snapshot 第一响应成功，下一请求返回 HTTP 429；实现清理 `.partial` 且没有写 verified receipt。 | 保持 `http-429-no-complete-receipt`；不做无上限重试，也不把单事件响应当完整 snapshot。 |
| 2026-08-10 | Miami-Dade 历史 item `31cd319f45544648b59f0418aea60091` 与 `BuildingPermit_gdb/FeatureServer` 的 schema/point metadata 匹配历史 artifact；匿名访问被拒。 | 精确来源已找到；当前公开两年 table item `6db5f56e886446df88313ca279e59120` 不等价，仍需历史服务授权或 owner archive。 |
| 2026-08-10 | Figshare v4/ORNL 公开 EAGLE-I parent、CC BY 4.0 与 2014–2023 files 可证明；Git tracks 52 derivatives，但仓库只发现消费者，未发现生成这 52 filenames 的 transform/join code。 | 保持 `tracked-derived-lineage-unproven`，不得从结构相似推断可再生成。 |

## Live process ownership

| Process | Owner | Command / cwd / shared resources | Log path | Success / failure / stop | State |
| --- | --- | --- | --- | --- | --- |
| Earth Engine official API/CLI isolated install and auth preflight | Current agent | Commands: `py -m venv cache/p1-authorized-source-acquisition-20260810/earth-engine-venv`; venv `python -m pip install earthengine-api`; venv import/version and CLI help only; cwd: repository root; no port/database/shared cache; output: task venv | `cache/p1-authorized-source-acquisition-20260810/logs/earth-engine-preflight.log` | Success: `earthengine-api 1.7.38` and CLI help. Stopped before any browser login, project selection, terms, billing, quota, IAM, or credential-content access. Cleanup: retained task venv for reproducible handoff; integration owner may delete only this ignored venv after review | complete; no live process |
| TIGER/Line and WorldPop bounded downloads | Current agent | Serial exact-URL downloads; cwd: repository root; cache/output: `cache/p1-authorized-source-acquisition-20260810/downloads/{tiger-2020,worldpop-2020}/`; no port/database/shared cache | `cache/p1-authorized-source-acquisition-20260810/logs/public-downloads.log` | Success: 4/4 exact byte counts and SHA-256 receipts. No collision or partial remains. Cleanup: retained 1,157,714,975 bytes of verified assets plus small receipts for integration-owner transfer/review | complete; no live process |
| OSM 26-event attic snapshot | Current agent | `py project/data/acquisition/source_receipts.py osm-snapshot --events-manifest project/data/manifests/osm_modeled_event_scope_v1.json --as-of 2026-08-09T00:00:00Z --output-dir cache/p1-authorized-source-acquisition-20260810/osm-20260809 --pause-seconds 2 --execute`; cwd: repository root; endpoint `https://overpass-api.de/api/interpreter`; no port/database/shared cache | `cache/p1-authorized-source-acquisition-20260810/logs/osm-snapshot.log` | Failure signal: HTTP 429 on the next request after one completed response. Stop condition fired; all `.partial` files removed and no verified receipt written. Output directory is empty. Stop/cleanup complete by process exit | stopped fail-closed; no live process |

## Handoff

- Integration owner 只应整合 tracked 小文件；ignored cache 和任何 credential 均不属于交付 diff。
- 不要把某一来源的 receipt 当作 full-upstream 已完成；应以 `project/modeling/reproducibility.py --scope full-upstream` 的 fresh 结果为准。
- 建议先整合 acquisition script/tests 与两个新 manifest，再由明确 data owner 决定是否安全转移 ignored cache；不得把 cache 路径写成 clean-checkout receipt。

## Next step

交给 integration owner 审查 unstaged 小文件；Earth Engine 分支等待用户完成官方浏览器认证并选择已注册 project，Miami 分支等待历史 service access 或 owner archive，OSM 分支等待获批的低负载有界重跑窗口。
