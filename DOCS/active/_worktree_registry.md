# Worktree Registry

| Worktree / path | Task | Base branch / commit | Current branch / HEAD | Goal | State | Hotspots | Tests | Overlap | Order | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `C:\Users\raede\Desktop\essay help master\Practicum` | Next evidence phase integration owner | `main` / `223fc653dba2768dad99df9d032beaedd9234d6a` | `main` / `c72e26e`, ahead of `origin/main` pending release | Supervise original P1/P2/P3 tasks, review deliveries, integrate, validate, push, and clean | `ready-for-review` | Shared task records, integration, release and claim boundaries | Dashboard 21/21 + bundle; public 123/123 + build/boundary/browser; Python 112 + 7 subtests; reviewed/full preflights verified | User research WIP remains untracked and excluded | 0 | Commit lane archives and release record, push, verify Pages, then clean and archive umbrella |
| `C:\Users\raede\.codex\worktrees\3fa1\Practicum` | Original P1 `019fe697-b3b0-7782-a3e8-f6fb37853c26` | detached `223fc65` | candidate `5ede841`, integrated as `a410ae2` | Resolve or classify full-upstream provenance, licensing, receipt, and access blockers without history rewrite | `integrated` | `project/data/manifests`, modeling entrypoints, direct tests | 26 + 7 subtests; reviewed ready; full-upstream blocked with seven gaps | No P2/P3 paths; no shared coordination edits | 3 | Remove registered worktree after release and archive original task |
| `C:\Users\raede\.codex\worktrees\3ab1\Practicum` | Original P2 `019fe697-c99b-7273-b3a6-18e9e58c6a13` | detached `223fc65` | candidate `223631a`, integrated as `232577e` | Produce authoritative research-backed proxy evaluation and an independent solo-feasible next plan | `integrated` | `project/nightlight-public/**` | 123/123 + build/boundary; main browser focus/320px/console/network checks passed | No P1/P3 paths; no shared coordination edits | 2 | Lane record archived; remove registered worktree after release and archive original task |
| `C:\Users\raede\.codex\worktrees\3a29\Practicum` | Original P3 `019fe6b2-49d7-7511-8a7b-da806bae3020` | detached `223fc65` | candidate `90dea17`, integrated as `7db66b9` | Measure production-like navigation timing and decide whether architecture work is evidence-backed | `integrated` | `project/nightlight-dashboard/**`; performance live processes released | 21/21 + bundle analysis; 56 selected samples rechecked | No P1/P2 paths; no shared coordination edits | 1 | Lane record archived; remove registered worktree after release and archive original task |

States: `in-progress`, `blocked`, `ready-for-review`, `ready-for-integration`, `integrated`, `abandoned`.

## Delivery package

- Summary: P3 bundle guardrail, P2 route accessibility plus blank user-study instrument, and P1 fail-closed reproducibility receipts integrated sequentially and published.
- Files: 32 changed release paths across task records, dashboard tooling, public UX/tests, and modeling provenance/tests; user research WIP remained excluded.
- Diff from base: five normal commits from `2d81121` through release-ready `2a47f36`; no history rewrite or force-push.
- Commit and branch state: only local `main` is registered; normal pushes only; no teammate-remote write.
- Divergence from main: none after release push; the docs-only closeout commit follows without retriggering Pages.
- Overlap and conflict risk: lane product paths were disjoint; shared state and live ports were supervised by the primary owner.
- Validation evidence: dashboard 20/20 plus bundle analysis; public 119/119 plus build, boundary, local/live browser; Python 106 plus 7 subtests and explicit blocker exit codes.
- Publication evidence: run `31317312185` succeeded at `2a47f36`; the live 11-file manifest matched byte length and SHA-256; Overview and Atlas live focus/title/overflow/console checks passed.
- Unverified risks: no full raw-input scientific reproduction; no new independent exact-SHA sign-off; no actual user study or manual assistive-technology matrix; MapLibre remains 803,051 B; Actions emit a Node 20 deprecation annotation while GitHub forces Node 24.
- Recommended integration method: future work starts from `main` and uses one bounded feature worktree with explicit ownership and closeout gates.

## Cleanup evidence

- Removed Git worktrees: P1 `3fa1`, P2 `3ab1`, canonical P3 `3a29`, and duplicate P3 `b19a` registrations; P1/P2 directory bodies were deleted.
- Archived Codex tasks: P1, P2, canonical P3, and duplicate P3.
- Generated artifacts: main Playwright sessions closed; owned ports 43217/55473/43189 released; exact Playwright snapshots and task logs deleted. One empty main temp directory shell remains because policy rejected directory removal.
- Recovery: lane commits remain reachable through `main` as `82d4f97`, `e7e95be`, and `1d2a0b1`; detached candidates also remain reachable through their integrated descendants until normal Git pruning.
- Non-Git residue: `3a29/Practicum` and `b19a/Practicum` directory bodies remain after Git registration removal because Windows/policy denied recursive deletion. Neither is a registered worktree or valid integration source.
