# Worktree Registry

| Worktree / path | Task | Base branch / commit | Current branch / HEAD | Goal | State | Hotspots | Tests | Overlap | Order | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `C:\Users\raede\.codex\worktrees\f00e\Practicum` | Nightlight UI Phase A / `019fec40-8293-7eb3-b03c-224ddcd85e6f` | `codex/nightlight-ui-ux-base` / `962431b680845a91b0b5b96807c77630dc82dd89` | detached / `962431b680845a91b0b5b96807c77630dc82dd89` | Repair Shell, navigation, focus, and accessibility behavior | `integrated` | `App.vue`, static-shell tests; port 5181 released | Integrated as `1061996`; final matrix passed | No unresolved overlap after B test repair and D hook verification | A | Remove after main push; recovery remains in integrated commit |
| `C:\Users\raede\.codex\worktrees\fd8e\Practicum` | Nightlight UI Phase B / `019fec40-8293-7eb3-b03c-226d54ef8486` | `codex/nightlight-ui-ux-base` / `962431b680845a91b0b5b96807c77630dc82dd89` | detached / `962431b680845a91b0b5b96807c77630dc82dd89` | Improve information hierarchy, long-page structure, and progressive disclosure | `integrated` | Four content views and two tests; port 5182 released | Integrated as `683f454`; warning-free build and final matrix passed | Hash-router anchors and scroll assertion reconciled | B | Remove after main push; recovery remains in integrated commit |
| `C:\Users\raede\.codex\worktrees\2a26\Practicum` | Nightlight UI Phase C / `019fec42-d0b0-7c92-9ed3-15176c8c3785` | `codex/nightlight-ui-ux-base` / `962431b680845a91b0b5b96807c77630dc82dd89` | detached / `962431b680845a91b0b5b96807c77630dc82dd89` | Improve Atlas task flow, mobile state, and URL persistence | `integrated` | Atlas view and comparison tests; port 5183 released | Integrated as `5dc247a`; URL/history and final matrix passed | D synchronized 900px and desktop visual placement | C | Remove after main push; recovery remains in integrated commit |
| `C:\Users\raede\.codex\worktrees\feae\Practicum` | Nightlight UI Phase D / `019fec40-b5ad-72b3-8be4-212c26dd35eb` | `codex/nightlight-ui-ux-base` / `962431b680845a91b0b5b96807c77630dc82dd89` | detached / `962431b680845a91b0b5b96807c77630dc82dd89` | Consolidate visual tokens, responsive system, and final hook styles | `integrated` | Global CSS, HTML metadata, platform tests; port 5184 released | Integrated as `d715f9d`; 70/70 platform and final matrix passed | Final integration repaired summary controls and cross-lane layout contracts | D | Remove after main push; recovery remains in integrated commit |
| `C:\Users\raede\Desktop\essay help master\Practicum` | Four-lane evidence/platform release integration | `main` / `ca8292040a402eae1d2e461708a4cc912867efcb` | product candidate `bd91194f85c6cc8ce1fc3d6ced80dd66d4bf6511`; docs-only closeout follows | Integrate UI, P2/P3, P1, and platform packages; freeze and publish one exact candidate; preserve acquired source cache; then archive and clean | `integrated` | Future P1 upstream evidence, human P2 evidence, P3 field/multi-browser evidence, and cross-platform artifact-byte normalization | Exact-SHA Python 130 + 7 subtests; public 167/167; Dashboard 24/24; audits 0; local/live browser green; Pages run 31352969379; live manifest 11/11 | Protected personal-research WIP remains untracked; four task worktrees removed; `591a` remains excluded | 0 | Start future work from synchronized main; keep seven full-upstream blockers and non-human P2 boundary explicit |
| `C:\Users\raede\.codex\worktrees\591a\Practicum` | Unrelated retained task | outside this phase | `223fc653dba2768dad99df9d032beaedd9234d6a` | Preserve another task's checkout and ownership | `in-progress` | Unknown to this integration and intentionally not inspected | Not run by this integration | Excluded from all package application, tests, cleanup, and Git mutation | excluded | Leave untouched |

States: `in-progress`, `blocked`, `ready-for-review`, `ready-for-integration`, `integrated`, `abandoned`.

## Four-lane evidence/platform closeout — 2026-08-10

- Product candidate: `bd91194f85c6cc8ce1fc3d6ced80dd66d4bf6511`; normal push synchronized local and remote main before this docs-only closeout.
- Publication: GitHub Actions/Pages run `31352969379` succeeded for the exact product SHA; live schema-v2 manifest SHA-256 `288f18bed58d9c5aa9a5a4a5b53bbec2b2885a617d56ae40f7e0f52c101293a6` verified 11/11 served files.
- Worktrees: `fa7d`, `0313`, `aefe`, and `1335` registrations removed. Windows left four empty non-Git directory shells because Codex/system handles denied final directory deletion. `591a` remains registered and untouched.
- Data preservation: four TIGER/WorldPop assets plus receipts are retained under the main ignored owner cache; 4/4 destination hashes and sizes verified. No credential, Earth Engine environment, log, or failed OSM partial was transferred.
- Boundaries: reviewed-modeling is ready, full-upstream retains seven blockers; P2 has no participant/manual-AT evidence; P3 is local Chromium evidence; MapLibre remains 803,051 bytes; Windows/Linux output manifests differ in two text-file whitespace encodings even though each environment's manifest is exact.

## Next evidence phase closeout

- Product release: `e18b91a32cd92d379e4328889df1c0139f43ccee`; Pages run `31322447014` succeeded.
- P1: LF-canonical 52-file receipt plus seven honest full-upstream blockers; reviewed-modeling remains ready, not independently scientifically validated.
- P2: proxy evidence and a solo-feasible next plan; no participant or assistive-technology claim.
- P3: measured runtime no-op; MapLibre remains isolated and external map-resource settle remains the main measured delay.
- Cleanup: original task IDs archived; three lane worktree registrations removed; main-owned browsers and ports released; personal research WIP untouched.
- Residue: three empty worktree directory shells remain because the Codex app holds Windows directory handles; they contain zero files and no Git metadata.

## Prior orchestration delivery package

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

## Prior orchestration cleanup evidence

- Removed Git worktrees: P1 `3fa1`, P2 `3ab1`, canonical P3 `3a29`, and duplicate P3 `b19a` registrations; P1/P2 directory bodies were deleted.
- Archived Codex tasks: P1, P2, canonical P3, and duplicate P3.
- Generated artifacts: main Playwright sessions closed; owned ports 43217/55473/43189 released; exact Playwright snapshots and task logs deleted. One empty main temp directory shell remains because policy rejected directory removal.
- Recovery: lane commits remain reachable through `main` as `82d4f97`, `e7e95be`, and `1d2a0b1`; detached candidates also remain reachable through their integrated descendants until normal Git pruning.
- Non-Git residue: `3a29/Practicum` and `b19a/Practicum` directory bodies remain after Git registration removal because Windows/policy denied recursive deletion. Neither is a registered worktree or valid integration source.
