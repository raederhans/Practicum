# Public Baseline Release Context

## Current truth

- Integration owner: the primary agent in `C:\Users\raede\Desktop\essay help master\Practicum`.
- Source branch: `codex/personal-project-sync` at `cce74cfe09c067a6d58e031fa59ad51a2dc38d03`.
- The source worktree contains one unrelated, user-owned untracked archive: `DOCS/archive/personal-project-evolution-research/`.
- The clean release clone is `cache/public-release/nightlight-disaster-dashboard-20260805-1639` at `main=4d7edac2771bfc4f5a477dd251aa544f4b553a22`, equal to `origin/main`.
- Standalone GitHub repository: `https://github.com/raederhans/nightlight-disaster-dashboard`.
- Existing stable Pages URL: `https://raederhans.github.io/nightlight-disaster-dashboard/`.
- Existing stable Vercel URL: `https://nightlight-disaster-dashboard.vercel.app/`.
- Existing Vercel project: `nightlight-disaster-dashboard` / `prj_IbPZoPT9AexNfUbrquAjPj2HLYv7`.
- Public pull request: `https://github.com/raederhans/nightlight-disaster-dashboard/pull/1`, merged by fast-forward at exact commit `4d7edac`.
- Successful GitHub Pages run: `https://github.com/raederhans/nightlight-disaster-dashboard/actions/runs/31014951772`.
- Vercel Production: `dpl_DiX1rjQ2QqQsAAQBdmPnNEQDpW5L`, READY, immutable URL `https://nightlight-disaster-dashboard-rhyo8uo57.vercel.app`.

## Decisions and deviations

| Time | Evidence or decision | Impact |
| --- | --- | --- |
| 2026-08-05 | The original `Practicum` history previously exposed partner-restricted EAGLE-I files. | Release only from the clean standalone repository; do not use the monorepo remote as the website deployment source. |
| 2026-08-05 | Phase 1 and Phase 2 are integrated and validated locally at `cce74cf`. | Lock this commit as the source snapshot for the Step 1 public baseline. |
| 2026-08-05 | The standalone clone is clean and matches `origin/main@c0ab511` before synchronization. | It is the controlled publication surface for the new release. |
| 2026-08-05 | Vercel CLI is absent, while GitHub CLI authentication is valid. | Install a pinned Vercel CLI version globally; do not alter project dependencies. |
| 2026-08-05 | Pushing GitHub `main` may trigger Git-linked Vercel Production automatically. | Push `4d7edac` to a release branch first, validate its Preview, then merge the exact commit to `main`. |
| 2026-08-05 | The standalone staged tree matched all 42 source blobs at `cce74cf`; Gitleaks found no leaks. | Commit `4d7edac` is the exact approved public candidate. |
| 2026-08-05 | Vercel Preview `dpl_ALxNpxM2vfqb3r2a6sDYkfLsfmgA` is READY for branch `codex/public-baseline-phase2`, commit `4d7edac`; its cloud build passed 85/85 tests and the public verifier. | The cloud artifact is the approved production candidate. |
| 2026-08-05 | Protected Preview browser navigation is redirected through Vercel SSO, whose asset proxy conflicts with the application's strict same-origin CSP. Three supported bypass approaches did not produce a clean browser page. | Stop retrying the same external protection condition. Use authenticated `vercel curl` for Preview integrity and move the real-browser gate to public Production with rollback available. |
| 2026-08-05 | Ten immutable Preview assets matched local bytes and SHA-256; served `index.html` differed only by Vercel's injected Preview Toolbar script, while the remote manifest retained the original index hash. | Treat the HTML delta as platform injection, not build drift; require clean Production index/hash verification after promotion. |
| 2026-08-05 | The temporary Vercel automation bypass secret appeared in local browser-session output. | Revoked it immediately; project automation bypass count is zero and SSO protection remains enabled. |
| 2026-08-05 | Public `main`, merged PR 1, and Pages run `31014951772` all resolve to `4d7edac`; both Pages jobs succeeded. | The GitHub release is exact-SHA verified. |
| 2026-08-05 | Git-linked Vercel Production rebuilt the same exact commit rather than promoting the Preview deployment object. Its cloud validation passed 85/85 tests and the public verifier. | Accept the intentional platform workflow deviation only because commit identity and emitted artifacts are independently verified. |
| 2026-08-05 | Stable Vercel and Pages manifests each listed 11 files; every downloaded file matched its declared bytes and SHA-256. Vercel security headers matched the reviewed policy. | Both public hosts serve the approved release artifact without detected drift. |
| 2026-08-05 | Real-browser smoke passed 20/20 cases across five routes, two hosts, mobile and desktop, with zero console issues, page errors, request failures, external requests, or horizontal overflow. | Step 1 meets its runtime acceptance gate. |
| 2026-08-05 | Vercel Production error-log query for the release deployment returned no errors, and the merged release branch was deleted after `main` verification. | Release cleanup is complete; rollback remains available through the prior production deployment. |

## Live process ownership

| Process | Owner | Log path | State |
| --- | --- | --- | --- |
| Local validation/build | Primary agent | `cache/public-baseline-release/local-validation.log` | Complete: 85/85 tests, root and Pages builds, two boundary checks, two 11-file manifests |
| GitHub Actions/Pages run | Primary agent | GitHub Actions run URL recorded in `task.md` | Complete: build and deploy succeeded at `4d7edac` |
| Vercel Preview/Production | Primary agent | Vercel deployment URLs recorded in `task.md` | Complete: Preview and Production READY at `4d7edac` |
| Browser smoke | Primary agent | `cache/public-baseline-release/browser-smoke.log` | Complete: Vercel 10/10 and Pages 10/10; all sessions closed |

## Handoff

Do not publish from `raederhans/Practicum`. Future website releases must continue from the standalone clean repository, preserve the aggregate-only boundary, and prove exact source-to-public commit identity before deployment.

## Next step

Begin the next product-development step from the verified `4d7edac` baseline; recommended next scope is the Phase 3 Compare Mode, with its scientific claim boundary designed before implementation.
