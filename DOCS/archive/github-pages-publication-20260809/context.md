# GitHub Pages Publication Context

## Current truth

- Publication owner: primary agent in the current task.
- User authorization: explicit authorization on 2026-08-09 to enable GitHub Pages and publish.
- Repository: `raederhans/Practicum`, `PUBLIC`, default branch `main`.
- Exact starting SHA: local `main == origin/main == 70dae02bcf23200cb247e840fa9f7200ebb07f5e`.
- Worktree topology: one registered worktree and one local branch (`main`).
- Preserved user WIP: untracked `DOCS/archive/personal-project-evolution-research/`; it is outside publication and commit scope.
- Pages pre-state: `GET /repos/raederhans/Practicum/pages` returns HTTP 404.
- Workflow: `.github/workflows/deploy-dashboard.yml` already uses Pages artifact upload and `actions/deploy-pages`, supports `workflow_dispatch`, and builds with `/Practicum/` as the Vite base path.
- Prior run `31309394968`: build/test/public-boundary/artifact upload passed; deploy alone failed because Pages was disabled.
- Published run: `31310854793` at exact product/documentation baseline `70dae02bcf23200cb247e840fa9f7200ebb07f5e`.
- Live URL: `https://raederhans.github.io/Practicum/`.

## Decisions

| Time | Evidence or decision | Impact |
| --- | --- | --- |
| 2026-08-09 19:26 +08:00 | Official GitHub REST documentation defines `build_type=workflow` for a custom Actions-based Pages deployment. | Enable the existing workflow without adding a legacy branch source or changing product code. |
| 2026-08-09 19:26 +08:00 | The repository is public and the authenticated account has `repo` and `workflow` scopes. | The current owner can perform the authorized Pages setting change and dispatch. |
| 2026-08-09 19:26 +08:00 | The app uses `createWebHashHistory(import.meta.env.BASE_URL)` and five declared routes. | Verify `#/`, `#/atlas`, `#/findings`, `#/methods`, and `#/credits`; no 404 fallback file is required. |
| 2026-08-09 19:26 +08:00 | The existing public verifier and release manifest passed on Linux before the earlier deploy-stage 404. | Reuse the existing workflow and demand a fresh successful run; do not weaken publication gates. |
| 2026-08-09 19:27 +08:00 | `POST /repos/raederhans/Practicum/pages` returned the public URL with `build_type=workflow`, `source.branch=main`, and `https_enforced=true`; a fresh GET returned the same state. | Pages is now enabled through the intended Actions workflow, with no legacy branch publication or custom domain. |
| 2026-08-09 19:29 +08:00 | Fresh run `31310854793` completed at exact `70dae02`; build job `93238356852` and deploy job `93238400908` both concluded `success`. | The exact-main public artifact is deployed; proceed to independent live HTTP and browser verification. |
| 2026-08-09 19:34 +08:00 | Root and `release-manifest.json` returned HTTP 200 on the first probe; HSTS is present; all 11 manifest-listed files matched both declared byte length and SHA-256. | The CDN-served artifact is byte-for-byte consistent with its release manifest, not merely reported successful by Actions. |
| 2026-08-09 19:44 +08:00 | Independent Playwright rendered all five hash routes; Atlas-to-Findings navigation worked; route-specific H1 values were present; 375x812 and 1280x720 had no page-level horizontal overflow; console reported 0 errors and 0 warnings. | Live routing and browser execution are verified. A conflicting `browse` fragment-navigation result was classified as a Windows daemon/command limitation after Playwright cross-verification. |

## Live process ownership

| Process | Owner | Command / shared resources | Log or evidence | State |
| --- | --- | --- | --- | --- |
| GitHub Pages workflow dispatch and monitor | Primary publication owner | Run `31310854793`, exact head `70dae02bcf23200cb247e840fa9f7200ebb07f5e`; GitHub `pages` concurrency group and `github-pages` environment | `https://github.com/raederhans/Practicum/actions/runs/31310854793` | Complete: build succeeded in 23s and deploy in 8s; monitor stopped after terminal success. |
| Live HTTP and browser verification | Primary publication owner | `https://raederhans.github.io/Practicum/`; no local server, port, DB, cache, or shared output directory | GitHub Pages API, 11-file hash verification, and Playwright route/console evidence captured in this task | Complete: HTTP, manifest, hashes, security metadata, five routes, mobile/desktop overflow, and console gates passed; browser session closed and generated snapshots removed. |

## Scientific and public-data boundary

- The deployed app is aggregate-only and is checked by `npm run verify:public -- --require-dist` before artifact upload.
- H4 remains reviewed-output consistency, not a complete upstream reproduction from every restricted or external raw input.
- The historical independent scientific signature remains tied to `992fe58`; this publication does not create a new independent scientific re-sign.

## Next action

Review the publication facts and documentation diff, reconcile the live registry, then create and push the documentation-only closeout commit.
