# Failure Atlas Phase 2 Context

## Current truth

- Owner/worktree: root agent in `C:\Users\raede\Desktop\essay help master\Practicum`.
- Branch/baseline: `codex/personal-project-sync` at `64acae7618b5de01b2c934b858a73840cbfd8f77`.
- Phase 1 is integrated locally and verified; no remote push or deployment has occurred.
- Untracked `DOCS/archive/personal-project-evolution-research/` is preserved user/supervisor evidence and is outside Phase 2 ownership.
- The public atlas contains 25 broad event records. Nine local readiness events map exactly to that public index; `dorian_freeport` does not and will remain unpublished.
- At Phase 2 start, the ignored local readiness CSV values were stale relative to the intended observation rule. They were regenerated after fixing the valid-zero censoring bug, then admitted through a tracked reviewed public manifest.
- The readiness generator also writes model-training decisions. Those outcome-derived recommendations are not a neutral observability contract and must not enter the public artifact.

## Public mapping

| Local analysis event | Public event ID |
| --- | --- |
| `ian_charlotteharbor` | `ian-charlotte` |
| `ian_fortmyers` | `ian-fortmyers` |
| `earthquake_sanjuan` | `eq-pr` |
| `ida_neworleans` | `ida` |
| `irma_miami` | `irma` |
| `laura_lakecharles` | `laura` |
| `earthquake_hatay` | `eq-hatay` |
| `maria_sanjuan` | `maria` |
| `michael_panamacity` | `michael` |

## Live process ownership

| Process | Owner | Shared resources / output | Log | Success / stop condition | State |
| --- | --- | --- | --- | --- | --- |
| Readiness regeneration | root agent | Successful command: `.venv\\Scripts\\python.exe project/modeling/experimental/intl_stage_repair_v1.py score-readiness`; working directory: repository root; ignored local outputs `project/modeling/output/event_readiness_{components,score}_v1.csv`, `event_training_decision_v1.csv`; tracked generator progress record | `cache/logs/failure-atlas-phase2-score-readiness.log` | Exit 0; only declared outputs and generator progress changed; reviewed component values match generator rules. | Complete / released |
| Public validation/build | root agent | Command: `npm run validate`; working directory `project/nightlight-public`; shared output `project/nightlight-public/dist/`; Pages `/Practicum/` and root builds ran serially in the same slot | `cache/logs/failure-atlas-phase2-validate.log`, `failure-atlas-phase2-pages-build.log`, `failure-atlas-phase2-root-build.log` | 82/82 tests; both builds emitted an 11-file manifest; source/dist verifier passed. | Complete / released |
| Production browser smoke | root agent | Command: `npm run preview -- --host 127.0.0.1 --port 43188 --strictPort`; production root build; port `127.0.0.1:43188`; isolated Playwright CLI sessions | `cache/logs/failure-atlas-phase2-preview.{stdout,stderr}.log` | Five Atlas widths and mobile/desktop checks for the other four routes passed; assessed/unassessed states switched; console 0 errors/warnings; only local static request; browser and listener released. | Complete / released |

## Decisions

| Time | Decision | Reason |
| --- | --- | --- |
| 2026-08-05 | Treat `Failure Atlas / Evidence Passport` as the next Phase 2 because the earlier one-page Generalization Autopsy was completed in Phase 1. | Avoids repeating already delivered work. |
| 2026-08-05 | Publish component assessments and a workflow band, but no overall numeric score. | A single score would look more authoritative than the evidence and encourage ranking. |
| 2026-08-05 | Show unassessed events honestly instead of filling gaps. | Prevents fabricated or silently inferred data. |

## Implementation log

| Time | Fact or action | Evidence / next step |
| --- | --- | --- |
| 2026-08-05 | Phase 2 opened on the current branch with Phase 1 cleanly integrated. | Next: add failing tests, regenerate source truth, then implement the narrow public contract. |
| 2026-08-05 | TDD red gate reproduced both gaps. | Python readiness contract: 1 failed/1 passed because six checked-in observation scores are 24 instead of the versioned-rule value 30. Public Evidence Passport test: 10 expected failures because the artifact and Atlas surface do not exist yet. |
| 2026-08-05 | Captured pre-regeneration SHA-256 values. | Components `bd39bed168c61faf81597c0a1d13679d2ebd929090ccbccd732dba4adac875f4`; score `9cf7bb60cdcc347d7b3feBC305a6d676ca375af0ed3b50acaa7af6c5d894733b`; training decision `1c6932aa660e21137fc25cdb83f0ef2879894a385dd537c1685145a131fa15ce` (case-insensitive hex). |
| 2026-08-05 | The first two regeneration attempts stopped before outputs because the selected Python environments lacked declared modeling dependencies. | System Python lacked `statsmodels`; the repository `.venv` then lacked `lifelines` and `seaborn`. Installed the project-declared `lifelines==0.30.1` and `seaborn==0.13.2` into the local ignored `.venv`; no dependency file changed. |
| 2026-08-05 | Successful regeneration exposed the executable root cause. | `_score_obs` used `value or default`, so valid `high_censoring_share == 0.0` became `1.0`. A focused test failed 24 vs 30, the function now distinguishes missing values with `pd.notna`, and the regression passed. |
| 2026-08-05 | Regenerated source truth and completed the TDD green gate. | Six high-quality events now receive 30 observation-quality points. Canonical-LF components hash: `5d2f93b69913cfe93c48cc2ea81e08499502536d92f81f72a1cbe2dcfe4a3586`. Modeling contract 3/3 passed; public artifact/UI/boundary target 52/52 passed; combined provenance checks 7/7 passed. |
| 2026-08-05 | Implemented the nine-event static Evidence Passport artifact and Atlas surface. | Five separate components, analysis-admission band, reviewed source hash, supported/unsupported claims, and attribution are public. Sixteen events render an explicit unassessed state. Dorian, total scores, raw counts/ratios, model increments, and training recommendations remain excluded. |
| 2026-08-05 | Completed static release gates for both target hosts. | `npm run validate`: 8 files, 82 tests, build, 11-file manifest, source/dist verifier. GitHub Pages build with `VITE_BASE_PATH=/Practicum/` passed. Final root/Vercel build and verifier passed. |
| 2026-08-05 | Corrected the repository-level deployment entry points. | A failing deployment contract proved the root Pages workflow still targeted the old dashboard and no root Vercel contract existed. Root GitHub Actions now builds/verifies `project/nightlight-public`; root `vercel.json` runs the same `validate` path. Deployment contract 2/2 passed. No production deployment was triggered. |
| 2026-08-05 | Completed production browser smoke and teardown. | Assessed Maria/Ian and unassessed Matthew states rendered semantically. Atlas passed at 320/373/375/768/1440 with no page overflow, visible active nav, present passport, and contained table scroll. Overview/Findings/Methods/Credits passed at 375 and 1440. Console: 0 errors, 0 warnings. Requests: local `observatory-mark.svg` only. Preview parent `71552`, listener `2736`, and browser sessions were stopped; port 43188 is released. |
| 2026-08-05 | Independent code and architecture review found two delivery blockers. | The full Python suite exposed an exact workflow-command contract mismatch; both reviewers also found that ignored private CSV files could not be a mandatory clean-clone provenance source. Architecture review additionally found the repository-root Vercel entry lacked a repository-root upload allowlist. |
| 2026-08-05 | Closed the clean-clone provenance and scoring-boundary findings. | Added a tracked, non-reversible nine-event `evidencePassportManifest.json`; the public JS artifact is generated from it, the release verifier checks its canonical SHA-256, and private CSV comparison is an optional local audit. A dependency-light scoring helper preserves valid zero censoring values without importing the full modeling stack. |
| 2026-08-05 | Closed the deployment upload and attribution findings. | Root `.vercelignore` denies everything except `vercel.json` and `project/nightlight-public/**`; the deployment contract asserts the exact allowlist. GitHub Pages uses the compatible `npm test` form. Public attribution now records NASA VNP46A2 Collection 2, Census 2022 ACS 5-year/TIGER-Line, and WorldPop Turkey 2020. |
| 2026-08-05 | Re-ran all local delivery gates after review fixes. | `project/tests`: 96 passed plus 7 subtests. Public `npm run validate`: 8 files, 84 tests, root build, 11-file manifest, source/dist verification. `/Practicum/` Pages build and final root build both passed. Vercel CLI 54.17.2 dry-run stopped before inspection because the repository is not linked; no link, upload, preview, or production deployment occurred. |
| 2026-08-05 | Corrected the final WorldPop product-level attribution after independent re-review. | The actual `Global_2000_2020/2020/TUR/tur_ppp_2020.tif` layer is now tied to WorldPop record `6443`, DOI `10.5258/SOTON/WP00645`, and the WorldPop/CIESIN 2018 citation. A regression test rejects the previously mismatched constrained-product record and DOI. |
| 2026-08-05 | Final reviews and completion gates passed. | Code review: `APPROVE`, zero findings. Architecture review: `CLEAR`. Final fresh evidence: Python 96 passed plus 7 subtests; public 85 tests, root and `/Practicum/` builds, release manifest, and source/dist verifier all passed. Temporary modeling packages were removed from `.venv`; pandas again resolves to the pre-task system 3.0.3 and the full Python suite remains green. |
