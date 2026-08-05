# Task

## Current status

Archived as complete on 2026-08-05 after independent verification. The verified local synchronization and publication phases are complete on `codex/personal-project-sync`. The aggregate-only website is published from an unrelated clean Git history, and both GitHub Pages and Vercel have passed live verification without publishing restricted data.

## Checklist

- [x] Inventory local files, languages, data, entry points, and Git history.
- [x] Fetch and inventory teammate remote refs.
- [x] Assess merge-base/history topology and classify commit/file divergence.
- [x] Locate and inspect the public demo.
- [x] Build the evidence-backed gap matrix.
- [x] Recommend staged synchronization and personal-project packaging.
- [x] Perform final review and first-principles check.
- [x] Create and verify an isolated full-history reference clone.
- [x] Produce initial data/source/result manifests.
- [x] Import the approved dashboard slice without overwriting protected local paths.
- [x] Import the approved Stage 2 and Stage 3 slices with provenance records.
- [x] Acquire reproducible upstream data and run bounded regeneration experiments.
- [x] Normalize canonical geographic scope and metrics.
- [x] Run targeted analysis validation, frontend build, and browser/static-asset smoke checks.
- [x] Review and create local Lore commits; leave remote publication gated.
- [x] Define and test the public code/data/license boundary.
- [x] Implement the standalone aggregate-only personal website.
- [x] Build and scan a clean release tree and production `dist`.
- [x] Verify all public routes and responsive behavior in a real browser.
- [x] Create and publish a new clean GitHub repository with workflow-driven Pages.
- [x] Create and publish an independent Vercel project from the same clean release tree.
- [x] Verify both live sites, emitted assets, routes, attribution, and absence of restricted files.

## Validation evidence

| Command or check | Result |
| --- | --- |
| Initial Git status | Clean `main` at `1e3bcda`; tracks `origin/main`. |
| Remote inventory | `origin` and `teammate` are configured. |
| Worktree inventory | One current worktree; `.git/worktrees` is absent, so no additional worktree is available as a synchronization source. |
| Fresh remote tips | `origin/main=1e3bcda`; `teammate/main=1f63e19`. |
| Tree comparison | 740 local files versus 2,416 teammate files; 601 local-only, 2,277 teammate-only, 139 shared. |
| History topology | Shallow repository; no locally provable merge base in this checkout; this does not prove unrelated complete histories. |
| Public website | HTTP 200; overview and interactive map loaded; 24 locations, 25 events, five basemaps, facility overlays. |
| Deployment source | GitHub Pages workflow builds `project/nightlight-dashboard` with Node 20, `npm ci`, and Vite. |
| Result consistency check | Stage 3 JSON reports `r_squared=0.7472` and `ratio_q3_q1=0.63`; slides still show 0.475 and 38%. |
| Geographic scope check | Full display config: 15 U.S. states + Puerto Rico + Turkey; Stage 3 panel: 22 events across 15 U.S. states. |
| Independent evidence review | PASS after correcting geographic scope, shallow-history wording, and final worktree inventory. |
| Independent plan critique | ACCEPT after adding permission, allowlist, data/build, and deployment gates. |
| Implementation authorization | User confirmed teammate permission and authorized necessary software installation, data downloads, program execution, and experiments for Steps 3–5. |
| Integration branch | `codex/personal-project-sync`, based on unchanged local `main@1e3bcda`. |
| Full-history reference clone | `C:/Users/raede/.codex/tmp/practicum-reference-20260805`; `--is-shallow-repository=false`. |
| Complete-history topology | Merge base `7b0445fb`; origin-only 17 commits, teammate-only 40 commits; neither tip is the other's ancestor. |
| Complete-history authorship | Origin-only range: 12 commits by `yu qiushi` and 5 by local `root`; teammate-only range: 40 commits by `Zhiyuan Zhao`. |
| Selective donor import | Commit-pinned plan copied 343 allowlisted files; the immediate import receipt found 343 identical and 0 conflicts. The final local comparison finds 24 intentionally personalized files and 319 still donor-identical. |
| Census 2020 acquisition | Official national ZCTA520 and County archives downloaded to ignored cache; advertised bytes and SHA-256 verified; receipt contract test passes. |
| ACS 2022 acquisition | Official table-based 5-year files normalized to 33,774 unique ZCTAs; all 1,002 Stage 3 rows join, 977 have both controls; SHA-256 recorded. |
| Hurricane-track acquisition | NHC HURDAT2 normalized to 55,605 rows / 2,004 storms; all required storm-years present; source and output SHA-256 recorded. |
| OSM reproducibility check | A compliant, receipt-producing Overpass downloader returned 399 current Zeta/Atlanta POIs; the five-row drift from the donor cache is recorded rather than silently substituted. |
| Formal Stage 3 regeneration | Ensemble TIF band 3 + EPSG:5070 produced 1,002 ZIP-event observations across 22 events; M1+ in-sample R² `0.7603`, adjusted R² `0.7543`, n `977`. |
| Statistical correction | OLS inference clusters by event; spatial KNN is event-blocked with 22 components / zero cross-event edges; event-constant NTL-drop controls are reported not identifiable. |
| Canonical claims | Donor `0.7472/0.7408/0.63` is retained only as a retired RF-band-1/Mercator baseline; current dashboard uses `0.7603/0.7543` and labels `0.551` descriptive-only. |
| Final review repairs | LOEO now renders all 25 events in a 1,380 px scrollable SVG; zero-POI downloads retain a readable schema; Moran residuals and coordinates are aligned before fitting with deterministic permutation inference; probability GeoJSON is non-empty and bounded to finite values in `[0,1]`; duplicate map clicks are removed. |
| Python validation | Fresh full suite: 85 tests plus 7 unittest subtests passed. Python compilation and seven tracked JSON documents also passed. |
| Dashboard validation | Clean install and audit found zero vulnerabilities; Vitest 17/17 and Vite 6.4.3 production build passed under Node 20.20.2. |
| Browser smoke | Fresh Chromium first loaded Maria probability and facility data, then switched to the glyph-free Satellite style with one map canvas and zero console warnings/errors/page errors. A separately injected Positron style failure produced the expected visible error and zero canvases; after the failure was removed, selecting Voyager restored one canvas and cleared the error without a page exception. The 25-event chart measured `viewBox=0 0 1380 240` with all 50 model bars in bounds. |
| Independent final code review | APPROVE with zero blocking findings after glyph fallback, stale-map ownership, moveend race, failure recovery, and teardown repairs. |
| Pages workflow hardening | Build permissions are limited to `contents: read`; only deploy receives Pages/OIDC write permissions; all four official Actions are pinned to reviewed commit SHAs. |
| Security release review | No high-confidence secrets or restricted raw fields were found in the dashboard artifact, but the existing public Git history exposes partner-restricted EAGLE-I files. Public push/deploy remains blocked pending rights confirmation and history/repository remediation. |
| Attribution and release state | `PROJECT_PROVENANCE.md` credits both contributors; the personal dashboard leads with Qiushi Yu and credits Zhiyuan Zhao; no push or deployment occurred. |
| Functional Lore commit | `acc4b2def96a4e653a604f33e8f462472fe1671d`; 370 reviewed files, local only. |
| Publication decision | User permits omission of data and authorizes choosing the license; selected boundary is MIT for owned public-site code only, with data excluded and separately documented. |
| Release vehicle | A new standalone repository will be created; the existing public `Practicum` history will not be reused because it already contains restricted data. |
| Standalone public validation | Vitest 53/53, Vite production build, release-manifest verification, fail-closed source/dist boundary, npm audit, and Gitleaks all pass. Independent code and security re-reviews both return APPROVE. |
| Standalone browser validation | Overview, Atlas, Findings, Methods, and Credits routes load in Chromium. Atlas selection/filter recovery works; 1,440 px and 390 px layouts render; console reports 0 errors/0 warnings; observed requests are same-origin static assets only. Preview PID ownership was verified and port 43185 was released. |
| Clean public history | New repository `raederhans/nightlight-disaster-dashboard`; final `main=c0ab511ac106cf5bc99f65d602afe3a8d4f71d85`. History contains three reviewed commits and 37 tracked files. Gitleaks history scan passes; forbidden raw/data/model path search is empty. |
| Vercel upload gate | Vercel CLI 58.5.1 dry-run reports Vite, exactly 34 approved upload files, and exact equality with tracked source minus GitHub-only metadata. A real RED exposed and repaired the original recursive-ignore rule. |
| GitHub Pages release | Workflow run `30991589506` succeeds for `c0ab511`; live URL `https://raederhans.github.io/nightlight-disaster-dashboard/`. All 11 emitted files match release-manifest bytes and SHA-256; five hash routes load with zero console warnings/errors. |
| Vercel release | Project `prj_IbPZoPT9AexNfUbrquAjPj2HLYv7`; production deployment `dpl_4Xv3c8vK1A8Ea1N8cqtJcma1rXxs` is READY for `c0ab511`; stable URL `https://nightlight-disaster-dashboard.vercel.app/`. Cloud build passes 53/53 tests and public boundary, all 11 emitted hashes match, security headers pass, and five routes report zero console warnings/errors. |
| Temporary credential cleanup | Vercel-generated root `.env.local` and `.vercel/.env.production.local` were removed by exact verified paths after use. They were ignored and never entered Git or the Vercel dry-run upload set; `.vercel/project.json` remains locally for project linking. |
| Independent completion verification | PASS with no blocking finding. The verifier matched GitHub `main`, Pages workflow/deployment, Vercel production metadata/build logs, task records, live assets, and both 11-file manifests to `c0ab511`; all live hash comparisons had zero mismatches. |

## Open risks and remaining work

- Complete history is now available in the isolated reference clone; the active worktree remains shallow by design and must not be used alone for later ancestry claims.
- Missing public data prevents a full Stage 0-to-3 rerun from this checkout alone.
- The original `Practicum` repository has no project-wide LICENSE and remains a historical/restricted-data source. The standalone public repository resolves this boundary with MIT for owned code/documentation, explicit collaborator attribution, and separate third-party/data notices.
- The personal branch now normalizes the headline numbers and geographic scope; the teammate's already-published site remains a historical reference and is unchanged.
- Stage 3 results are explicitly classified as EAGLE-I-derived targets with `rights-review-required`; local scientific validation does not itself grant publication rights.
- Some original data may require credentials or non-public transfer; each unavailable item needs a recorded reproducibility limitation and an official-data alternative where scientifically valid.
- The public `origin` history already exposes 52 partner-restricted EAGLE-I CSVs (215,544,230 bytes), and an unauthenticated raw URL returned HTTP 200 during security review. A deletion-only commit cannot retract history; keep deployment from the original `Practicum` history blocked. The standalone release is safe because it uses a separate clean repository.
- The clean public release is complete. The original `Practicum` repository remains unchanged as a historical/restricted-data source and must not be reused as this site's deployment history.
