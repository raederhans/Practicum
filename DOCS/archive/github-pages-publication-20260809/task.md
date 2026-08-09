# GitHub Pages Publication Task

## Current status

`complete` — workflow-driven Pages is enabled, exact product SHA `70dae02` is deployed and independently verified, the scientific boundary is unchanged, and the documentation-only closeout is prepared for archive/commit.

## Checklist

- [x] Load the task-record, live-test orchestration, and GitHub workflow instructions.
- [x] Confirm explicit external-publication authority.
- [x] Confirm `main == origin/main == 70dae02`, repository visibility, GitHub authentication, and Pages 404 pre-state.
- [x] Read the existing Pages workflow, Vite base-path contract, hash-router contract, and public boundary metadata.
- [x] Verify the official GitHub REST contract for workflow-driven Pages.
- [x] Enable Pages with `build_type=workflow` and capture the authoritative returned state.
- [x] Dispatch one fresh workflow run at exact `main` and record its run ID.
- [x] Monitor build and deploy jobs to a terminal successful conclusion.
- [x] Verify live root, manifest, listed assets, HTTPS, and response/security metadata.
- [x] Run browser smoke across all five hash routes and inspect console/network failures.
- [x] Review publication facts and preserve the narrower scientific interpretation boundary.
- [x] Reconcile the live registry and prepare this completed record for the documentation-only archive commit.

## Evidence

| Check | Result |
| --- | --- |
| Preflight Git exact state | Before creating these task records, local and remote main both resolved to `70dae02bcf23200cb247e840fa9f7200ebb07f5e`; the preserved user research was the only pre-existing untracked content. |
| Repository settings | `raederhans/Practicum` is `PUBLIC`, default branch `main`. |
| Authentication | Active `raederhans` GitHub login; token scopes include `repo` and `workflow`. |
| Pages pre-state | REST API returns HTTP 404. |
| Existing release workflow | `workflow_dispatch`; Node 20 install; tests/public verifier; `/Practicum/` base build; production verifier; Pages artifact; deploy-pages. |
| Routing/security contract | Vue hash history; routes `/`, `/atlas`, `/findings`, `/methods`, `/credits`; CSP and no-referrer meta tags in the built entry HTML. |
| Pages enablement | REST POST and follow-up GET return `https://raederhans.github.io/Practicum/`, `build_type=workflow`, `source.branch=main`, `public=true`, and `https_enforced=true`. |
| Fresh dispatch | Run `31310854793` was created once for exact head `70dae02bcf23200cb247e840fa9f7200ebb07f5e`: `https://github.com/raederhans/Practicum/actions/runs/31310854793`. |
| Workflow result | Build job `93238356852` succeeded in 23s; deploy job `93238400908` succeeded in 8s; overall conclusion `success`. |
| Initial live probe | Pages root and `release-manifest.json` returned HTTP 200 on the first post-deploy attempt; root is HTTPS with HSTS and the manifest lists 11 release files. |
| Live release identity | All 11 manifest-listed files match their declared byte lengths and SHA-256 hashes; zero mismatches. |
| Live HTML boundary | CSP meta retains `default-src 'self'` and `connect-src 'none'`; `no-referrer`, `/Practicum/` asset paths, and no source-map reference were confirmed. |
| Browser routes | Independent Playwright rendered `#/`, `#/atlas`, `#/findings`, `#/methods`, and `#/credits`; Atlas-to-Findings navigation also succeeded. |
| Browser quality | Route-specific H1 values were present; no page-level horizontal overflow at 375x812 or 1280x720; console reported 0 errors and 0 warnings. |
| Temporary artifacts | Playwright session closed; seven generated snapshot YAML files removed; no product or user WIP file was changed. |

## Open risks

- GitHub Actions warns that pinned actions still declare a deprecated Node 20 runtime and are currently forced onto Node 24. This is workflow-maintenance debt, not a failure of this deployment.
- The existing MapLibre dashboard bundle warning is performance debt, not a publication correctness failure.
- The release does not expand the scientific claim: H4 remains reviewed-output consistency, and no new independent exact-SHA scientific re-sign was performed.
