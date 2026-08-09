# Public Baseline Release Task

## Current status

Complete. Source snapshot `cce74cfe09c067a6d58e031fa59ad51a2dc38d03` is published through the separate clean repository as exact public commit `4d7edac2771bfc4f5a477dd251aa544f4b553a22` on GitHub Pages and Vercel.

## Checklist

- [x] Confirm source commit, branch, remote, worktree, authentication, and data boundary.
- [x] Confirm clean standalone release repository and existing Vercel project link.
- [x] Synchronize the standalone tree from the reviewed source snapshot.
- [x] Run local tests, public-boundary verifier, root build, Pages base-path build, and release-manifest checks.
- [x] Review the standalone diff and scan the complete public history for forbidden files/secrets.
- [x] Commit and push the standalone release to GitHub `main`.
- [x] Verify GitHub Actions and the live GitHub Pages deployment at the exact commit.
- [x] Deploy and verify a Vercel Preview for the same commit.
- [x] Release the same exact commit to Vercel Production and verify the resulting artifact.
- [x] Verify stable URLs, security headers, route behavior, and emitted asset hashes.
- [x] Update release records, worktree registry, and final Git facts.

## Validation evidence

| Command or check | Result |
| --- | --- |
| `git status --short --branch` | Source branch at `cce74cf`; only unrelated user archive is untracked. |
| `git worktree list --porcelain` | Main integration worktree plus retained, already-integrated Phase 1 source worktree. |
| `gh auth status` | Authenticated as `raederhans` with repository/workflow access. |
| Standalone `git fetch/status` | Clean `main`, equal to `origin/main@4d7edac`. |
| Vercel project link | Existing project `nightlight-disaster-dashboard`, project ID `prj_IbPZoPT9AexNfUbrquAjPj2HLYv7`. |
| Standalone source identity | All 42 staged blobs matched `cce74cf:project/nightlight-public`; 14 files changed relative to `c0ab511`. |
| `npm ci` | 73 packages installed, 0 vulnerabilities. |
| `npm run validate` | 8 test files and 85/85 tests passed; root build, 11-file manifest, and source/dist verifier passed. |
| Pages base-path build | `VITE_BASE_PATH=/nightlight-disaster-dashboard/`; 11-file manifest and source/dist verifier passed. |
| Gitleaks `v8.30.1` | Scanned complete three-commit history plus worktree; no leaks found. |
| Standalone release commit | `4d7edac2771bfc4f5a477dd251aa544f4b553a22`, 14 files, 1539 insertions, 62 deletions. |
| GitHub candidate | Release branch and PR 1 head were exactly `4d7edac`; Vercel Preview check succeeded. PR is now merged and the temporary branch is deleted. |
| Vercel Preview | `dpl_ALxNpxM2vfqb3r2a6sDYkfLsfmgA`, READY, `https://nightlight-disaster-dashboard-e44x1885s.vercel.app`, cloud build 85/85 and public verifier passed. |
| Protected Preview integrity | Ten immutable assets matched local bytes/SHA-256; `index.html` content matched except for Vercel's injected Preview Toolbar tag. Expected security headers were present through authenticated access. |
| Preview browser limitation | Vercel SSO proxy rewriting conflicted with the strict CSP; the isolated Preview session was closed and the browser gate was moved to, and passed on, public Production. |
| Bypass-secret hygiene | Temporary automation bypass secret revoked; project bypass count is zero. |
| Public Git state | `origin/main=4d7edac`; PR 1 is merged with merge commit `4d7edac`; the merged release branch was deleted. |
| GitHub Pages | Run `31014951772` succeeded for head SHA `4d7edac`; build and deploy jobs both completed successfully. |
| Vercel Production | `dpl_DiX1rjQ2QqQsAAQBdmPnNEQDpW5L`, READY, stable alias points to the same deployment; cloud build passed 85/85 tests and the public verifier. |
| Stable artifact identity | Pages 11/11 and Vercel 11/11 files matched their live manifests by byte length and SHA-256; Vercel manifest equals the reviewed local root build manifest. |
| Stable security policy | CSP, `Referrer-Policy`, `X-Content-Type-Options`, and `Permissions-Policy` match the reviewed Vercel configuration. |
| Production browser smoke | Vercel 10/10 and Pages 10/10 across five routes, mobile and desktop; zero console issues, page errors, request failures, external requests, and horizontal overflow. |
| Post-deploy error scan | Vercel error-log query for the Production deployment returned no error entries. |
| Process cleanup | All isolated Playwright sessions closed; no browsers remain. |

## Remaining maintenance notes

- GitHub reports that some current JavaScript actions still run through a Node.js 20 compatibility path that will be forced to Node.js 24; update those action versions in a later maintenance change.
- The globally installed Vercel CLI `58.5.1` emitted transitive-package deprecation warnings during installation; this did not alter project dependencies or block authenticated release operations.
- The original `Practicum` history remains unsuitable as a public deployment source because of its historical restricted-data exposure; keep using the clean standalone repository.
- Raw/restricted data remains unpublished by design; only reviewed aggregate artifacts are present in the public release.
