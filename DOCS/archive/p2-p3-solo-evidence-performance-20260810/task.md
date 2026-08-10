# P2 Solo Evidence, P3 Performance, and Actions Maintenance Task

## Current status

`integrated` — committed as `215bafe` and admitted in exact product candidate `bd91194f85c6cc8ce1fc3d6ced80dd66d4bf6511`. GitHub Actions run `31352969379` and live Pages verification passed. P2 remains non-human proxy evidence; P3 remains local/Chromium evidence rather than a field-performance claim.

## Checklist

- [x] Read applicable instructions, workflow Skills, registry, archived proxy plan, and P2/P3 closeouts.
- [x] Verify detached exact base `ca8292040a402eae1d2e461708a4cc912867efcb` and an initially clean state.
- [x] Establish writable/non-writable boundaries and the single-owner live-process contract.
- [x] Record official P2 method sources with access date, version/status, applicability, and boundary.
- [x] Pin P2 target/environment; complete semantic contract and material-claim evidence matrix.
- [x] Complete adapted CDC-style solo proxy with no score.
- [x] Run and reconcile two independent non-human AI/adversarial review passes.
- [x] Run P2 automated and headed-browser accessibility regressions and strict claim release gate.
- [x] Capture P3 headed, throttled cold/warm samples for preview image, MapLibre ready, and external/CARTO settle.
- [x] Decide and verify the smallest product change or a measured no-op.
- [x] Verify official Actions Node runtime evidence and update only compatible immutable official action pins.
- [x] Run final focused/full validation, diff review, bug search, and first-principles simplification review.
- [x] Stop owned services/browser, verify ports free, and remove generated dependencies/build/browser/log artifacts.
- [x] Complete the `ready-for-integration` delivery package without staging, committing, pushing, deploying, or editing coordination state.

## Validation evidence

| Command or check | Result |
| --- | --- |
| `git rev-parse HEAD` | `ca8292040a402eae1d2e461708a4cc912867efcb` |
| `git status --short --branch` | `## HEAD (no branch)` with no initial changes |
| repository instruction scan | No repository-local `AGENTS.md`; no root `lessons learned.md` |
| initial port scan | `43241`–`43244` free; `43241`/`43242` reserved for this task |
| local runtime | Node `v22.23.0`; npm `11.18.0`; exact test/build/browser versions will be recorded after dependency/browser startup |
| preview image size | `project/nightlight-dashboard/public/map_preview.png` = `885,131 B` |
| P2 focused RED/GREEN | Expected 4 failures before the report; final P2 gate merged into the existing allowlisted proxy test and passes |
| P2 final application gate | `npm run validate`: 11 files, 127/127 tests; production build, 11-file release manifest, and source/dist public boundary passed |
| P2 headed browser | 20/20 route-width and 4/4 Compare checks; Skip link, keyboard route focus, atomic polite status; zero console/page/network/HTTP error |
| Dashboard focused gate | `performanceSignals.test.js`: 3/3 passed |
| Dashboard final test | `npm test`: 4 files, 24/24 passed |
| Dashboard bundle gate | `npm run analyze:bundle` passed; MapLibre isolated from home; bundle sizes retained as context only |
| P3 corrected sample gate | 56 durable sample rows, eight groups of 7; recomputed medians/p95 match `p3-performance.md` |
| P3 final headed smoke | no pre-scroll preview node/request; post-scroll image loaded; real focused link accepted Enter; MapLibre ready reached |
| Actions gate | four official immutable pins occur once; action metadata confirms Node 24 or official composite upload-artifact v7; project `node-version: 20` unchanged |
| diff hygiene | `git diff --check` passed; only the assigned task records, P2 evidence test, Dashboard paths, and workflow are modified/untracked |
| cleanup | owned browser sessions closed; 43241/43242 free; public/dashboard `node_modules`, `dist`, and task `.runtime` removed |

## Open risks and remaining work

- One Chromium/CDP/local-server profile cannot supply field performance, other-browser behavior, user-perceived speed, or assistive-technology evidence.
- No participant, usability, comprehension, manual screen-reader, manual assistive-technology, WCAG-conformance, or independent scientific-validation evidence exists.
- CARTO/external requests were noisy and often failed while MapLibre still reached `load`; production external-settle behavior remains unverified.
- The GitHub-hosted workflow was not dispatched, so runner admission, artifact upload, environment protection, and Pages deployment remain integration-owner/CI gates.
- A failed `npx --prefix ... playwright-cli list` diagnostic wrote one log in the shared user npm cache outside this worktree; the shared cache was left untouched by ownership policy.
