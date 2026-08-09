# P2 User Understanding and Accessibility Context

## Baseline

- Worktree: `C:\Users\raede\.codex\worktrees\3ab1\Practicum`
- HEAD/base: `a79921b2becb81388762b10a365744d014026198`
- Branch: detached HEAD; this lane is not an integration owner.
- Deployed application: `https://raederhans.github.io/Practicum/` (read-only audit target).
- Existing local routes: Overview, Study Atlas, Findings, Methods, and Credits / Policy.
- Prior evidence establishes accessibility as an admission gate and requires extending existing responsibilities instead of adding a primary route.

## Live-process ownership contract

| Resource | Sole owner | Command / target | Runtime evidence | Success condition | Failure / stop condition |
| --- | --- | --- | --- | --- | --- |
| Local production preview | P2 lane agent | `npm run preview -- --host 127.0.0.1 --port 43189 --strictPort` from `project/nightlight-public` | First listener PID 71772; final listener PID 16344; HTTP 200; temporary logs summarized and deleted | Complete: final browser gates passed | Released: both owned listeners stopped after command-line and port verification; 43189 has no listener |
| Playwright CLI browser | P2 lane agent | Isolated CLI sessions against deployed site and `http://127.0.0.1:43189/` | Sessions `p2live`, `p2local`, and `p2final`; snapshots summarized below | Complete: deployed baseline and final local gates passed | Released: all sessions closed; `playwright-cli list` returned no browsers; snapshot files deleted |

The lane did not own any pre-existing server, browser, database, cache, output directory, or remote deployment. Port 5174/PID 9956 and port 5175/PID 25772 remained listening and untouched after teardown. All run-local log, snapshot, and `dist` files were deleted; only empty ignored directory shells remain because recursive directory removal was blocked by the execution policy.

## Evidence and decisions

| Date | Evidence | Decision |
| --- | --- | --- |
| 2026-08-09 | Local base is detached at `a79921b2`; the user assigned only `project/nightlight-public/**` plus this lane-local record. | Do not mutate Git state or shared coordination records. |
| 2026-08-09 | Current source already includes semantic SVG titles/descriptions, text/table alternatives, `aria-live` updates, visible `:focus-visible`, reduced-motion handling, responsive navigation, and explicit readiness-versus-recovery boundaries. | Treat these as existing controls; search for narrower reproducible gaps rather than restyling the product. |
| 2026-08-09 | No actual participant evidence is available and prior release notes list manual assistive-technology study as unverified. | Produce an executable protocol with blank results; do not claim user validation. |
| 2026-08-09 | Port 5174 is owned by unrelated PID 9956 (`authenticated_demo_server.py` for Hackathon1); 5175 is also occupied; 43189 is free. | Preserve both existing listeners and use strict, lane-owned port 43189. |
| 2026-08-09 | Deployed baseline browser audit covered all five routes at 320/373/375/768: 20/20 had no document overflow and the active mobile navigation item was visible. Console was 0 errors / 0 warnings and all requests were same-origin static files. | Preserve the existing responsive, semantic, contrast, and interpretation controls. |
| 2026-08-09 | On the deployed build, activating Study Atlas left `document.activeElement` on the navigation link and every route kept the same document title. | Repair SPA route context with unique titles and focus the new H1 after, but not during, the first asynchronous route render. |
| 2026-08-09 | The first local browser pass proved that Vue's initial Transition also emits `after-enter`; an unconditional handler focused the initial H1 and skipped earlier tab stops. | Add a one-render gate, then focus only after later route transitions. This was reproduced before the repair and rechecked in a clean browser session afterward. |
| 2026-08-09 | Palette calculation for actual text/background uses found normal-text ratios of at least 4.80:1 for `--faint` on `--ink-raised`, 5.06:1 on `--ink`, and higher ratios for muted/paper/status colors. Source and snapshots retain text/state labels rather than color-only meaning. | No contrast color change was justified. Remove the two local `outline: none` suppressions so shared focus treatment remains visible. |
| 2026-08-09 | Final local production browser audit covered five routes plus Compare Mode at 320/373/375/768: 24/24 had no document overflow, five route titles were unique, active navigation remained visible, Compare retained the always-on outcome boundary and polite live summary, and console/network gates were clean. | Mark the browser repair ready for integration; do not claim manual assistive-technology or real-user validation. |
| 2026-08-09 | The executable protocol defines 5 planned participants, setup, five tasks, prompts, success/error coding, critical-error triggers, stop rules, closing questions, and blank tables that explicitly say no sessions have run. | Treat it as a study instrument only. Results must be added later from consented sessions; no finding is present in this lane. |

## Handoff checkpoint

- State: `ready-for-integration`; no Git index, ref, branch, worktree, or remote state was changed.
- Product repair: unique per-route titles; focus moves to each new route H1 only after later route transitions; initial load retains normal Skip link/navigation tab order; main and Atlas filters use the shared visible focus treatment.
- Study artifact: `project/nightlight-public/USER_STUDY_PROTOCOL.md`; this is an executable blank instrument, not evidence of usability or comprehension.
- Tests/verifier: focused RED failures were observed before implementation; final `npm run validate` passed 119/119 tests, built 11 manifest files, and passed source/dist boundary verification. After deleting `dist`, the final source-only test and verifier are recorded in `task.md`.
- Browser evidence: deployed baseline 20 route-width checks; final local 20 route-width plus 4 Compare-width checks; keyboard route change focused the new Credits H1 with a visible 2px outline; Atlas search/select controls showed the same 2px outline; console 0 errors/warnings; local requests only.
- Teardown: Playwright reports no browsers; port 43189 is released; 0 generated files remain in the lane temp and `dist` directories; 5174 and 5175 remain owned by unrelated processes.
- Residual risk: no actual participants, screen-reader user, manual screen-reader matrix, browser family beyond Chromium, zoom/reflow above 100%, or deployment verification of this unintegrated change.
- Recommended integration: the integration owner should inspect this worktree against base `a79921b2`, stage only the 12 modified public files plus `USER_STUDY_PROTOCOL.md`, `tests/user-study-protocol.test.js`, and these 3 lane records, create one scoped commit, then rerun `npm run validate` and the 320/373/375/768 production browser gate on the integration candidate. Resolve any overlap in `App.vue`, `routes.js`, `main.css`, the five H1 views, or shared tests by preserving both the one-render focus gate and any newer product behavior.
