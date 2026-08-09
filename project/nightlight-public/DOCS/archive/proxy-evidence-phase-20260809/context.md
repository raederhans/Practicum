# Proxy Evidence Phase Context

## Baseline

- Repository root: verified isolated Practicum worktree; local absolute path intentionally omitted from this public-source record.
- HEAD/base: `223fc653dba2768dad99df9d032beaedd9234d6a`
- Branch: detached HEAD; this lane is not an integration owner.
- Initial Git state: clean.
- Ownership: `project/nightlight-public/**` only.
- Existing baseline: 10 Vitest files, 119/119 tests after `npm ci`; `npm run verify:public` passed.
- The prior focus/title/320px evidence is accepted as historical technical evidence and will be supplemented by a fresh local browser proxy check.

## Evidence model

- WCAG/WAI checks can support tested orientation, semantics, keyboard, focus, contrast, and reflow contracts. They cannot prove scientific comprehension or establish full conformance from this bounded audit.
- W3C cognitive accessibility guidance supports clear words, short blocks, structure, labels, and explanations for technical/numerical concepts; it also recommends testing with real users, which this personal-project phase does not perform.
- The CDC Clear Communication Index is used only as an adapted solo checklist. No CDC score or usability-validation claim is permitted because CDC recommends independent reviewers, non-developer scoring, and primary-audience testing for finished materials.
- GOV.UK defines moderated usability testing as observing participants attempt tasks and think aloud. No such evidence exists in this phase.
- AI review, if performed later under the plan, is explicitly non-human and cannot supply participant evidence.

## Authoritative sources checked on 2026-08-09

- `https://www.w3.org/TR/WCAG22/`
- `https://www.w3.org/WAI/WCAG22/Understanding/focus-order.html`
- `https://www.w3.org/WAI/WCAG22/Understanding/headings-and-labels`
- `https://www.w3.org/TR/coga-usable/`
- `https://www.cdc.gov/ccindex/`
- `https://www.cdc.gov/ccindex/tool/how-to-use.html`
- `https://www.gov.uk/service-manual/user-research/using-moderated-usability-testing`
- `https://raw.githubusercontent.com/vercel-labs/web-interface-guidelines/main/command.md`

## Live-process ownership contract

| Resource | Owner | Command / cwd | Shared resource and log | Success condition | Stop condition |
| --- | --- | --- | --- | --- | --- |
| Production preview | P2 proxy-evaluation lane | `node node_modules/vite/bin/vite.js preview --host 127.0.0.1 --port 43231 --strictPort` from `project/nightlight-public` | Port 43231; log `%TEMP%\practicum-p2-proxy-evaluation-20260809\preview.log` | One owned listener, HTTP 200, and browser checks complete | Stop after checks or immediately on port/ownership mismatch |
| Playwright CLI session | P2 proxy-evaluation lane | Global `playwright-cli`, isolated sessions `p2proxy`, `p2proxyfirsttab`, and `p2proxyfinal` | Chromium processes and temporary CLI output only | Required route/viewport/keyboard/console checks recorded | Close after checks or immediately on unexpected target/process |

Preflight: port 43231 was free. Port 5174/PID 9956 and port 5175/PID 25772 belong to unrelated Hackathon services and must remain untouched.

The first `Start-Process` attempt was rejected before launch. Two later shell/log attempts produced redirect or terminal errors after Vite child processes had started; their exact worktree command lines and port ownership were checked, and owned residual PIDs 24196 and 80768 were stopped. The final stable preview used the direct Vite Node entry above, had the sole listener PID 59108, and returned HTTP 200 before browser work. No further server restart was attempted after that owner was established.

## Decisions

- Do not create product UI changes to manufacture a diff. Product code changes require both an authoritative-rule mapping and current reproducible evidence.
- Mark `USER_STUDY_PROTOCOL.md` as a deferred, optional historical instrument. It is not the current phase and not a release gate.
- Publish the current evaluation as a proxy report with explicit allowed and prohibited conclusions.
- Keep the new next-phase plan independently executable by the project owner without recruitment.
- Repair the two browser-reproduced focus defects only: preserve the initial Skip-link Tab target by centering navigation through `scrollLeft`, and identify the first real route change by entered route path before focusing the new H1.

## Final technical evidence

- Fresh first-Tab order in Chromium: BODY → Skip link → identity → Overview → Study Atlas. Every tested focus target had a visible 2px solid acid outline; Atlas controls also had a 4px outline offset.
- Activating Study Atlas by keyboard produced the title `Study Atlas | Nightlight Disaster Observatory` and moved focus to the visible H1 `Broad context, bounded detail`.
- Five routes at 320, 373, 375, and 768 CSS pixels: 20/20 checks passed with no page-level horizontal overflow, missing route label/title/H1, inactive navigation state, or unlabeled native form control.
- Atlas Compare Mode at the same widths: 4/4 checks passed.
- Final diagnostics: zero console warnings/errors, zero request failures, and zero HTTP responses at 400 or above; observed local assets returned 200/304.
- Fresh bounded contrast calculations from current CSS: faint/raised 4.80:1, faint/ink 5.06:1, muted/ink 8.34:1, paper/ink 15.75:1, and acid/ink 17.33:1.
- Complete gate: `npm run validate` passed 11 test files and 123/123 tests, built the production candidate, and passed the required-dist public boundary.
- Post-cleanup gate: `npm test` passed 123/123 and `npm run verify:public` passed with generated `dist` absent.

## Teardown evidence

- The final owned listener PID 59108 was rechecked against port 43231 and the complete worktree command line, then stopped; port 43231 was free afterward.
- All P2 Playwright sessions were closed. The immediate teardown check showed only the unrelated P3 session, which P2 left untouched; the final read-only check later showed `(no browsers)` after that external session had ended.
- P2 `.playwright-cli` snapshots, `%TEMP%` preview log/directory, and generated `dist` files/directories were removed after validation.
- Ports 5174/PID 9956 and 5175/PID 25772 were not stopped or modified.

## Handoff checkpoint

Current state: `integrated`. Research, documentation, minimal focus repair, TDD, static validation, production-browser proxy checks, review, and owned-resource cleanup are complete. Candidate `223631a02356eaa99866be17278f417a4174cbbc` was integrated into `main` as `232577e`; the umbrella integration task owns remote release, Pages verification, task archival, and worktree cleanup.
