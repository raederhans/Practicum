# Proxy Evidence Phase Task

## Status

`integrated` — candidate `223631a02356eaa99866be17278f417a4174cbbc` was integrated into `main` as `232577e`. Remote release and Pages verification are owned by the umbrella integration task.

## Checklist

- [x] Verify cwd, detached HEAD, clean status, ownership, and applicable `AGENTS.md`.
- [x] Read workflow skills and prior repository evidence.
- [x] Verify required W3C, CDC, GOV.UK, and current interface-guideline sources.
- [x] Restore isolated dependencies and confirm the 119/119 baseline.
- [x] Establish the proxy evidence model and live-process ownership contract.
- [x] Add RED tests for the proxy report, owner-run plan, and deferred historical protocol status.
- [x] Write the research-supported proxy report and minimal protocol/verifier changes.
- [x] Run fresh static and local production-browser technical checks.
- [x] Decide whether product code is a measured no-op or requires an evidence-backed repair.
- [x] Run full validation, review, cleanup, and ready-for-integration handoff.

## Current blockers

None. Real-participant evidence is intentionally outside the current completion gate and must not be fabricated.

## Validation ledger

- Baseline: `npm test` passed 119/119 after `npm ci`; `npm run verify:public` passed.
- Proxy/protocol RED: 3 expected failures and 1 pass before the report and protocol status existed.
- Focus-contract RED: 2 expected failures and 27 passes against the prior focus implementation.
- Focus-contract GREEN: 29/29 passed after the minimal product repair.
- Targeted final suites: 72/72 passed.
- Complete gate: `npm run validate` passed 123/123 tests, production build, and `verify:public -- --require-dist`.
- Post-cleanup gate: `npm test` passed 123/123 and `npm run verify:public` passed with `dist` absent.
- Browser: 20/20 route-width checks and 4/4 Compare checks passed; console warnings/errors, request failures, and HTTP responses at 400 or above were all zero.
- Teardown: P2 browser sessions were closed, port 43231 was released, and P2 browser snapshots, logs, and generated `dist` files were removed. Unrelated 5174/5175 listeners and the P3 browser session were not changed.

## Integration evidence

- Main independently reran `npm run validate`: 123/123 tests passed together with the production build and required public/dist boundary.
- Fresh main-owned Chromium verification reproduced the intended first-Tab order, moved focus to the Atlas H1 after keyboard route activation, found no 320px page overflow, reported zero console errors, and observed nine static requests returning HTTP 200.
- The lane is integrated and locally admitted. It still does not constitute participant, primary-audience, screen-reader, speech, switch-device, or scientific-comprehension evidence.
