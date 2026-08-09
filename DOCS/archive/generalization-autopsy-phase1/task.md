# Generalization Autopsy Phase 1 Task

## Status

`integrated` — Phase 1 is present on `codex/personal-project-sync` after cherry-picking source commit `5607c0758e45412930728bed46704806e6e7907d` and fixing the Windows/LF provenance-hash boundary. No push or deployment was performed.

## Checklist

- [x] Record implementation worktree, branch, base, and HEAD in `context.md`.
- [x] Read the repository evidence, existing public contract, verifier, tests, and archived roadmap.
- [x] Confirm the exact Stage 0 failure modes with targeted tests and responsive measurements.
- [x] Add regression coverage for mobile active-route visibility and essential-text contrast behavior.
- [x] Implement and verify the Stage 0 accessibility admission fixes.
- [x] Define the `Public Generalization Artifact v1` schema, roles, enums, lineage, publication status, and prohibited fields.
- [x] Identify only publication-reviewed aggregate source values; withhold anything unclear.
- [x] Implement the narrow offline/build-time artifact path and exact verifier allowlist changes.
- [x] Add positive and negative contract/verifier tests.
- [x] Implement the Generalization Autopsy sections in the existing Findings route.
- [x] Add accessible summaries, semantic tables, non-color status cues, headings, and SVG descriptions.
- [x] Update Methods/Data Policy only as required by the new contract.
- [x] Run targeted scientific, contract, UI, and accessibility tests.
- [x] Run the complete supported public-site validation and both deployment-base checks.
- [x] Run fresh five-route responsive/browser smoke with no owned process or port collision.
- [x] Perform a final review, bug search, and first-principles simplification pass.
- [x] Update `context.md` and this task with exact evidence and remaining risks.
- [x] Return `ready-for-integration` delivery package to the supervising thread; do not push or deploy.

## Validation evidence

| Check | Result |
| --- | --- |
| Targeted artifact/UI/public-boundary test | `npm run test -- tests/generalization-artifact.test.js tests/public-boundary.test.js tests/static-shell.test.js` — 3 files, 58 tests passed. |
| Artifact wording regression | `npm run test -- tests/generalization-artifact.test.js` — 1 file, 19 tests passed; locks cross-event logit damage-ranking family/check wording and rejects hash/role/lineage violations. |
| Private-source provenance gate | `py -3 -m pytest project/tests/test_public_generalization_provenance.py -q` — 1 passed; recalculates both reviewed source SHA-256 values while asserting the private source path is absent from public artifact code. |
| Complete public validation | `npm run validate` — 7 files, 72 tests passed; production build emitted 11 manifest files; `verify:public -- --require-dist` passed. |
| GitHub Pages base path | `VITE_BASE_PATH=/nightlight-disaster-dashboard/ npm run build && npm run verify:public -- --require-dist` — 11 manifest files and public boundary verification passed. |
| Production browser smoke | Preview at `127.0.0.1:43187`: all five existing routes at 320/373/375/768/1440 (25 checks) passed with no document overflow or hidden active nav; zero console errors/warnings and no non-local requests. Keyboard focus and reduced-motion rules passed in the same production build. |
| Independent review | Code review: approve. Architecture/provenance review: CLEAR after standalone public-test separation and withheld-metric contract correction. |
| Post-integration provenance | `py -3 -m pytest project/tests/test_public_generalization_provenance.py -q` — 2 passed, including LF/CRLF canonicalization. Public source pointers now use canonical LF SHA-256 values. |
| Post-integration public validation | `npm run validate` — 7 files and 72 tests passed; production build emitted 11 manifest files; source/dist verifier passed. Log: `cache/logs/generalization-autopsy-phase1-integration-validate-post-hash-fix.log`. |

## Open risks

- Data rights remain limited to reviewed CC BY 4.0-derived aggregates with attribution; the raw cross-event decision JSON is intentionally not bundled or referenced by path in public source/dist.
- `0.4814` is only a held-out **damage-ranking** Logit AUC, not recovery transport, calibration, future-event readiness, community recovery, fairness, causality, or policy effect. Recovery transport is explicitly unavailable in v1.
- Browser checks cover keyboard focus, semantic alternatives, responsive overflow, and reduced motion, but do not replace a human assistive-technology audit.
- `.playwright-cli` is a local tool temporary directory whose attempted recursive cleanup was blocked by the runtime safety policy; it is untracked and excluded from the implementation commit.
