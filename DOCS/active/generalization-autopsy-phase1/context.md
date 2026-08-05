# Generalization Autopsy Phase 1 Context

## Current truth

- Supervisor worktree: `C:\Users\raede\Desktop\essay help master\Practicum`.
- Supervisor branch and baseline at handoff: `codex/personal-project-sync` at `b5b0b42db7a5e9d81415cdaac10fec0e85904c90`.
- Implementation worktree: `C:\Users\raede\.codex\worktrees\c24c\Practicum`, branch `codex/generalization-autopsy-phase1`, source commit `5607c0758e45412930728bed46704806e6e7907d` based on `b5b0b42db7a5e9d81415cdaac10fec0e85904c90`.
- Phase 1 is integrated locally on `codex/personal-project-sync`. The pre-integration supervisor copies of the three task records are preserved at `C:\Users\raede\.codex\integration-backups\Practicum\20260805-phase1-supervisor-before-integration\generalization-autopsy-phase1`.
- The existing public application lives under `project/nightlight-public/` and intentionally publishes only reviewed aggregates and broad event metadata.
- The public verifier is fail-closed: it rejects prohibited formats, fields, unknown release paths, runtime network calls, model artifacts, credentials, and fine-grained data.
- The private/local modeling workspace contains substantially richer outputs. Private availability is not public permission.
- Modeling evidence separates explanatory fit, damage ranking, recovery transport, and secondary interpretation. These roles must not be collapsed.
- Current cross-event recovery transport remains weak; model failure/domain shift is the product subject, not a result to hide.
- Fresh visual research found a 640-pixel mobile navigation surface that can hide the active Methods/Credits route at approximately 373 pixels, plus essential tiny text using a faint color of approximately 3.81:1 against the base background.
- Research and adversarial roadmap review are archived under `DOCS/archive/personal-project-evolution-research/`.
- The supervisor workspace already contains unrelated untracked archived research records. The implementation owner must preserve them and must not revert or delete user/supervisor WIP.

## Decisions

| Time | Decision | Impact |
| --- | --- | --- |
| 2026-08-05 | Treat the first shippable Phase 1 as accessibility admission fixes + public artifact contract + one-page Findings MVP. | Avoids delivering a schema with no user value and avoids expanding an inaccessible surface. |
| 2026-08-05 | `Generalization Autopsy` is the immediate product line; `Recovery Blind Spots Observatory` remains a distant theme only. | Prevents claims about continuous monitoring, real recovery gaps, or policy blind spots. |
| 2026-08-05 | Keep the application static, local-asset-only, aggregate-only, and without runtime external requests. | Preserves GitHub Pages/Vercel deployability and the public data boundary. |
| 2026-08-05 | Extend Findings instead of adding a new primary route. | Produces the smallest coherent public increment. |
| 2026-08-05 | The implementation thread owns code and tests in its worktree; the current thread owns supervision and later integration. | Prevents cross-worktree Git and file ownership conflicts. |

## Live process ownership

| Process | Owner | Port/output | State |
| --- | --- | --- | --- |
| None at handoff | N/A | N/A | The implementation owner must register any long-running server, browser smoke, or build before starting it. |
| Vite responsive smoke | Phase 1 implementation owner | `127.0.0.1:43186`; `cache/logs/generalization-autopsy-phase1-vite.log`; browser state and screenshots only under ignored `cache/browser-artifacts/generalization-autopsy-phase1/` | Stopped. Vite PID `8328` owned the listener and was terminated; port is released. Development-only HMR was blocked by the deliberate `connect-src 'none'` and `style-src 'self'` CSP, producing two Vite-client console errors. This is not release-smoke evidence; do not weaken CSP. |
| Production build and preview smoke | Phase 1 implementation owner | Final build logs: `cache/logs/generalization-autopsy-phase1-final-validate.log`, `cache/logs/generalization-autopsy-phase1-pages-base-build.log`, and `cache/logs/generalization-autopsy-phase1-final-root-build.log`; preview log `cache/logs/generalization-autopsy-phase1-final-preview.log`; preview port `127.0.0.1:43187` | Stopped and released. Final preview command: `npm run preview -- --host 127.0.0.1 --port 43187 --strictPort`; parent PID `54396` and owned Node listener PID `22376` were stopped after smoke, and the port was confirmed released. Final production smoke: all five hash routes at 320/373/375/768/1440 (25 checks) had no page overflow or hidden active nav; zero console errors/warnings and no non-local network requests. Reduced motion computed to `0.00001s`; keyboard focus outline was visible. |
| Post-integration public validation | Supervising integration owner | `npm run validate` in `project/nightlight-public/`; logs `cache/logs/generalization-autopsy-phase1-integration-validate.log` and `cache/logs/generalization-autopsy-phase1-integration-validate-post-hash-fix.log`; shared output `project/nightlight-public/dist/`; no port | Complete and released. Initial 72/72/build/verifier passed; provenance then exposed checkout-line-ending-dependent hashes. After canonical LF hashing, provenance passed 2/2 and the final public validation again passed 72/72, build, 11-file manifest, and source/dist verifier. No live process or port remains. |

## Required handoff updates

The implementation owner must update this file with:

- its worktree path, branch, base, and HEAD;
- exact changed-file groups;
- live process owner/log/teardown evidence if applicable;
- source artifacts and hashes admitted to the public contract;
- deviations from `plan.md` and why;
- test/build/browser evidence;
- unresolved publication, scientific, accessibility, or integration risks;
- recommended merge/cherry-pick method.

## Next action

Phase 1 is integrated and verified locally. Keep raw/private sources outside the public tree; treat push and deployment as a separate release decision.

## Implementation log

| Time | Fact or action | Evidence / next step |
| --- | --- | --- |
| 2026-08-05 | Phase 1 implementation started in the isolated Codex worktree. | Required active and archived task records read. Root and subtree `AGENTS.md` plus `lessons learned.md` were absent. Existing untracked task records are preserved; `_worktree_registry.md` remains read-only. |
| 2026-08-05 | Implemented Stage 0 admission, aggregate-only artifact v1, and Findings Generalization Autopsy MVP. | Changed public UI, static verifier, data policy, artifact contract, and focused tests only; no dependency, README, remote, deployment, or registry changes. |
| 2026-08-05 | Locked source provenance without importing private data into the public application. | Public artifact carries reviewed SHA-256 allowlisted pairs. `project/tests/test_public_generalization_provenance.py` recalculates `study.js` (`7eb682…9138a6`) and private cross-event JSON (`f6cc12…d96dd`) from the monorepo only; public code neither imports the private JSON nor contains `project/modeling`/the JSON filename. |
| 2026-08-05 | Corrected the cross-event evidence task language after supervisory scientific review. | The admitted `0.4814` evidence now has family `Cross-event logit damage-ranking model` and validation `leave-one-event-out cross-event damage-ranking`; Findings calls it a cross-event damage-ranking check. Recovery transport has no admitted v1 metric. |
| 2026-08-05 | Verified the ready-for-integration candidate. | `npm run validate`: 72 tests, build, 11-file manifest, and public verifier passed. Pages base-path build/verifier passed. Browser smoke passed 25 route/width checks with zero console errors/warnings/external requests; preview port was released. Independent code review approved and architecture review returned CLEAR. |
| 2026-08-05 | Integrated the candidate onto `codex/personal-project-sync` and corrected a cross-platform provenance defect found only after integration. | The source pointer had been derived from checkout-specific CRLF bytes. The monorepo gate now canonicalizes Git text to LF, explicitly tests LF/CRLF equivalence, and binds the public pointers to canonical hashes `8d00fd…6565ee` and `306c6c…3cb1e`. Provenance passed 2/2 and full public validation passed 72/72 after the fix. |

## Ready-for-integration handoff

- Candidate branch: `codex/generalization-autopsy-phase1`, based on `b5b0b42db7a5e9d81415cdaac10fec0e85904c90` in `C:\Users\raede\.codex\worktrees\c24c\Practicum`.
- Recommended integration: inspect the single candidate commit, then `git cherry-pick <candidate-sha>` onto the supervisor branch. Do not add `DOCS/active/_worktree_registry.md` or `DOCS/archive/personal-project-evolution-research/` from this worktree.
- Public/source boundary: source hashes lock approved lineage but are not permissions to publish raw sources. Cross-event raw source remains outside `project/nightlight-public/` and must not enter public dist.
- Residual verification limit: no manual screen-reader audit was performed. The production browser run validates semantic structures, focus visibility, reduced motion, responsive navigation, overflow, console state, and local-only requests.
