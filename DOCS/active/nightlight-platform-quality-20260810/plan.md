# Nightlight Platform Quality Plan

## Goal

Prepare a ready-for-integration platform-quality delivery for `project/nightlight-public` that proves the static-only architecture decision, makes public data/error semantics machine-checkable, hardens the public release boundary, and defines deterministic release and maintenance gates without changing UI-owned paths or claiming publication.

## Scope

- B0: fix the base SHA, dependency versions, validation entry points, release artifact shape, and fresh baseline evidence.
- B1: evaluate the four backend trigger conditions and keep static-only/local-assets-only/aggregate-only/no-analytics/no-external-requests unless evidence requires otherwise.
- B2: add a consumer-side aggregate value/error-state contract that distinguishes numeric zero, unavailable, not assessed, not applicable, suppressed, load failure, and validation failure.
- B3: fail closed on non-allowlisted release files, private paths, credentials, source maps, oversized files, runtime requests, and unsupported CSP claims.
- B4: document and mechanically support existing directory responsibilities; make only evidence-backed, minimal non-UI refactors.
- B5: define and exercise unit, integration, public-boundary, security, error-contract, and release-manifest gates.
- B6: produce a deterministic manifest with schema, base-path/build contract, stable file ordering, byte lengths, and SHA-256 hashes; keep CI/deployment claims externally anchored.
- B7: define dependency, schema-version, deprecation, migration, rollback, and periodic review policy without repeating Actions-owner research.
- B8: run targeted and full validation, security/release review, `git diff --check`, and a first-principles simplification review.

## Sources of truth

- Base commit `ca8292040a402eae1d2e461708a4cc912867efcb` in detached worktree `C:\Users\raede\.codex\worktrees\1335\Practicum`.
- `DOCS/active/_worktree_registry.md` plus read-only status and changed-file lists from P1 `aefe`, P2/P3 `0313`, UI candidate `fa7d`, and retained `591a`.
- `DOCS/archive/public-baseline-release/`, `DOCS/archive/github-pages-publication-20260809/`, and `DOCS/archive/practicum-next-evidence-phase-20260809/`.
- `project/nightlight-public/{README.md,DATA_POLICY.md,CREDITS.md,index.html,vite.config.js,package.json,package-lock.json}`.
- `project/nightlight-public/scripts/{verify-public.mjs,release-manifest.mjs}` and current tests.

## Stages

- [x] Stage B0: capture and freshly reproduce the baseline.
- [x] Stage B1: record and verify the backend/static architecture decision.
- [x] Stage B2: implement and test data/error-state contracts.
- [x] Stage B3: harden public boundary and security gates.
- [x] Stage B4: close evidence-backed integrity issues and record no-op architecture decisions.
- [x] Stage B5: establish test hierarchy and gate ownership.
- [x] Stage B6: strengthen deterministic release admission and claim levels.
- [x] Stage B7: document long-term maintenance boundaries.
- [x] Stage B8: run full verification and prepare the ready-for-integration package.

## Acceptance criteria

- Static-only remains the default and no API, database, login, analytics, runtime third-party dependency, or production mock is introduced without a proven trigger.
- Numeric zero cannot be confused with unavailable, not assessed, not applicable, suppressed, load failure, or validation failure; invalid combinations fail closed.
- Source and `dist` public boundaries use exact allowlists and reject restricted/private paths, secrets, external runtime requests, source maps, and oversized files.
- The release manifest is deterministic and binds files/hashes to the base-path/build contract without self-asserting merge, CI, deployment, or publication state.
- Directory responsibilities, dependency governance, schema lifecycle, error handling, and test commands are grounded in code or tooling.
- No UI, P1, P2/P3, workflow, registry, Git index/ref/remote, or user-WIP ownership boundary is crossed.
- Fresh targeted and full validation evidence is recorded; any unavailable gate is reported as a gap rather than a pass.

## Non-goals

- No UI/UX, route, focus, visual, zoom, text-spacing, claim/proxy, rate-limit performance, Actions upgrade, modeling, raw-data, or dashboard work.
- No server, API, database, authentication, analytics, telemetry, runtime network dependency, mock production data, TypeScript migration, state-management framework, or large architecture rewrite.
- No commit, staging, branch/ref/worktree mutation, push, deployment, registry edit, or publication claim.
- No root README change.

## Risks and constraints

- UI lane ownership is path-based even while `fa7d` is currently clean; prohibited UI files remain off-limits.
- P2/P3 may concurrently use public-app tests or live resources; B owns only its named tests and runs shared `node_modules`/`dist` commands serially from this worktree.
- `package.json` and lock/config files are shared integration hotspots and may change only for a reproduced minimal need.
- CSP meta can constrain document-loaded resources but cannot supply HTTP response headers; Pages claims must remain limited accordingly.
- Existing local/CI/browser passes are historical evidence only; this task requires fresh local proof for its candidate.
