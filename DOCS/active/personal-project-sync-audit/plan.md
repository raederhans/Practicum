# Plan

## Goal

Identify what the local project is missing relative to the teammate's final public repository and demo site, then define a safe synchronization path that preserves the user's original analysis and collection work.

After the audit, implement the approved Steps 3–5: verify complete history in isolation, create source/result manifests, acquire reproducible upstream data where possible, and selectively synchronize the teammate-authored Stage 2, Stage 3, dashboard, and deployment-ready slices under explicit permission.

## Scope

- Compare local `main`, the user's `origin`, and the teammate's public repository.
- Inventory code, data references, generated artifacts, documentation, and visible product behavior.
- Inspect the public demo as supporting evidence when its URL can be verified.
- Produce a gap matrix and a staged synchronization recommendation.

## Sources of truth

- Current local working tree and Git history.
- `origin` (`raederhans/Practicum`) and `teammate` (`ZhiyuanZhaoMicheal/Practicum`) remote refs after a fresh fetch.
- Public demo linked from repository metadata or documentation.

## Stages

- [x] Stage 1: Map local repository structure, history, branches, and data footprint.
- [x] Stage 2: Refresh and map teammate repository structure, history, and released/demo surfaces.
- [x] Stage 3: Compare commits, files, code responsibilities, data dependencies, and user-visible features.
- [x] Stage 4: Produce a ranked gap report and a non-destructive synchronization sequence.
- [x] Stage 5: Create an isolated full-history reference clone and verify ancestry, authorship, and donor boundaries.
- [x] Stage 6: Build machine-readable data, source, and canonical-result manifests.
- [x] Stage 7: Create the personal-project branch and import only reviewed dashboard, Stage 2, and Stage 3 slices.
- [x] Stage 8: Acquire publicly obtainable upstream data and run bounded regeneration experiments.
- [x] Stage 9: Normalize geographic scope and published metrics, then verify analysis and dashboard behavior.
- [x] Stage 10: Review, create local Lore commits, and leave public push/Pages deployment behind an explicit production gate.
- [x] Stage 11: Define an aggregate-only public product, MIT code-license boundary, third-party notices, and a fail-closed data policy.
- [x] Stage 12: Implement and test a standalone public website that remains useful without restricted source data or fine-grained derived artifacts.
- [x] Stage 13: Produce a clean allowlisted release tree, scan both source and built output, and verify every public route in a real browser.
- [ ] Stage 14: Publish the clean tree to a new GitHub repository, enable GitHub Pages, deploy an independent Vercel project, and verify live assets and routes.

## Acceptance criteria

- Every material gap is backed by a file, commit, or live-site observation.
- Missing data is distinguished from missing code and generated output.
- User-authored work is separated from teammate-authored additions where Git evidence permits.
- The recommended sync order states what to preserve, selectively import, clean-room rewrite, or leave out.
- No teammate-authored code, prose, images, or generated assets are copied into a republished personal project without explicit reuse permission and agreed attribution/license terms.
- Import allowlists, protected paths, data gates, and deployment gates are concrete enough to execute without guessing.
- Unknowns and evidence limits are explicit.
- The personal branch preserves all protected local analysis paths and imports no broad donor directory by accident.
- Every acquired dataset has an official source, retrieval method, license/redistribution classification, checksum or stable identifier, and reproducibility status.
- Canonical scope and result values are asserted by automated checks and used consistently by the dashboard build.
- The dashboard installs from its lockfile, builds locally, and passes route/static-asset smoke checks before any deployment action.
- The public repository and its complete new history contain no raw EAGLE-I files, fine-grained probability grids, facility-coordinate exports, local model artifacts, credentials, or parent-project analysis data.
- The public website works with only event-level public metadata, methodology, attribution, and aggregated/descriptive results; unavailable layers use an explicit withheld state rather than mock or fallback data.
- MIT applies only to original public-site code and owned documentation; datasets and third-party materials are separately excluded or attributed in the public documentation.
- Vercel uses a root-relative build, GitHub Pages uses its repository subpath, and both live surfaces pass root, emitted-asset, route, and browser-console checks.

## Non-goals

- No merge, rebase, cherry-pick, push, deployment, branch cleanup, or production change in this audit.
- During implementation, no push or public Pages deployment occurs without a final explicit production-release decision.
- No claim of authorship where the Git history does not support it.
- No copying of unpublished or unavailable data.
- No legal conclusion beyond the repository evidence and GitHub's general licensing guidance.
- No rewrite, deletion, or reuse of the existing public `Practicum` repository history as the release vehicle; publication uses a new clean repository.

## Risks and constraints

- The teammate may have omitted large or sensitive data files from GitHub.
- The demo may be built from a commit or environment not represented by the public repository.
- Fresh Git evidence shows only the current worktree; no additional worktree is available as a synchronization source.
- Repository history may contain merges or squashes that limit precise authorship attribution.
- Neither public tree contains a repository-level license; absent separate permission, teammate-authored material is reference-only.
- On 2026-08-05 the user confirmed that teammate reuse permission exists; attribution and provenance still remain required project records.
- Some official data sources require credentials, institutional access, rate-limited APIs, or acceptance of terms and therefore may remain only partially reproducible.
- Large downloads and regeneration jobs require a single live-process owner, stable logs, isolated caches, and explicit stop conditions.
- A public notice cannot cure unauthorized redistribution. Restricted or uncertain data must stay outside the new Git history, Vite inputs, `dist`, and Vercel upload boundary.
