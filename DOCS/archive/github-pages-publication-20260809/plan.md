# GitHub Pages Publication Plan

## Goal

Enable workflow-driven GitHub Pages for `raederhans/Practicum`, deploy the public observatory from exact `main`, verify the live artifact and routes, and leave local/remote Git plus task records consistent.

## Scope

- Confirm exact local/remote SHA, repository visibility, authentication, Pages state, and workflow contract.
- Enable Pages with `build_type=workflow` through GitHub's supported REST endpoint.
- Trigger a fresh `workflow_dispatch` run of `.github/workflows/deploy-dashboard.yml` at exact `main`.
- Monitor the single build/deploy run to completion and capture its SHA, URL, jobs, and conclusion.
- Verify the authoritative Pages API state, HTTPS endpoint, public release manifest, asset reachability, hash routes, CSP metadata, and browser console/network behavior.
- Preserve all user-owned untracked research and avoid changing scientific claims or publication content.
- Reconcile the worktree registry, archive this task record after successful verification, commit documentation only, and synchronize `main`/`origin/main`.

## Sources of truth

- `git rev-parse HEAD`, `git rev-parse origin/main`, and `git status --short --branch`.
- GitHub Pages REST API and the official `build_type=workflow` contract.
- GitHub Actions run metadata and job logs for the fresh exact-main dispatch.
- Live HTTP responses and the deployed `release-manifest.json`.
- Browser smoke evidence for the five hash routes.

## Stages

- [x] Stage 1: establish publication authority and capture the pre-publication snapshot.
- [x] Stage 2: enable workflow-driven Pages and verify the authoritative Pages settings.
- [x] Stage 3: dispatch and monitor one fresh exact-main workflow run under a single live-process owner.
- [x] Stage 4: verify the live URL, manifest, assets, routes, security metadata, and browser console/network state.
- [x] Stage 5: review the release result, reconcile records, and prepare the documentation-only closeout boundary.

## Acceptance criteria

- The Pages REST endpoint returns `200`, `build_type=workflow`, the expected `html_url`, and HTTPS enforcement.
- A fresh workflow run for exact starting SHA `70dae02bcf23200cb247e840fa9f7200ebb07f5e` completes with successful build and deploy jobs.
- The live root, release manifest, and all manifest-listed files are reachable over HTTPS.
- The five Vue hash routes load in the deployed app without page errors, failed same-origin requests, or unexpected console errors.
- The live HTML retains the public boundary metadata, including CSP and no-referrer directives.
- No product, scientific-method, restricted-data, or user-owned WIP content is changed.
- Local `main` and `origin/main` are synchronized after the documentation closeout, with the preserved research directory still untracked.

## Non-goals

- Do not create a custom domain, analytics integration, production database, or Vercel deployment.
- Do not rewrite history, force-push, change repository visibility, or publish restricted/private artifacts.
- Do not claim full scientific reproduction or a new independent exact-SHA scientific sign-off.
- Do not edit or stage `DOCS/archive/personal-project-evolution-research/`.

## Risks and controls

- External publication is irreversible in the practical sense of public exposure, but it is explicitly authorized by the user; the deploy artifact remains constrained by the existing public-boundary verifier.
- Only the primary agent owns the workflow dispatch and monitoring lane; no duplicate dispatch or concurrent Pages build will be started.
- Hash routing is required because GitHub Pages does not provide arbitrary SPA fallback routing; the app already uses `createWebHashHistory`.
- The starting product SHA has fresh local and GitHub build evidence, but the final publication claim will be limited to the deployed public artifact and automated gates.
