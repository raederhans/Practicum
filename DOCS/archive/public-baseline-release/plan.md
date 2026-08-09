# Public Baseline Release Plan

## Goal

Publish the completed Phase 1 Generalization Autopsy and Phase 2 Failure Atlas as one aggregate-only, reproducible public baseline on GitHub Pages and Vercel.

## Scope

- Release source: `project/nightlight-public` at monorepo commit `cce74cfe09c067a6d58e031fa59ad51a2dc38d03`.
- Public history: the separate clean repository `raederhans/nightlight-disaster-dashboard`.
- Hosts: GitHub Pages and Vercel.
- Release method: validate locally, update a release branch in the clean public repository, verify its Vercel Preview, fast-forward the exact commit to public `main`, then verify the resulting GitHub Pages and Vercel Production artifacts byte-for-byte and in real browsers.

## Sources of truth

- Monorepo source commit `cce74cfe09c067a6d58e031fa59ad51a2dc38d03`.
- Existing clean public baseline `raederhans/nightlight-disaster-dashboard@c0ab511ac106cf5bc99f65d602afe3a8d4f71d85`.
- `project/nightlight-public/scripts/verify-public.mjs`.
- `project/nightlight-public/scripts/release-manifest.mjs`.
- `DOCS/archive/personal-project-sync-audit/{plan,context,task}.md`.

## Stages

- [x] Stage 1: Confirm repository, worktree, authentication, deployment, and restricted-data boundaries.
- [x] Stage 2: Materialize the exact public source into the clean standalone repository and run local release gates.
- [x] Stage 3: Commit and push the standalone release to GitHub `main`; verify Actions and GitHub Pages.
- [x] Stage 4: Push the standalone commit to a release branch and validate its protected Vercel Preview through cloud logs, headers, and authenticated asset checks; move the real-browser gate to public Production because Preview SSO cannot be automated without platform proxy rewriting.
- [x] Stage 5: Release the exact verified commit through the Git-linked Vercel Production build and record exact deployment evidence.

## Acceptance criteria

- The standalone repository contains only the reviewed public application and its public release support files.
- Restricted/raw EAGLE-I data, facility coordinates, probability grids, model files, credentials, and reconstructable fine-grained derivatives are absent.
- Local tests, root-path build, GitHub Pages base-path build, public-boundary verifier, and release manifest pass.
- GitHub public `main`, the successful Pages workflow, and the Pages deployment resolve to the new standalone release commit.
- A Vercel Preview for the same standalone release passes cloud tests, security-header checks, and authenticated manifest/hash verification; the real-browser gate moves to public Production when Preview SSO prevents a clean same-origin browser session.
- Vercel Production is built from the same exact public commit, passes the cloud validation again, and emits the same reviewed manifest and asset bytes.
- The stable GitHub Pages and Vercel URLs return the Phase 1 and Phase 2 experience without runtime external requests.

## Non-goals

- Do not deploy from or push the original `Practicum` history as the website release history.
- Do not publish raw or restricted data.
- Do not add a backend, runtime API, analytics, accounts, new product features, or new dependencies.
- Do not change the Phase 1 or Phase 2 scientific claims during release.
- Do not clean the retained Phase 1 source worktree or the user's untracked research archive.

## Risks and constraints

- The original public `Practicum` history contains previously exposed partner-restricted files and must remain excluded from deployment.
- GitHub Pages uses the repository base path while Vercel uses `/`; both builds must be verified independently.
- Vercel CLI `58.5.1` is installed globally for release operations without modifying project dependencies.
- Production release is allowed only after the Preview is tied to the intended public commit and passes the available live verification gates.
