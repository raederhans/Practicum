# Dashboard Performance and Architecture Plan

## Goal

Measure the dashboard's production bundle and route-loading boundaries, then make only a low-risk change whose value is supported by before/after evidence.

## Scope

- Owned: `project/nightlight-dashboard/**`.
- Measure the app shell plus home-route payload, the incremental map-route payload, and the approximately 803 kB MapLibre chunk.
- Preserve UI behavior, route behavior, map behavior, data semantics, and existing project conventions.

## Sources of truth

- Starting Git SHA: `a79921b2becb81388762b10a365744d014026198` (detached worktree).
- `src/router/index.js` for route boundaries.
- `src/views/MapView.vue` for the sole MapLibre import and map lifecycle.
- `vite.config.js` and fresh Vite production manifests for chunk boundaries.
- Fresh Vitest, production-build, and route-request evidence from this worktree.

## Stages

- [x] Stage 1: map current Vite, router, import, and MapLibre behavior.
- [x] Stage 2: capture reproducible baseline bundle and route metrics.
- [x] Stage 3: test the smallest credible improvement or record a measured product-code no-op.
- [x] Stage 4: run final tests/build/route smoke and compare before/after evidence.
- [x] Stage 5: stop lane-owned processes, remove generated artifacts, and hand off without Git mutations.

## Acceptance criteria

- Report minified and gzip bytes for the MapLibre chunk.
- Report the app-shell plus home-route payload and the incremental map-route payload.
- Prove whether the home route requests MapLibre.
- Report whether the Vite warning changes.
- Preserve existing behavior and pass scoped tests plus a production build.
- Leave no lane-owned listener or generated browser/test artifact behind.

## Non-goals

- Do not silence warnings by raising `chunkSizeWarningLimit`.
- Do not add dependencies or weaken map behavior.
- Do not edit public-app, scientific/data, workflow, shared coordination, registry, or production-setting paths.
- Do not stage, commit, change refs, merge, rebase, push, or change worktree topology.

## Risks and constraints

- MapLibre may already be correctly isolated from initial navigation; further splitting can add latency or lifecycle races without reducing map-route work.
- Build hashes alone are not stable identifiers; comparisons use manifest module identities and exact byte sizes.
- Ports `5174` and `5175` are owned by other tasks. Port `43189` is free but intentionally not reused.
