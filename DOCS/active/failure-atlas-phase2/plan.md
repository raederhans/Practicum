# Failure Atlas Phase 2 Plan

## Goal

Turn the existing public event atlas into a `Failure Atlas / Evidence Passport` MVP. A visitor should be able to select an event and first see whether that event is sufficiently observable for the project's analytical workflow, before interpreting model results.

This phase answers:

> What evidence is present, missing, or too weak to support cross-event analysis for this event?

It does **not** rank community recovery, disaster severity, resilience, fairness, causal effects, or policy performance.

## First-principles contract

- The public page is a static, build-time artifact for GitHub Pages and Vercel; it makes no runtime network requests.
- Public permission is narrower than local availability. Only non-reversible component assessments for already public event IDs may enter the web bundle.
- Do not publish raw counts, ratios, time series, facility records, coordinates beyond the existing broad event center, model-performance impact, training recommendations, or a single overall score.
- Component evidence is shown separately so readers can see why an event is ready, sensitivity-only, repair-first, or not assessed.
- A workflow band is an analysis-admission heuristic, not a measurement of recovery outcome or community quality.
- Events without reviewed readiness evidence are explicitly `not assessed in v1`; no mock or inferred passport is allowed.

## Scope

1. Lock and regenerate readiness source truth
   - Add focused tests for the current observation-quality thresholds and public mapping invariants.
   - Run the existing `score-readiness` pipeline under a single owner with logged output.
   - Review the regenerated diff and reject unrelated or outcome-derived fields.

2. Add `Public Evidence Passport Artifact v1`
   - Publish exactly the nine readiness events that map to the current 25-event public index.
   - Exclude `dorian_freeport` because it is not admitted to the public event index.
   - Include per-component points and maxima, workflow band, plain-language interpretation, supported/unsupported claims, source identity/hash, artifact version/date, and attribution status.
   - Exclude `event_count`, raw quality ratios, POI counts, `total_score`, model increment impact, and recommended training decisions.

3. Extend the existing Atlas route
   - Keep existing map, search, filtering, and selected-event behavior.
   - Add a semantic Evidence Passport panel for assessed events.
   - Add an explicit not-assessed state for the other sixteen public events.
   - Use text and structure, not color alone, for component status.

4. Enforce the public boundary
   - Add exact artifact schema/allowlist validation and negative tests.
   - Add a monorepo provenance test that canonicalizes LF/CRLF, binds the public artifact to the reviewed readiness CSV hash, validates the event mapping, and proves forbidden fields are absent.
   - Update data policy and third-party notices only where the new published derivative requires it.

## Acceptance criteria

- Exactly 9 reviewed passports and 16 explicit unassessed public events.
- No Dorian passport, overall score, raw count/ratio, outcome-derived recommendation, restricted source path, or runtime request in public source/dist.
- Every passport's five component scores match the regenerated canonical readiness source.
- Atlas search, filters, selection, keyboard behavior, responsive layout, and local-only behavior remain correct.
- Targeted tests fail before implementation and pass afterward.
- Full `npm run validate`, GitHub Pages base-path build/verifier, Vercel/root build/verifier, and production browser smoke pass.
- Final code review and first-principles simplification find no unresolved correctness blocker.

## Non-goals

- No new route, backend, API, account, live data feed, external map tile, or dependency.
- No expansion of the public event cohort.
- No facility or pixel-level evidence, daily outage/recovery sequence, raw imagery, or geographic grids.
- No total `Observability Score` and no comparison leaderboard.
- No remote push or production deployment in this phase.

## Stop conditions

Stop and report if publication requires reconstructable restricted detail, source lineage cannot be reproduced, the regenerated source changes unrelated modeling results, or three runs fail under the same assumption.
