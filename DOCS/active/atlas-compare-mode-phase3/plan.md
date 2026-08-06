# Atlas Compare Mode Phase 3 Plan

## Goal

Extend the existing public Atlas with an honest, category-first comparison mode. A visitor may choose any two distinct public events and see how their public context and reviewed Evidence Passport components differ without turning the result into a recovery ranking, severity ranking, or total observability score.

## User decisions locked for this phase

- Keep Compare Mode inside the existing Atlas route.
- Allow any two distinct events from the current 25-event public index.
- Put hazard family, measurement limits, and compatibility warnings before numeric detail.
- Use only coverage counts as compact navigation aids; do not make value agreement the visual spine.
- Generate summaries with deterministic rules; use no runtime LLM or exposed API key.
- Include several curated preset comparisons with short authored notes.
- Defer real-user comprehension testing; research a minimal privacy-respecting web-analytics option only after implementation and verification.

## First-principles comparison contract

- This mode compares public metadata and the availability/state of reviewed evidence. It does not compare recovery outcomes.
- Hazard-family match is the primary compatibility signal. Broad-region match, international-context mismatch, year gap, and Evidence Passport coverage are secondary context.
- The original candidate used four independent numeric summaries: year gap, approximate broad-center distance, reviewed Passport coverage, and matching component states when both events have reviewed Passports.
- Compare Mode does not compute a new total, average, rank, or composite shape/area. The displayed admission band was assigned upstream from a weighted sum of the five workflow-rule outputs and must be labeled as such.
- Approximate distance is calculated only from the already-public one-decimal event centers and is rounded to avoid false precision.
- Missing Passport evidence remains `not assessed`; no values, states, or compatibility claims are imputed.
- Compatibility language describes whether the comparison is easier or harder to interpret. It never declares two disasters equivalent.

### Post-architecture refinement

The initial candidate summaries included year gap and approximate broad-center distance. The architecture review rejected those two numbers because event centers are published only as map-orientation references and year proximity does not establish measurement compatibility. A pre-audit implementation replaced them with reviewed/comparable/exact/different counts. The scientific audit then retired exact/different headline counts because agreement between rule-bin values does not establish measurement equivalence. The current design keeps only reviewed-Passport and paired-row coverage counts, followed by a component ledger and explicit measurement limits.

## Scope

1. Lock the domain contract with focused failing tests.
2. Add dependency-free comparison rules and curated presets as pure public-domain logic.
3. Add grouped event selectors, swap control, preset buttons, compatibility callout, dynamic numeric summaries, and component-by-component comparison inside `AtlasView.vue`.
4. Preserve the single-event Atlas and Evidence Passport behavior.
5. Extend the fail-closed public file allowlist and static accessibility contracts.
6. Run targeted tests, the full public validation, both deployment-base builds, responsive browser smoke, and a final scientific/security review.
7. After the product work is complete, research lightweight privacy-respecting analytics for later user-understanding proxies; do not add a runtime network tool in this phase.

## Preset comparison stories

- Same storm, two Florida references: Hurricane Ian at Charlotte Harbor and Fort Myers.
- Same hazard, different evidence readiness: Hurricane Irma at Miami and Hurricane Michael at Panama City.
- Same place, different hazards: Hurricane Maria and the Puerto Rico Earthquake at the same broad San Juan reference.
- Same hazard, international context boundary: Puerto Rico Earthquake and Turkey Earthquake.

## Acceptance criteria

- Both selectors expose all 25 events grouped by hazard family and prevent a self-comparison.
- Selecting from the existing Atlas index updates Event A; Event B stays stable unless it would duplicate Event A, in which case a deterministic category-first peer is chosen.
- Every pair produces a text compatibility status and explicit caveat; cross-hazard and international-context comparisons are visibly warned.
- Dynamic summaries update for every pair and use `—` rather than invented component values.
- Paired component evidence appears only when both Passports exist; otherwise the interface explains exactly which event is unassessed.
- Four curated presets select the expected pairs and expose an authored interpretive note without changing the rule-generated facts.
- Native controls, visible focus, logical keyboard order, `aria-live` updates, text-plus-color status, and no horizontal overflow work at 375, 768, 1024, and 1440 px.
- No new dependency, route, backend, API, runtime network request, external font, raw data, restricted field, overall score, leaderboard, or LLM credential enters source or dist.
- Targeted tests demonstrate red before implementation and green afterward; full `npm run validate`, GitHub Pages base-path build/verifier, and root build/verifier pass.

## Non-goals

- No recovery, severity, resilience, fairness, causal, or policy-performance comparison.
- No recommendation engine and no claim that a preset is the statistically best comparison.
- No public expansion beyond the current 25-event index or nine reviewed Passports.
- No modification of the reviewed Evidence Passport manifest or its canonical hash.
- No deployment or remote push during this implementation phase.
- No production analytics or real-user study in this phase.

## Risks and stop conditions

- Stop if a proposed number requires restricted detail, a new unreviewed aggregate, or implies a total score.
- Stop if arbitrary comparison cannot clearly distinguish metadata-only, partial-evidence, and paired-evidence states.
- Stop if a third identical test failure shows the assumed contract or environment is wrong.
- Keep the user's untracked `DOCS/archive/personal-project-evolution-research/` untouched.

## Scientific audit remediation addendum

The independent post-implementation audit changed the release gate without rewriting the original delivery history. The remediation phase must:

- remove exact/different match counts from the headline summary and live-region announcement;
- treat paired component values as descriptive v1 rule-bin rows, never as a similarity percentage;
- fail closed when Passport versions, component IDs, order, maxima, points, or statuses do not match the reviewed schema;
- separate the display event type from a documented hazard-family field;
- state that per-event measurement-frame equivalence is not established by the public artifact;
- disclose that the displayed readiness band was assigned upstream from a weighted sum, while Compare Mode computes no new total;
- label presets as editorial examples rather than a representative sample;
- preserve the private provenance gate: no public claim that the private inputs were independently recomputed until restricted-environment tests complete with zero skips.

Public deployment remains blocked until the four High findings have either been closed with evidence or, for private provenance, explicitly satisfied by the restricted verification gate.
