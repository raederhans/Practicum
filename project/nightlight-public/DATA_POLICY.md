# Data Policy

## What this repository publishes

- Aggregate study scale: Stage 2 covers 25 events across 17 jurisdictions.
- Aggregate modeling scale: Stage 3 contains 1,002 ZIP-event observations across 22 events and 15 U.S. states.
- Descriptive model summaries: R² 0.7603, adjusted R² 0.7543, and n = 977.
- A value of 0.551 labeled only as descriptive sensitivity, with no causal or fairness conclusion.
- `Public Generalization Artifact v1`: a small allowlisted set of aggregate model-role metrics, including an explicitly labeled held-out-event damage-ranking result. Each value carries cohort, sample lock, validation design, source artifact identifier/version/SHA-256, quality, publication status, license, and attribution.
- `Public Evidence Passport Artifact v1`: five separate, coarse analysis-readiness components for nine reviewed events in the public index, plus an analysis-admission band and a SHA-256-bound reviewed manifest. Sixteen other public events are explicitly unassessed. Compare Mode computes no new total, average, or rank. The displayed admission band was assigned upstream from a weighted sum of the five workflow-rule outputs; neither the band nor any arithmetic over visible points is an event-quality, recovery, or outcome measure. Independent recomputation from the private inputs requires the restricted source environment.
- The original study's 25-event index, reduced to disaster name, year, broad location, hazard family, and one center rounded to one decimal.
- Methods, interpretation limits, credits, and licensing notices.

## What this repository does not publish

- Raw outage records or source CSV files.
- Time-series extracts or other temporal records.
- Facility names, facility locations, or facility-level analysis.
- Pixel-level surfaces, probability surfaces, rasters, or geographic grids.
- Reversible fine-grained tables, model files, or private intermediate outputs.
- Readiness input counts or ratios, a dedicated overall readiness-score field, model-increment impacts, or training-role recommendations.

No synthetic substitute or hidden fallback is used when these materials are absent. The interface states the boundary directly.

## Optional local research log

The app includes a narrowly allowlisted research log for two questions: which fixed evidence surfaces a consenting participant opens, and whether they select Atlas Explore or Compare. It is off by default and begins only after an explicit opt-in for the current browser tab.

- Storage is limited to `sessionStorage` for the current tab, with an in-memory fallback when browser storage is unavailable. Closing the tab, choosing **Stop and clear**, or clearing browser session data removes it.
- No cookie, account, participant identifier, cross-session identifier, IP address, user agent, referrer, exact URL/query, precise location, free text, model input, restricted source field, or timestamp is recorded.
- Only the fixed `surface_viewed` and `atlas_mode_selected` schemas are admitted. Unknown event names, extra properties, and values outside the explicit enum are rejected. There is no custom payload channel.
- The log is capped at 100 events. A participant can view the local list, export the same bounded JSON locally, clear events with one action, or withdraw consent and delete the session.
- The application has no analytics endpoint or transport. Nothing is sent to the project owner or a third party. An exported file leaves the tab only when the participant chooses what to do with it.

This is a research convenience, not representative usage evidence. Opt-in selection, small samples, and a page-open event do not establish comprehension or population behavior. A future server-side study would require a separate privacy, consent, retention, infrastructure-log, deletion, and research-design review; this release does not implement one.

## Consumer value and error contract

Public numeric consumers use `Public Aggregate Value schema v1`, enforced by `src/lib/aggregateValueContract.js` and its negative tests.

- `available` carries a finite number. Numeric `0` is valid only in this explicit state.
- `unavailable`, `not_assessed`, `not_applicable`, `suppressed`, `load_failure`, and `validation_failure` carry `null` plus a stable reason code.
- A missing, withheld, failed, or invalid value must never be coerced, defaulted, or serialized as numeric zero.
- Load and validation failures are operational states, not published measurements. Consumers must preserve the state or stop; they must not silently fall back to demo or mock content.

Evidence Passport component points are reviewed workflow-rule outputs, not replacements for missing measurements. A zero-point rule output must remain paired with its explicit component status and must not be reused as an available measured value under the aggregate value schema.

Schema changes require a version change, negative fixtures, exact public-allowlist updates, and a documented migration or explicit rejection of the older version. Breaking changes do not receive an indefinite compatibility fallback.

## Source freshness and failure contract spike

`Public Source Value schema v1` is a testable future-carrier boundary. It is not wired to an external source in this release. Each record owns a stable source id and explicitly carries `version`, `effectiveDate`, `retrievedAt`, and `validatedAt`; those four fields are complete only for a validated snapshot. The value state distinguishes:

- `available`: a validated finite value, including a legitimate zero.
- `stale`: the last validated finite value after `offline`, `rate-limited`, `auth-required`, or `source-failure`, visibly labeled with its effective date.
- `unavailable`, `offline`, `rate_limited`, `auth_required`, `source_failure`, and `validation_failure`: `null` value, null snapshot metadata, and an allowlisted reason code.

A stale snapshot is displayable only when it has complete validated metadata, an admitted failure cause, a visible **as of** label, and an effective date no more than 30 days before the explicit evaluation time. At day 31 it fails closed and must not be displayed as a value. A snapshot that failed validation is never eligible for stale display. No status may turn a missing or failed value into numeric zero.

The current adapter is bundled-static only. It validates reviewed in-memory records and returns `validation_failure` rather than leaking an invalid value. It does not acquire, retry, cache, or request data at runtime.

## Source rights and attribution

The aggregated outage-related results were independently processed from ORNL EAGLE-I Recorded Electricity Outages, DOI `10.6084/m9.figshare.24237376`:

https://figshare.com/articles/dataset/The_Environment_for_Analysis_of_Geo-Located_Energy_Information_s_Recorded_Electricity_Outages_2014-2022/24237376

The source dataset is licensed under CC BY 4.0:

https://creativecommons.org/licenses/by/4.0/

The public edition does not sublicense the source dataset under MIT. Neither ORNL nor the U.S. Department of Energy endorses this project or its conclusions.

The Evidence Passport is also a non-reversible quality assessment derived from NASA Black Marble acquisition coverage, OpenStreetMap-derived contextual coverage, U.S. Census Bureau 2022 ACS 5-year/TIGER-Line covariates for the U.S. events, and the WorldPop Turkey 2020 population layer for Hatay. Their individual source and attribution terms remain in force; no raster, point record, facility record, query result, census row, geometry, or raw population layer is included in the public application. See `THIRD_PARTY_NOTICES.md` for exact products and source links.

## Enforcement

`npm run verify:public` fails closed on prohibited data and model formats, restricted artifact names, data directories, credential-shaped text, runtime network calls, non-reviewed runtime dependencies, weakened HTML security metadata, oversized files, and structural paths outside the release allowlist. The release contract records `local-opt-in-only` analytics with no transport and no persistent identifier. A production verification also compares every built file against `dist/release-manifest.json` by relative path, byte length, SHA-256, base path, and static build contract.

The artifact verifiers also reject a public metric or Evidence Passport when it lacks required source lineage, drifts from the reviewed manifest hash, uses an unreviewed component value or band, duplicates an event, adds a dedicated overall-score field, or introduces prohibited raw or reversible fields before the site can build a release.
