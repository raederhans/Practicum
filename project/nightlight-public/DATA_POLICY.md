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

## Source rights and attribution

The aggregated outage-related results were independently processed from ORNL EAGLE-I Recorded Electricity Outages, DOI `10.6084/m9.figshare.24237376`:

https://figshare.com/articles/dataset/The_Environment_for_Analysis_of_Geo-Located_Energy_Information_s_Recorded_Electricity_Outages_2014-2022/24237376

The source dataset is licensed under CC BY 4.0:

https://creativecommons.org/licenses/by/4.0/

The public edition does not sublicense the source dataset under MIT. Neither ORNL nor the U.S. Department of Energy endorses this project or its conclusions.

The Evidence Passport is also a non-reversible quality assessment derived from NASA Black Marble acquisition coverage, OpenStreetMap-derived contextual coverage, U.S. Census Bureau 2022 ACS 5-year/TIGER-Line covariates for the U.S. events, and the WorldPop Turkey 2020 population layer for Hatay. Their individual source and attribution terms remain in force; no raster, point record, facility record, query result, census row, geometry, or raw population layer is included in the public application. See `THIRD_PARTY_NOTICES.md` for exact products and source links.

## Enforcement

`npm run verify:public` fails closed on prohibited data and model formats, restricted artifact names, data directories, credential-shaped text, runtime network calls, and structural paths outside the release allowlist. A production verification also compares every built file against `dist/release-manifest.json` by relative path, byte length, and SHA-256.

The artifact verifiers also reject a public metric or Evidence Passport when it lacks required source lineage, drifts from the reviewed manifest hash, uses an unreviewed component value or band, duplicates an event, adds a dedicated overall-score field, or introduces prohibited raw or reversible fields before the site can build a release.
