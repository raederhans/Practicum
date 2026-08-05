# Data Policy

## What this repository publishes

- Aggregate study scale: Stage 2 covers 25 events across 17 jurisdictions.
- Aggregate modeling scale: Stage 3 contains 1,002 ZIP-event observations across 22 events and 15 U.S. states.
- Descriptive model summaries: R² 0.7603, adjusted R² 0.7543, and n = 977.
- A value of 0.551 labeled only as descriptive sensitivity, with no causal or fairness conclusion.
- The original study's 25-event index, reduced to disaster name, year, broad location, hazard family, and one center rounded to one decimal.
- Methods, interpretation limits, credits, and licensing notices.

## What this repository does not publish

- Raw outage records or source CSV files.
- Time-series extracts or other temporal records.
- Facility names, facility locations, or facility-level analysis.
- Pixel-level surfaces, probability surfaces, rasters, or geographic grids.
- Reversible fine-grained tables, model files, or private intermediate outputs.

No synthetic substitute or hidden fallback is used when these materials are absent. The interface states the boundary directly.

## Source rights and attribution

The aggregated outage-related results were independently processed from ORNL EAGLE-I Recorded Electricity Outages, DOI `10.6084/m9.figshare.24237376`:

https://figshare.com/articles/dataset/The_Environment_for_Analysis_of_Geo-Located_Energy_Information_s_Recorded_Electricity_Outages_2014-2022/24237376

The source dataset is licensed under CC BY 4.0:

https://creativecommons.org/licenses/by/4.0/

The public edition does not sublicense the source dataset under MIT. Neither ORNL nor the U.S. Department of Energy endorses this project or its conclusions.

## Enforcement

`npm run verify:public` fails closed on prohibited data and model formats, restricted artifact names, data directories, credential-shaped text, runtime network calls, and structural paths outside the release allowlist. A production verification also compares every built file against `dist/release-manifest.json` by relative path, byte length, and SHA-256.
