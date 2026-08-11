# Nightlight Disaster Observatory

A standalone, aggregate-only personal research portfolio about disaster recovery, nighttime lights, and recorded electricity outages.

## Public boundary

This repository intentionally does not publish raw outage records, time-series extracts, facility locations, pixel-level surfaces, reversible fine-grained tables, or fitted model artifacts. The Atlas contains only the original study queue reduced to event names, years, broad locations, hazard families, and one-decimal centers.

The site does not request external fonts, scripts, map tiles, analytics services, or application data at runtime. An optional research log is off by default, stays inside the current tab after explicit opt-in, and can be viewed, exported, or cleared locally without a network request.

## Local use

Requires Node.js 20 or newer.

```sh
npm ci
npm run validate
npm run preview
```

`npm run validate` runs unit tests, builds the production bundle, writes `dist/release-manifest.json`, and verifies the public boundary plus every manifest hash.

## Deployment

- Vercel builds at `/` using `vercel.json`.
- GitHub Pages sets `VITE_BASE_PATH` to `/<repository-name>/` in the pinned workflow.
- Hash routing keeps every route compatible with static hosting.

## License and attribution

Original code and documentation in this public edition are MIT licensed. Dataset rights and attribution are separate; see `CREDITS.md`, `DATA_POLICY.md`, and `THIRD_PARTY_NOTICES.md`.
