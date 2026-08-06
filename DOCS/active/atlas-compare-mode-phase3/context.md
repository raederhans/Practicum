# Atlas Compare Mode Phase 3 Context

## Current truth

- Integration owner: primary agent in `C:\Users\raede\Desktop\essay help master\Practicum`.
- Branch/starting commit: `codex/personal-project-sync` at `c3bde868bd57841ad20cf25ba2eae7c041e0c852`.
- Public source: `project/nightlight-public/`, a static Vue 3/Vite application with no runtime network access.
- Atlas currently owns one selected event, a 25-event public metadata index, broad one-decimal centers, and nine reviewed Evidence Passports.
- `evidencePassportManifest.json` is hash-bound reviewed source truth and must not change for Compare Mode.
- The public verifier uses an exact source/test file allowlist. Any new domain/test file must be explicitly admitted.
- Existing tracked state is clean; the only unrelated workspace item is the user's untracked `DOCS/archive/personal-project-evolution-research/`.
- No repository-native Phase 3 plan existed before this task; the baseline record only recommended Compare Mode after designing its scientific boundary.

## Decisions and deviations

| Date | Decision | Impact |
| --- | --- | --- |
| 2026-08-06 | Keep comparison inside Atlas rather than adding a sixth route. | Reuse the existing event selection, visual language, route tests, and public shell. |
| 2026-08-06 | Use deterministic rules for all runtime facts and warnings. | No DeepSeek key or LLM output enters the static client; results remain reproducible and testable. |
| 2026-08-06 | Treat hazard family as primary compatibility context. | Different hazard families always receive a prominent cross-category warning. |
| 2026-08-06 | Use year gap, rounded broad-center distance, Passport coverage, and matching component states as independent numeric summaries. | Numbers change with selection but cannot be added into a recovery or observability ranking. |
| 2026-08-06 | Reject radar charts and composite compatibility scores. | Avoid implied overall area/quality and preserve accessible text plus paired component tracks. |
| 2026-08-06 | Curated presets carry authored notes while factual status remains rule-generated. | Editorial guidance is distinguishable from computed evidence. |
| 2026-08-06 | Defer analytics integration until post-implementation research. | Preserve `connect-src 'none'`, avoid collecting visitors before a privacy/measurement decision, and keep this phase deployable offline. |
| 2026-08-06 | Domain TDD passed after an expected missing-module red run. | `compareEvents.js` now owns category-first peer repair, compatibility language, four independent summaries, missing-evidence behavior, and four presets; no product template has changed yet. |
| 2026-08-06 | Architecture review rejected year-gap and broad-center-distance summary cards. | Broad centers remain orientation-only and year proximity is not compatibility. Cards now count reviewed/comparable/exact/different evidence categories without summing points. A second red-green TDD cycle locked the refinement. |
| 2026-08-06 | UX review recommended a two-mode Atlas with independent comparison state. | Default `Explore one event` preserves the existing map/index; `Compare events` uses separate left/right IDs, grouped native selects, preset buttons, one live summary, and a responsive category ledger. |
| 2026-08-06 | Compare Mode production code passed focused tests, compile, manifest generation, and source/dist verification. | The next gate is fresh full validation followed by real-browser interaction and responsive inspection. |
| 2026-08-06 | Full validation and real-browser inspection passed. | The 95-test suite, Root/Pages builds, three responsive viewports, all evidence states, console/network capture, screenshots, and live-resource cleanup are complete. |
| 2026-08-06 | Defer analytics code, but recommend Umami Cloud for a later opt-in measurement phase. | Vercel Hobby Web Analytics cannot record the required custom events, while Umami documents a free personal-project tier, a sub-2KB tracker, SPA navigation, and custom events. Either choice would still require changing CSP, footer claims, privacy disclosure, deployment configuration, and verification. |
| 2026-08-06 | Final review found duplicate live-region semantics and an incomplete native radio group. | Both defects were reproduced with failing tests, fixed by removing `role=status` and adding a common radio `name`, then verified with static tests and real arrow-key browser interaction. |
| 2026-08-06 | Directed-pair coverage replaced the earlier unordered-pair test. | The maintained suite now proves all 600 Event A/Event B orders, 25 self-comparison rejections, 25 deterministic peer repairs, and reverse-order summary symmetry. |

## Live process ownership

| Process | Owner | Log path | State |
| --- | --- | --- | --- |
| Targeted/full tests and builds | Primary agent | `cache/atlas-compare-mode-phase3/validation.log` | Complete: 95/95, Root and Pages builds/verifiers passed; Root `dist` restored |
| Browser smoke | Primary agent | Codex tool transcript plus `cache/atlas-compare-mode-phase3/mobile.png` and `desktop.png` | Complete: 375/768/1024/1440 passed on final Root `dist`; one explicit-or-implicit live region; arrow-key radio grouping; session closed; exact preview tree stopped; port 4174 free |

## Analytics research outcome

### Recommendation

Keep Phase 3 offline and analytics-free. If comprehension proxies are approved as a separate phase, use an environment-gated Umami Cloud integration so the tracker appears only in explicitly measured production builds. Do not reuse or expose any DeepSeek key; an LLM is unnecessary for event collection or interpretation.

Why Umami is the better later fit:

- Umami's official documentation describes a tracker under 2KB, SPA route support, no cookies/fingerprinting/personal data, and custom click events.
- Umami Cloud documents a free Hobby tier suitable for personal projects and can receive events from both GitHub Pages and Vercel.
- Vercel Web Analytics is privacy-oriented and free for up to 50,000 monthly Hobby events, but Hobby does not include custom events and the integration is Vercel-specific.
- Cloudflare Web Analytics is free and privacy-first, but its current official FAQ says custom events are not supported, so it cannot measure Compare Mode interactions.
- GitHub repository traffic is useful as a zero-code background signal, but it only exposes repository visits/clones and a 14-day window; it cannot measure Atlas comprehension.

Official references:

- https://docs.umami.is/docs
- https://docs.umami.is/docs/track-events
- https://docs.umami.is/docs/cloud/faq
- https://vercel.com/docs/analytics/limits-and-pricing
- https://vercel.com/docs/analytics/custom-events
- https://developers.cloudflare.com/web-analytics/faq/
- https://docs.github.com/en/repositories/viewing-activity-and-data-for-your-repository/viewing-traffic-to-a-repository

### Minimal privacy-preserving event contract

| Event | Allowed properties | Purpose |
| --- | --- | --- |
| `compare_open` | none | Measures whether Atlas visitors discover Compare Mode. |
| `compare_preset_select` | `preset_key` from the four fixed public presets | Measures which guided story starts exploration. |
| `compare_manual_pair` | coarse `hazard_relation` and `passport_coverage` only | Measures movement from guided examples to arbitrary exploration without sending event names or locations. |
| `compare_boundary_seen` | none | Intersection-based proxy that the scientific boundary reached the viewport; not proof it was read or understood. |
| `compare_helpfulness_vote` | `answer=yes|no` | Optional explicit one-click self-report; still not a formal comprehension test. |

Never send event IDs, location text, free-form input, URLs with personal query parameters, device identifiers, or a stable user ID.

### Preset baseline versus real results

The current automated baseline is legitimate preset evidence: all four presets select the intended pair, the same-storm preset yields `2/2`, `5/5`, `5/5`, `0/5`, the same-place cross-hazard preset yields `2/2`, `5/5`, `4/5`, `1/5`, and required warnings appear. These are deterministic product results, not visitor analytics.

A future dashboard may define target hypotheses for discovery, manual-pair continuation, boundary exposure, and helpfulness votes. Those targets must be labeled `target` until real traffic exists; the product must never ship seeded or fabricated observed counts.

## Handoff

Implementation is complete. Do not edit the Passport manifest, add network calls, expose an API key, create a total score, or install analytics during final review. Keep all final mutations in the current integration worktree; delegated agents remain read-only reviewers.

## Next step

Phase 3 is locally integrated and verified. A later release may push and deploy this candidate only through the repository's separate release gates. Analytics remains an optional Phase 4 product/privacy decision, not hidden follow-up work in this phase.
