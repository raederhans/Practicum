# Atlas Compare Mode Phase 3 Task

## Current status

Scientific audit remediation commit `a7a577e3960c8259459db9d2ec47513510868e25` is locally integrated into `codex/personal-project-sync`. The original implementation evidence remains historical. High findings H1-H3 are closed by repository evidence; H4 remains a restricted-environment provenance gate and continues to block public deployment.

## Checklist

- [x] Recover the prior roadmap and inspect the current Atlas/publication contract.
- [x] Lock user decisions, scientific boundaries, presets, and acceptance criteria.
- [x] Add focused failing domain tests for pair rules, summaries, missing evidence, full cohort, and presets.
- [x] Implement deterministic comparison rules and preset definitions.
- [x] Integrate Compare Mode into the existing Atlas.
- [x] Add responsive/accessibility styles without new dependencies.
- [x] Extend the fail-closed public allowlist and negative tests.
- [x] Run targeted and full public validation for root and Pages base paths.
- [x] Run responsive browser smoke and inspect console/network/layout behavior.
- [x] Complete code/scientific/security review and simplification pass.
- [x] Research lightweight web analytics for later comprehension proxies and record a recommendation.
- [x] Create a scoped local Lore commit and update registry/final facts.
- [x] Remove headline match-count scorecard framing and revise live-region semantics.
- [x] Add conservative measurement-frame boundary metadata without inventing per-event equivalence.
- [x] Make comparison schema handling fail closed for missing, duplicate, malformed, reordered, or future components.
- [x] Separate event display type from documented hazard family.
- [x] Disclose the upstream weighted readiness-band construction.
- [x] Label presets as editorial and non-representative.
- [x] Run focused/full/Root/Pages/browser verification.
- [x] Create a remediation Lore commit and integrate it into `codex/personal-project-sync`.
- [ ] Obtain restricted-environment private provenance verification with zero skips before public scientific release.

## Validation evidence

| Check | Result |
| --- | --- |
| Repository/worktree preflight | `codex/personal-project-sync@c3bde86`; retained Phase 1 worktree; unrelated untracked evolution research preserved. |
| Existing public tests | Pre-Phase-3 baseline was 85/85; the current Phase 3 candidate has fresh 95/95 full-suite evidence plus Root/Pages build verification. |
| UI/UX design-system query | Preserve dark editorial language; use visible focus, semantic native controls, responsive cards/tables, value labels, and no horizontal overflow. |
| Initial architecture review | Inline Atlas extension and one pure domain module are the smallest coherent boundary; no route or new artifact is required. |
| Domain TDD red | New six-test file failed 6/6 because `compareEvents.js` did not exist; failure matched the intended missing-feature boundary. |
| Domain TDD green | `tests/compare-events.test.js` passed 6/6; all 300 distinct pairs returned four finite-or-unavailable summaries and warnings. |
| Surface/boundary TDD red | Four assertions failed for missing Compare Mode controls/live summary and unallowlisted new domain/test paths. |
| Surface/boundary TDD green | Focused Compare/Atlas/public-boundary suite passed 62/62. |
| First production compile | Vite transformed 40 modules and emitted an 11-file manifest; public source/dist verifier passed. |
| Full public validation | `npm run validate` passed 95/95 tests. Root and Pages-base builds each emitted the expected 11-file manifest and passed source/dist verification; Root `dist` was restored. |
| Responsive browser smoke | Playwright passed at 375x812, 768x900, and 1440x900: 25 options per selector, four presets, four summaries, 44px controls, no overflow, no hidden critical text, no console/page/request failures, and no external requests. |
| Visual inspection | Mobile and desktop screenshots preserve the existing editorial Atlas language; numeric summaries lead the comparison and the five components remain visibly separate. |
| Live-resource cleanup | Playwright session `atlas-compare-phase3` closed; the exact preview process tree was stopped; port 4174 is free. |
| Analytics research | Do not install analytics in Phase 3. For a later opt-in phase, prefer Umami Cloud over Vercel Web Analytics for cross-host custom events; preserve a no-PII event schema and treat initial thresholds as hypotheses, not observed results. |
| Final-review TDD | A new accessibility assertion reproduced the duplicate implicit `role=status` live region, then passed after its removal. A second red-green cycle added the shared radio `name` required for native arrow-key grouping. |
| Exhaustive comparison coverage | The committed domain suite now locks all 600 directed A/B pairs, all 25 self-comparison rejections, all 25 deterministic category-first peer repairs, and reverse-order summary symmetry. |
| Independent code review | Two accessibility findings were reproduced and fixed with red-green evidence: the duplicate implicit live region and missing shared radio-group name. No correctness or comparison-domain defect remained. |
| Independent security/scientific review | Risk `LOW`; no blocking finding, secret, LLM/runtime network use, new dependency, restricted field, aggregate ranking, or source/dist boundary defect. Full dependency audit reported zero known vulnerabilities. |
| Final candidate verification | Fresh 95/95 suite; Root and Pages 11-file builds/verifiers; Root `dist` restored; browser smoke passed 375/768/1024/1440 with one live region, arrow-key radio grouping, zero console/page/request failures, and zero external requests. |
| Remediation TDD red | The first focused run failed 19 assertions for the deliberately absent fail-closed schema contract, measurement boundary, hazard-family split, and revised UI semantics; 24 pre-existing assertions still passed. |
| Remediation focused green | 49/49 across Compare rules, Evidence Passport, study taxonomy, and static shell. Negative inputs cover missing/undefined/duplicate/reordered/extra components, changed maxima, invalid status, empty arrays, and current-version plus future-version schema drift. All 72 directed pairs among the nine controlled reviewed Passports are explicitly locked as `paired-v1`. |
| Remediation full validation | Fresh 111/111 suite; Vite transformed 40 modules; Root build emitted 11 files and the source/dist verifier passed. |
| Root/Pages release boundary | Pages base `/Practicum/` and Root base each emitted 11 files and passed `verify:public -- --require-dist`; Root `dist` was restored. `npm audit --audit-level=low` reported 0 vulnerabilities. |
| Controlled artifact result | All nine current reviewed Passports pass the public artifact validator and the 600 directed-pair suite. The fail-closed findings apply to future/abnormal public-function inputs, not an observed error in the current nine controlled Passports. |
| Private provenance gate | `test_public_evidence_passport_provenance.py`: 2 passed, 1 skipped because the private readiness source is unavailable in this clone. This is not zero-skip provenance evidence and H4 remains High. |
| Remediation browser QA | Root build passed at 375/768/1024/1440 and 1280→640 200%-equivalent reflow with zero horizontal overflow. Measurement limits precede two coverage-only summaries; one live region; native radio arrow-key behavior; visible 2px focus; reduced-motion durations `0.01ms`; cross-hazard and missing-Passport states correct; no console warnings/errors or external resources. |
| Contrast spot check | New measurement-boundary and coverage-summary text measured 8.02:1 to 16.66:1 against the composited dark background. |
| Live-resource cleanup | Playwright session `atlas-remediation` closed; exact preview process stopped; port 4176 verified free. The verifier caught Playwright's temporary snapshot directory; those six owned snapshots and the empty directory were removed, then the full 111/111 validation passed. Screenshots are outside the repository under the task visualization directory. |
| Post-integration target validation | `codex/personal-project-sync@a7a577e`: fresh 111/111 tests, Root 11-file build, and source/dist public verifier passed in the target worktree. The unrelated untracked evolution-research directory remained untouched. |

## Open risks

- Component points are workflow evidence, not outcome measurements. Paired bars must never be described as recovery performance.
- One or both events may be unassessed; missing component comparisons remain visibly unavailable and have dedicated automated coverage.
- There is no real-user comprehension evidence. Browser smoke proves mechanics and readability constraints, not that visitors interpret the scientific boundary correctly.
- Analytics integration remains a separate product/privacy change because it would replace the current no-analytics and `connect-src 'none'` contract.
- The public clone cannot independently recompute the private readiness inputs; this remains a High release gate until restricted provenance tests complete with zero skips.
