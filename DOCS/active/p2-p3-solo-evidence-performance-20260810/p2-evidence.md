# P2 Independent Personal-Agent Evidence

## Conclusion and evidence class

This is an owner-run, research-supported proxy review of the public Nightlight application. The fixed repository target is `ca8292040a402eae1d2e461708a4cc912867efcb`, evaluated with the uncommitted evidence-gate files listed in the final handoff. The source review and initial automated baseline were performed on 2026-08-10 using Node `v22.23.0`, npm `11.18.0`, Vite `6.4.3`, Vitest `3.2.7`, and the five existing public routes. The exact Chromium version and browser matrix are recorded below after the live gate.

The evidence can support bounded statements about source wording, semantic structure, tested keyboard/focus behavior, tested reflow, tested browser diagnostics, and whether material claims expose an evidence boundary. It cannot establish that people understand, trust, or can successfully use the application.

**Scientific understanding remains unknown.** No participant was recruited or simulated. No participant session, interview, task-completion rate, think-aloud result, user quote, SUS score, manual screen-reader session, or manual assistive-technology result exists. No final CDC CCI score is reported.

## Official method basis

Only official or first-party material was used for the method boundary. All sources were accessed 2026-08-10.

| Source and status | Applicable use here | Boundary retained here |
| --- | --- | --- |
| [CDC: How to Use the Clear Communication Index](https://www.cdc.gov/ccindex/tool/how-to-use.html), current CDC web guidance, accessed 2026-08-10 | Provides prompts for main message, language, headings/chunks, known/unknown science, numbers, risk context, and independent review | CDC asks for independent reviewers when reviewing/revising material. This owner/AI pass is not independent review and therefore produces no CDC score or clearance claim. |
| [CDC Clear Communication Index](https://www.cdc.gov/ccindex/), current CDC tool landing page, accessed 2026-08-10 | Establishes that the Index is a research-based communication-development and assessment tool | It does not convert a developer self-review into user research or usability validation. |
| [USWDS Accessibility](https://designsystem.digital.gov/documentation/accessibility/), current USWDS documentation, accessed 2026-08-10 | Supports plain language, accurate headings/images/links, logical layout, automated plus manual testing, and broad-user testing as separate practices | This project uses the content and technical prompts only; it did not conduct the broad-user or manual assistive-technology testing USWDS recommends. |
| [WCAG 2.2](https://www.w3.org/TR/WCAG22/), W3C Recommendation dated 2023-10-05, accessed 2026-08-10 | Supplies testable criteria used by existing source/browser checks: non-text alternatives, structure, keyboard, bypass, page titles, focus, labels, status messages, and 320 CSS-pixel reflow | WCAG conformance applies to full pages and all required criteria. This bounded subset must not be reported as WCAG 2.2 conformance. |
| [WAI: Conformance Evaluation and Reports](https://www.w3.org/WAI/test-evaluate/conformance/), page updated 2023-08-01, accessed 2026-08-10 | Separates a conformance evaluation methodology from ad hoc checks and recommends involving users with disabilities | The current owner-run regression is neither WCAG-EM conformance evaluation nor disability-user evidence. |
| [WAI Cognitive Accessibility Guidance](https://www.w3.org/TR/coga-usable/), W3C Working Group Note, accessed 2026-08-10 | Supports clear words, short blocks, visible structure/labels, nearby explanations, and understandable number/context alternatives | The guidance also supports testing with real users; no such evidence is available in this personal-project phase. |

## Evaluation target and environment

| Field | Fixed value |
| --- | --- |
| Base revision | `ca8292040a402eae1d2e461708a4cc912867efcb` |
| Git state at start | Detached HEAD; clean index and worktree |
| Candidate scope | Base plus current uncommitted task record and directly corresponding evidence-gate test; no P2 product copy/UI change at this checkpoint |
| Install baseline | `npm ci`: 73 packages installed, 74 audited, 0 vulnerabilities |
| Automated baseline | `npm test`: 11 files, 123/123 tests passed before the new evidence gate |
| Routes | Overview, Study Atlas, Findings, Methods, Credits / Policy |
| Browser widths | 320, 373, 375, and 768 CSS pixels; fixed height 900 |
| Browser engine | Headed Chromium `151.0.7922.76` through isolated Playwright CLI session `practicum-p2-solo-20260810` |

## Automated semantic contract

Passing a row means only that the named contract is present in current source/tests or the bounded browser sample. It does not prove comprehension, task success, or assistive-technology interoperability.

| Contract | Current evidence | Gate state before live browser |
| --- | --- | --- |
| Five unique route titles | `src/router/routes.js`; `tests/routes.test.js` | Passed in 123/123 baseline |
| One programmatically focusable H1 per route | Five route views; `tests/static-shell.test.js` | Passed in baseline and 20/20 route-width checks; keyboard Atlas transition focused H1 |
| Skip-link and main target | `src/App.vue`; static-shell focus/navigation tests | Passed in baseline; fresh first Tab reached `#main-content` Skip link |
| Active-route state | `aria-current="page"` in navigation plus prior route tests | Correct route label in 20/20 live checks |
| Native labeled controls | Atlas radios, selects, and buttons; static-shell tests | No unlabeled input/select/textarea/button in 20/20 route checks; Compare entered by keyboard |
| Semantic alternatives for complex visuals | SVG title/description, captions, tables, and text summaries in route views | Named SVG alternatives/tables remained in relevant route snapshots; all images had `alt` attributes |
| Polite status regions | Three Atlas `aria-live="polite"` regions and atomic comparison status | Source test passed; Compare atomic polite summary present in 4/4 widths |
| Reflow without page-level horizontal overflow | Existing CSS/source contract | 20/20 route checks and 4/4 Compare checks passed at 320/373/375/768 CSS pixels |
| Claim fail-closed behavior | Generalization, Evidence Passport, comparison, and public-boundary tests | Passed in baseline |

## Material public claim-evidence matrix

Every row is searchable by exact visible wording or a uniquely identifying phrase. A `supported` gate status means the current public artifact and adjacent boundary text support only the narrow interpretation in that row; it is not independent scientific validation.

| Claim ID | Exact visible wording and route | Evidence source | Supported interpretation | Prohibited interpretation | Gate status |
| --- | --- | --- | --- | --- | --- |
| `claim-overview-purpose` | Overview H1: “Reading recovery in the dark.”; lead: “A two-stage study of disaster impacts, electricity outages, and changes in nighttime light” | `OverviewView.vue`, `STUDY_SUMMARY`, and adjacent research note | The public app presents an observational research narrative about nighttime-light changes after disasters | Direct measurement of household recovery, community resilience, causal recovery drivers, or validated user understanding | supported |
| `claim-stage-counts` | Overview metrics: “25 event studies”, “22 modeled events”, “17 jurisdictions” | `src/content/study.js` fixed public summary | Counts declared by the reviewed public summary for the stated study stages | Complete coverage of all disasters, current live coverage, or an independently reproduced scientific result | supported |
| `claim-r2-role` | Overview/Findings: “descriptive R², n = 977” and R² `0.7603` | Reviewed public generalization artifact and `STUDY_SUMMARY.descriptiveModel` | Description of variation within the specified analyzed fixed-control sample | Future-event accuracy, recovery forecast, causal effect, policy effect, or external validity | supported |
| `claim-auc-role` | Findings: held-out damage-ranking AUC `0.4814`, below a `0.50` reference | Reviewed aggregate `cross-event-logit-auc` record | Performance on the declared leave-one-event-out damage-ranking task | Recovery transport, calibrated recovery probability, future readiness, fairness, policy failure, or community rank | supported |
| `claim-sensitivity-role` | Findings: “Descriptive sensitivity” `0.551` | `FINDINGS_COPY` and reviewed public generalization artifact | A descriptive ratio under its stated analysis conditions | Causal mechanism, fairness conclusion, benefit, harm, or transport improvement | supported |
| `claim-passport-purpose` | Atlas: Evidence Passport is an “analysis admission heuristic” | Reviewed Evidence Passport artifact, manifest, version, and public component rules | Whether the project may inspect an event under the declared public analysis-admission rules | Recovery outcome, resilience, severity, event quality, fairness, policy performance, reliability, or rank | supported |
| `claim-not-assessed` | Atlas: “Not assessed” means no reviewed public Passport for the event | Explicit artifact absence plus the visible Atlas definition | Public v1 assessment is unavailable for that event | Zero, bad data, failure, worse recovery, or a negative score | supported |
| `claim-compare-scope` | Atlas Compare: “does not compare recovery outcomes” and computes no new total, average, score, or rank | Pairwise public component comparison and measurement-boundary text | Inspect differences in public context, component states, and evidence availability | Event equivalence, similarity, overall observability, total readiness, recovery comparison, or leaderboard | supported |
| `claim-withholding` | Findings/Methods/Credits: unavailable values remain withheld rather than being replaced | Public boundary policy and artifact `publicationStatus`/`withheldReason` | A value is intentionally not public or not available in the admitted artifact | Zero, demonstration substitute, proof of poor quality, or evidence of failure | supported |
| `claim-data-boundary` | Overview/Credits: “Fine-grained analytical layers are not published in this public edition.” | `DATA_BOUNDARY`, public scanner, CSP, and release-manifest boundary | The site intentionally publishes study-scale facts and reviewed aggregates only | Proof that private layers were independently audited, complete scientific reproducibility, or absence of all privacy/security risk | supported |

## Adapted CDC-style solo proxy

No final CDC CCI score is reported. The statuses below are owner-run prompts, not independent CDC scores and not usability results.

| Adapted item | Solo proxy status | Product evidence | Limitation |
| --- | --- | --- | --- |
| Main message appears early | met | Overview opens with the research framing, study stages, and public boundary | “Reading recovery” is metaphorical and may still be read as direct outcome measurement; no audience interpretation data exists |
| Headings, lists, and chunks expose structure | met | Each route has one H1 and descriptive section headings; claims use cards, tables, lists, captions, and short sections | Presence of structure does not prove scanning success or reading order in assistive technology |
| Familiar language or nearby explanation | partly met | R², AUC, `Not assessed`, `Unavailable`, Evidence Passport, and comparison limits have nearby definitions | “Sensitivity-only”, “analysis admission”, “transport”, “calibration”, and “fixed-control” remain specialist terms |
| Known and unknown science are separated | met | Findings and Atlas distinguish admitted, withheld, unavailable, descriptive, ranking, and unsupported claims | The public source cannot independently reproduce withheld/private analysis |
| Numbers include role, reference, and limitation | met | R² includes sample/role; AUC includes task/reference; components expose maxima/status definitions | No evidence shows nontechnical readers can explain these values correctly |
| Evidence/risk states are not isolated `high`/`low` labels | met | Readiness bands and components include definitions; `Not assessed`/`Unavailable` are explicit | Band ordering may still be mistaken for outcome quality without human evidence |

## Two-pass AI review

AI output is a heuristic review note, not a user quote, participant observation, comprehension result, or authority. The two passes below are **non-human**. They are not human research and do not satisfy CDC independent-review guidance. Pass B used a fixed hostile-reader checklist rather than Pass A's candidate list, but both were produced in one owner-agent session; therefore they are not blind or statistically independent evidence.

### Pass A — non-human clarity and terminology review

Prompt boundary: inspect the visible public wording and the matrix above for unexplained terms, missing definitions, weak known/unknown boundaries, and candidate overclaims. Do not invent a reader reaction and do not score the material.

| Candidate | Source reconciliation | Decision |
| --- | --- | --- |
| The H1 “Reading recovery in the dark” can sound like direct outcome measurement | The immediately following lead and research note say the signal is not household-level and not a verdict on resilience; the matrix prohibits direct recovery claims | Retain as a bounded metaphor; medium interpretation risk, not a source-backed contradiction |
| “What shapes the visible pace of recovery?” can sound causal | The following sentence changes the operative verb to “align” and the Findings copy explicitly says descriptive/not causal | Retain; mark causal extension prohibited |
| “Analysis admission”, “transport”, “calibration”, and “fixed-control” are specialist terms | Nearby role/unsupported-claim text reduces but cannot eliminate the burden | `partly met`; future plain-language refinement is optional, not evidence of a current failed contract |
| AUC `0.4814` below `0.50` could be read as universal failure | The visible role is a specific held-out damage-ranking design and explicitly not recovery transport/readiness | Supported only in the named task; no universal or policy conclusion |
| Readiness bands can appear ordinal and outcome-like | Atlas visibly defines them as analysis-admission states and Compare forbids total/rank computation | Retain with strict prohibited-claim gate; human interpretation remains unknown |

### Pass B — non-human adversarial review

Prompt boundary: attempt only these hostile misreadings from the same public source and matrix—readiness as recovery, absence as zero, R² as future accuracy, AUC as calibrated recovery transport, and Compare as a ranking. Do not use Pass A's decisions as the evaluation criterion.

| Hostile misreading | Counterevidence in product | Gate result |
| --- | --- | --- |
| “Observation-ready” means the event recovered well | Evidence Passport purpose and band definitions say analysis admission only; no outcome field is exposed | Rejected; any wording asserting recovery is blocked |
| `Not assessed`/`Unavailable` means zero, worse, or failed | The app explicitly maps these states to absent/withheld reviewed public evidence | Rejected; any zero/worse substitution is blocked |
| R² `0.7603` predicts future disaster recovery with 76% accuracy | The metric is labeled descriptive, in-sample, fixed-control, with n = 977 and an explicit unsupported future-accuracy claim | Rejected; any forecast wording is blocked |
| AUC `0.4814` is a calibrated recovery probability or transport result | The artifact labels it ranking, specifies the held-out damage-ranking design, and explicitly denies recovery transport/calibration | Rejected; any recovery-probability/transport wording is blocked |
| Compare produces the better event or an overall readiness ranking | Compare exposes paired components and availability only, with no computed total/average/score/rank | Rejected; any leaderboard/equivalence statement is blocked |

Reconciliation result: both non-human passes found interpretation risks already bounded by adjacent source text and fail-closed tests. They found no authoritative-source contradiction requiring a product copy change before the browser gate. This is a **P2 product-code measured no-op at the source-review checkpoint**, subject to fresh accessibility/browser regression.

## Claim audit and proxy release gate

The candidate passes the content portion of the proxy gate only if every material claim row remains `supported`, `withheld`, or `unavailable`, all automated/browser checks pass or remain explicit blockers, and no integration edit weakens these boundaries.

The release handoff:

- must not claim WCAG 2.2 conformance;
- must not claim usability validation;
- must not claim participant testing;
- must not claim manual assistive-technology validation;
- must not call either AI pass human, a participant, a user, independent audience evidence, or authority;
- must not turn R² into future-event accuracy, prediction percentage, causal effect, or recovery forecast;
- must not turn AUC into recovery transport, calibrated recovery probability, readiness, fairness, or community ranking;
- must not turn `Not assessed` or `Unavailable` into zero, failure, bad data, or worse recovery;
- must not turn an Evidence Passport/readiness band into recovery, resilience, severity, event quality, policy performance, fairness, or rank;
- must not turn passing software/browser checks into independent scientific validation or proof that intended users understand the material.

## Validation ledger and live evidence

| Check | Current result |
| --- | --- |
| Clean install | Passed; 0 vulnerabilities |
| Pre-gate automated baseline | Passed; 11 files, 123/123 tests |
| Evidence-gate RED | Expected 4/4 failures because this report did not yet exist |
| Evidence-gate GREEN | Passed; 1 file, 4/4 tests |
| Complete `npm run validate` | Passed; 11 files, 127/127 tests, production build, 11-file release manifest, and required source/dist public boundary |
| Headed route/width matrix | Passed; 20/20 route-width checks, unique expected title, one expected H1, active navigation, no unlabeled native control, and no page-level horizontal overflow |
| Headed keyboard/focus/Compare matrix | First Tab reached Skip link; keyboard Atlas navigation focused the H1 with 2px solid/4px-offset outline; Compare passed 4/4 widths with atomic polite status and no outcome/rank language |
| Browser console/network gate | 0 console warning/error, 0 page error, 0 failed request, and 0 HTTP response at 400 or above |

The browser probe required two calibration corrections before its final run: it first mistook the vertical scrollbar width for horizontal overflow and cross-element H1 text spacing for a missing title; it then read outgoing transition content before the requested route H1 entered. Those probe defects were corrected by comparing against `window.innerWidth` and waiting for the exact target H1/focus success signals. The final run had `failures: []`; no public product source was changed.

## Remaining limitations

- No real-user, intended-audience, participant, interview, task-success, or comprehension evidence exists.
- No manual screen-reader, speech-input, switch-access, browser-family, high zoom, custom text-spacing, reduced-data, OS high-contrast, or disability-user matrix was run.
- Automated source checks can confirm the presence of labels, alternatives, and status semantics, but not their quality in every assistive technology.
- The claim matrix traces current public artifacts and reviewed aggregates; it is not an independent scientific replication of the underlying private/restricted analysis.
- The same non-human executor produced both AI passes; fixed prompts reduce cross-contamination but do not make the passes independent reviewers.
