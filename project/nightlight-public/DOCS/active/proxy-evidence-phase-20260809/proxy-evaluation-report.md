# Nightlight Public App: Research-Supported Proxy Evaluation Report

## Evaluation status

This is a **research-supported proxy evaluation** of a local, uncommitted candidate based on base commit `223fc653dba2768dad99df9d032beaedd9234d6a`, evaluated on 2026-08-09. It combines source inspection, existing automated tests, prior deployed-browser evidence, current local production-browser checks, and adapted communication-review criteria.

It is not participant validation. No participant session, interview, task-completion rate, think-aloud observation, screen-reader-user result, or scientific-comprehension finding exists in this report. No final CDC CCI score is reported.

## Allowed conclusions

- The tested local candidate satisfies the bounded technical contracts enumerated below for titles, headings, native control labels, keyboard operation, visible focus, route orientation, semantic alternatives, polite status messaging, and reflow at 320, 373, 375, and 768 CSS pixels.
- The current public wording exposes several evidence-backed clarity characteristics: early purpose statements, headings and chunks, explicit known/unknown boundaries, and contextual explanations for R², AUC, readiness bands, `Not assessed`, and `Unavailable`.
- The adapted CDC items below can be used as a solo structured proxy checklist and as prompts for revision.
- The current review found and repaired two concrete focus-order defects: initial navigation scrolling changed Chromium's first Tab target, and the first SPA route activation did not always focus the new page heading.
- The public artifact makes the analysis-admission/readiness versus recovery-outcome boundary inspectable in source, semantic text, tables, and Compare Mode.

## Prohibited conclusions

- Do not claim that users find the site clear, usable, intuitive, understandable, or easy to navigate.
- Do not claim that users understand the scientific distinction between analysis readiness and real recovery outcomes.
- Do not call this usability validation, moderated usability testing, participant research, audience confirmation, or a screen-reader audit.
- Do not claim WCAG 2.2 conformance, certification, or coverage of every disability, user agent, zoom mode, input method, or assistive technology.
- Do not publish a CDC Clear Communication Index score from this self-review.
- Do not claim that R² is future-event accuracy, AUC is calibrated recovery transport, or an Evidence Passport is an outcome, resilience, severity, event-quality, fairness, policy, causal, or ranking measure.
- Do not claim that `Not assessed` or `Unavailable` means zero, bad data, failure, or worse recovery.
- Do not treat passing software checks as validation of the underlying scientific method, private inputs, or real recovery outcomes.

## Evidence layers

| Evidence layer | What this phase can assess | What remains outside the evidence |
| --- | --- | --- |
| Technical accessibility contract | Markup, titles, focus order, visible focus, keyboard reachability, labels, status regions, semantic alternatives, tested contrast pairs, and bounded reflow | Full WCAG conformance, assistive-technology interoperability, every browser, every zoom/text-spacing combination, and disability-specific usability |
| Information clarity proxy | Whether visible content has prominent purpose, headings/chunks, familiar or explained terms, known/unknown statements, and numerical context | Whether the intended audience actually finds the material clear or can use it without assistance |
| Scientific understanding | Source text can expose the intended interpretation and prohibited claims | **Scientific understanding remains unknown** without evidence from people in the intended audience |

## Evaluation target and method

- Target: the local production build derived from base `223fc653dba2768dad99df9d032beaedd9234d6a` plus the uncommitted P2 proxy/focus patch.
- Existing automated baseline: 10 Vitest files, 119/119 tests.
- New TDD evidence: proxy/report tests failed because the report and deferred status were missing; focus tests failed against the prior `hasRenderedRoute` and `scrollIntoView` implementation; the focused suites passed after the minimal repair.
- Local production preview: `npm run preview -- --host 127.0.0.1 --port 43231 --strictPort` from the public-app root.
- Browser: Chromium through Playwright CLI, isolated P2 sessions.
- Viewports: 320, 373, 375, and 768 CSS pixels, height 900.
- Routes: Overview, Study Atlas, Findings, Methods, and Credits / Policy; Atlas Compare Mode was checked separately at each width.
- Static review: current public source, artifacts, tests, policy boundaries, and the current Web Interface Guidelines checklist.
- Communication proxy: only the CDC items that fit a static analytical website; no composite or final score.

## Authoritative external basis

- [W3C WCAG 2.2](https://www.w3.org/TR/WCAG22/) provides testable criteria for non-text alternatives, contrast, 320 CSS-pixel reflow, keyboard operation, bypass blocks, page titles, headings/labels, visible focus, and status messages. W3C also states that accessibility evaluation combines automated testing and human evaluation and that the guidelines do not address every user need.
- [W3C Understanding Focus Order](https://www.w3.org/WAI/WCAG22/Understanding/focus-order.html) says sequential focus should preserve meaning and operability; static elements may receive focus when they do not create a confusing order.
- [W3C Understanding Headings and Labels](https://www.w3.org/WAI/WCAG22/Understanding/headings-and-labels) says headings and labels should describe topic or purpose so people can orient themselves.
- [W3C cognitive accessibility guidance](https://www.w3.org/TR/coga-usable/) recommends clear words, short sentences and blocks, clear structure, visible labels, explanations for unfamiliar terms, and alternatives/context for numerical concepts. It also explicitly recommends testing with real users; that evidence is unavailable here.
- [CDC Clear Communication Index](https://www.cdc.gov/ccindex/) describes a research-based tool for developing and assessing public communication materials.
- [CDC guidance for using the Index](https://www.cdc.gov/ccindex/tool/how-to-use.html) permits self-study and clarity review, but finished-material guidance calls for at least two independent reviewers, recommends that developers not score their own materials, and recommends testing with the primary audience. Therefore this report adapts selected items without assigning a final score.
- [GOV.UK moderated usability testing guidance](https://www.gov.uk/service-manual/user-research/using-moderated-usability-testing) defines the method as observing participants attempt tasks and asking them to think aloud. This phase did not perform that method.

## Judgment matrix

| Judgment | Product evidence | External basis | Confidence | Limitations | Prohibited product claim |
| --- | --- | --- | --- | --- | --- |
| Tested page orientation contract passes | Five routes expose one H1, unique route title after asynchronous settlement, `aria-current="page"`, and visible active navigation in 20/20 route-width checks | WCAG 2.4.2; WCAG 2.4.6; W3C headings/labels guidance | High | Chromium plus source/tests only; not full assistive-technology coverage | “Every user can orient themselves” |
| Tested focus order and route focus pass after repair | Fresh session: BODY → Skip link → identity → Overview → Study Atlas; Enter moved focus to the visible Atlas H1. `scrollLeft` replaced `scrollIntoView`, and route-path tracking replaced the unreliable first-render flag | WCAG 2.4.1; WCAG 2.4.3; WCAG 2.4.7; W3C Focus Order | High | One browser engine and one representative route transition | “Keyboard usability is validated for all users” |
| Native-control and visible-focus contract passes | Atlas radios, search, select, and buttons were keyboard-reachable; browser-computed focus outline was 2px solid `rgb(232, 255, 117)` with a 4px offset; no unlabeled input/select/textarea was found in the 20 route checks | WCAG 2.1.1; WCAG 2.4.7; Web Interface Guidelines native-control and focus rules | High | Does not test speech input, switch access, or every control state | “All input modalities are accessible” |
| Bounded reflow contract passes | Five routes × four widths produced 20/20 checks with no page-level horizontal overflow; Compare Mode added 4/4 passes | WCAG 1.4.10 identifies 320 CSS pixels as the vertical-content reflow reference | High | Height fixed at 900; no 200%/400% zoom or custom text-spacing run | “The site conforms to all reflow and resize requirements” |
| Tested text and focus-color pairs meet the minimum ratios used by this audit | Fresh calculation from current CSS: faint/raised 4.80:1, faint/ink 5.06:1, muted/ink 8.34:1, paper/ink 15.75:1, acid/ink 17.33:1 | WCAG 1.4.3 requires 4.5:1 for normal text; WCAG 2.4.7 requires visible focus | Medium | Not a pixel-level inventory of every transparency, state, SVG mark, or composited background | “All colors and non-text components are WCAG-conformant” |
| Semantic-alternative contract is present | Current source and accessibility snapshots expose SVG `title`/`desc`, figure captions, tables, text summaries, labels, native controls, and polite atomic Compare status | WCAG 1.1.1; WCAG 1.3.1; WCAG 4.1.3 | High | Presence does not establish screen-reader quality or equivalent comprehension | “Screen-reader users understand every visual” |
| Main-message and structure proxy is met | Overview opens with purpose and publication boundary; routes use one H1 plus descriptive section headings; content is split into sections, lists, cards, tables, and short explanatory blocks | CDC adapted main-message/information-design items; W3C COGA clear structure and chunks | Medium | Solo structural judgment; prominence and clarity were not confirmed by the target audience | “The main message is clear to users” |
| Known/unknown and state-of-science proxy is met | Findings separates in-sample description, held-out damage ranking, and unavailable recovery transport; Atlas lists known and unknown comparison conditions; Methods treats withholding/unavailability explicitly | CDC adapted state-of-science item; W3C COGA clear and unambiguous content | Medium | Scientific correctness of underlying private analysis is outside this public-app review | “The scientific method is validated” |
| Numerical-context proxy is met | R² is labeled explanatory/in-sample and not future accuracy; AUC is labeled held-out damage ranking, below the 0.50 reference, not recovery transport or calibrated prediction; component points retain their own maxima and do not form a new Compare score | CDC adapted numbers/context item; W3C COGA alternatives for numerical concepts | Medium | Real readers may still misunderstand the metrics; no comprehension evidence exists | “Users understand R² and AUC” |
| Evidence-state wording avoids isolated high/low risk labels | Readiness labels are paired with definitions, supported/unsupported claims, component states, and explicit `Not assessed`/`Unavailable` explanations | CDC adapted risk/context principle; W3C COGA clear words and explanations | Medium | Terms such as `Sensitivity-only` remain technical and may still require audience evidence | “Readiness labels are intuitive” |
| Information clarity has inspectable supporting characteristics | The adapted checklist rows above are source-backed and repeatable | CDC Index and W3C COGA | Medium | This is an information clarity proxy, not an observed clarity outcome | “The website is proven clear or easy to use” |
| Scientific understanding remains unknown | No human evidence exists; source/tests can check wording but not mental models | W3C COGA recommends real-user involvement; GOV.UK defines participant observation for moderated testing | High confidence in the evidence gap | The owner has explicitly chosen a proxy path suitable for a personal project | “Users correctly distinguish readiness from recovery” |

## Adapted CDC CCI solo checklist

No final CDC CCI score is reported. `Met` below means the feature was located in current source/browser evidence by the project owner/agent; it is not an independently reviewed CDC score and not a usability result.

| Adapted item | Proxy judgment | Current evidence | Limitation |
| --- | --- | --- | --- |
| Main message is prominent and early | Met | Overview's first content block states the study purpose and public/aggregate boundary; each route opens with an H1 and short framing text | No audience confirmation of what they identify as the main message |
| Headings, lists, and chunks expose structure | Met | One H1 per route; descriptive H2/H3 sections; evidence tables, cards, lists, and captions | Structure presence does not prove scanning success |
| Language is familiar or technical terms are explained | Partly met | `Not assessed`, `Unavailable`, Evidence Passport, R², and AUC receive nearby explanations and prohibited interpretations | `Sensitivity-only`, admission, transport, and calibration remain specialist language |
| State of science says what is known and unknown | Met | Findings role matrix and Atlas measurement-boundary sections separate admitted, withheld, unavailable, known, and unknown material | Public source cannot independently reproduce withheld/private analysis |
| Numbers include meaning and context | Met | R², AUC, component maxima, sample/reference context, and unsupported claims are visible | No evidence that nontechnical readers can explain the values |
| Risk/evidence states are not isolated high/low labels | Met | Readiness bands and component states include definitions and explicit non-outcome boundaries | Readiness wording may still be perceived as ordinal without human evidence |

## Current public claim-evidence matrix

| Claim ID | Visible claim and location | Evidence source | Supported interpretation | Prohibited interpretation | Status |
| --- | --- | --- | --- | --- | --- |
| `passport-purpose` | Atlas: “analysis admission heuristic” | Reviewed Evidence Passport artifact and manifest | Whether the project can inspect an event under the declared public rules | Recovery, severity, resilience, fairness, policy effect, event quality, or rank | Supported |
| `not-assessed` | Atlas: missing reviewed public evidence, not zero or worse recovery | Absence of reviewed Passport for that event | Public assessment is unavailable in v1 | Zero evidence, bad data, failure, or worse recovery | Supported |
| `compare-scope` | Atlas Compare: does not compare recovery outcomes and computes no new total/average/rank | Pairwise public component comparison plus measurement-boundary text | Inspect public context and evidence-state differences | Equivalence, similarity score, total observability, or outcome ranking | Supported |
| `r2-role` | Findings: R² 0.7603 is explanatory, in-sample, fixed-control | Reviewed aggregate generalization artifact | Description within the specified observed sample | Future-event accuracy, causal effect, or recovery forecast | Supported |
| `auc-role` | Findings: AUC 0.4814 is held-out damage ranking below the 0.50 reference | Reviewed aggregate generalization artifact | Performance on the declared held-out damage-ranking task | Recovery transport, calibrated probability, readiness, or community rank | Supported |
| `withholding` | Methods/Credits: unavailable values are withheld rather than replaced | Public boundary policy and artifact publication status | Absence is represented explicitly | Demo substitute, zero, or proof of poor quality | Supported |

## Repairs justified by this proxy audit

1. **Initial Tab order:** the initial `scrollIntoView()` call made Chromium start sequential focus at the item after the active Overview link, so the first Tab landed on Study Atlas instead of the Skip link. The repair centers the active item by assigning the navigation container's `scrollLeft`, which preserves the mobile visibility contract without moving the sequential focus start point.
2. **First SPA route focus:** `hasRenderedRoute` depended on an initial Transition `after-enter` hook that does not run consistently without an appearing transition. The first navigation could therefore be mistaken for the first render and leave focus on the activating link. The repair tracks the path whose content has entered and focuses the new H1 whenever a different route finishes entering.

No other product-code change was justified. Existing semantic alternatives, clarity boundaries, labels, status regions, contrast palette, responsive layout, and scientific claim limits were preserved.

## Verification summary

- Baseline after clean dependency install: 10 test files, 119/119 passed.
- Focus-contract TDD RED: 2 expected failures and 27 passes against the prior implementation.
- Focus-contract GREEN: 29/29 focused tests passed.
- Final fresh browser focus flow: BODY → Skip link → identity → Overview → Study Atlas; Enter focused the visible Study Atlas H1; subsequent Atlas controls showed a visible 2px focus outline.
- Final route matrix: 20 checks, 0 failures.
- Final Compare matrix: 4 checks, 0 failures.
- Final browser diagnostics: 0 console warnings/errors, 0 request failures, 0 responses at 400 or above; recorded requests were local `127.0.0.1` assets with 200/304 responses.
- Complete project gate: `npm run validate` passed 11 test files and 123/123 tests, built the production candidate, and passed `verify:public -- --require-dist`.
- Post-cleanup gate: generated browser/log/`dist` artifacts were removed; `npm test` again passed 123/123 and `npm run verify:public` passed with `dist` absent.

## Remaining limitations

- No real-user or intended-audience evidence exists; clarity and scientific understanding are not validated.
- No manual screen-reader, speech-input, switch-access, browser-family, high zoom, custom text-spacing, or OS high-contrast matrix was run.
- The audit is bounded to current source, current tests, Chromium, four widths, and selected contrast pairs.
- AI/adversarial review is only planned for the next phase and, if executed, must remain labeled non-human.
- This local candidate is not committed, integrated, pushed, deployed, or release-admitted.
