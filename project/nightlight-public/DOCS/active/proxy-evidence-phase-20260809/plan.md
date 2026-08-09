# Owner-Run Proxy Evidence and Release Gate Plan

## Status

Active next-phase plan for a personal project. Every result produced by this plan is a research-supported proxy or technical check, never participant validation.

## Goal

Give the project owner a repeatable way to check accessibility contracts, content clarity signals, and scientific claim boundaries without requiring participant recruitment. The process should catch regressions and unsupported wording while keeping three questions separate:

1. Does the interface satisfy the tested technical accessibility contract?
2. Does the content exhibit evidence-backed clarity characteristics?
3. Do real users actually understand the scientific meaning?

Only the first two can receive proxy evidence here. The third remains unknown unless optional future human research is performed.

## Scope

- Existing public routes and static public artifacts under `project/nightlight-public`.
- Reproducible semantic, keyboard, focus, reflow, contrast, console, network, and claim checks.
- A content-evidence matrix and an adapted subset of the CDC Clear Communication Index used as a solo structured proxy.
- Two-pass AI review used only to generate review candidates and adversarial counterexamples.
- A release gate that blocks unsupported claims or failed technical checks.

## Phases

### Phase 0 — Pin the evaluation target

1. Record the exact commit, tested URL, Node/browser versions, date, and viewport set.
2. Run `npm ci`, `npm test`, and `npm run verify:public` from this directory.
3. Stop if the target is dirty, the public boundary fails, or the evaluated build cannot be identified.

### Phase 1 — Automated semantic contract

1. Extend existing Vitest files before creating a new test when the responsibility overlaps.
2. Check unique page titles, one `h1` per route, descriptive labels, native controls, skip-link target, active-route state, text/table alternatives for complex visuals, and polite status regions.
3. Fail when a tested contract disappears. Passing means only that the markup contract is present.

### Phase 2 — Content evidence matrix

Create or update a matrix with one row per material public claim:

| Field | Required content |
| --- | --- |
| Claim ID and exact visible wording | Searchable text, not a paraphrase only |
| Route / component | Where a reader encounters it |
| Evidence source | Public artifact field, reviewed aggregate, policy, or explicit absence |
| Supported interpretation | The narrow conclusion the evidence permits |
| Prohibited interpretation | Recovery, causality, ranking, fairness, policy, or other unsupported extension |
| Status | `supported`, `withheld`, `unavailable`, `needs revision`, or `blocked` |
| Check | Test, static review, browser observation, or unresolved manual review |

Any row without an evidence source or explicit absence is a release blocker.

### Phase 3 — Adapted CDC CCI solo proxy

Use only items that fit this static analytical product:

- main message is visible and appears early;
- headings, lists, and chunks expose the information structure;
- visible language is familiar or technical terms are explained nearby;
- the state of the science says what is known and unknown;
- numbers include meaning, role, unit/reference, and limitation;
- risk or evidence states are not reduced to isolated `high`/`low` labels.

Record `met`, `partly met`, `not met`, or `not applicable` with a file/line or browser reference. Do **not** calculate or publish a CDC Index score: CDC finished-material guidance calls for independent reviewers, recommends that developers not score their own material, and recommends primary-audience testing.

### Phase 4 — Two-pass AI/adversarial review

1. Pass A receives the public text and the evidence matrix. It identifies unclear terms, missing definitions, and candidate overclaims.
2. Pass B receives the same public text plus a hostile-reader prompt, but not Pass A's conclusions. It tries to misread readiness as recovery, absence as zero, R² as future accuracy, and AUC as calibrated recovery transport.
3. The owner reconciles both outputs against source code and reviewed artifacts. AI output is a heuristic review note, not a user quote, participant observation, comprehension result, or authority.
4. An unresolved source-backed contradiction blocks release; an AI-only stylistic preference does not.

### Phase 5 — Accessibility regression and browser gate

1. Run `npm run validate`.
2. Start a strict-port production preview on a freshly checked high port.
3. In Chromium, check all five routes at 320, 373, 375, and 768 CSS pixels for page-level horizontal overflow, visible active navigation, unique titles, logical focus movement, visible focus, and clean console/network behavior.
4. Exercise Atlas Explore and Compare modes using keyboard input. Confirm semantic status text remains available without relying on color.
5. Stop the owned browser/server and remove generated logs, snapshots, and `dist` files before handoff.

### Phase 6 — Claim audit and release gate

A candidate is proxy-ready only when:

- source and tests identify the exact evaluated revision;
- automated semantic and browser checks pass or every failure is explicitly blocked;
- every material claim has a content-evidence row;
- adapted CDC items have evidence and no final CDC score is claimed;
- AI passes are labeled non-human and their disagreements are resolved against authoritative evidence;
- no wording turns readiness/admission into recovery, severity, resilience, event quality, fairness, policy performance, causality, or ranking;
- no wording turns `Not assessed` or `Unavailable` into zero;
- no wording turns R² into future-event accuracy or AUC into recovery transport/calibration;
- the report states that technical checks do not prove user comprehension or WCAG conformance.

## Acceptance criteria

- One command path reaches every automated test used by the gate.
- Browser commands, port, owner, viewport set, and teardown evidence are recorded.
- Each proxy judgment includes product evidence, external basis, confidence, limitation, and prohibited claim.
- No participant, interview, task-completion, think-aloud, screen-reader-user, or comprehension result is invented.
- The owner can repeat the complete gate without recruiting anyone or accessing restricted analytical inputs.
- Failures remain failures or explicit blockers; they are not converted into fallback claims.

## Non-goals

- No participant usability validation or claim that users understand the science.
- No WCAG 2.2 conformance certification or complete assistive-technology audit.
- No validation of the underlying scientific method, real recovery outcomes, causality, fairness, policy effect, or model transport.
- No replacement of independent editorial review with a self-assigned CDC score.
- No new primary route, dependency, production setting, deployment, or restricted-data publication.

## Optional escalation to human research

Human research remains optional for this personal project. Consider it later only if the owner chooses to support a broader public or institutional audience, receives a recurring accessibility or interpretation complaint, plans a high-stakes decision use, sees repeated unresolved disagreement across independent reviewers, or has the capacity to recruit representative participants ethically. Until then, retain the proxy label and the prohibited-claim boundary.
