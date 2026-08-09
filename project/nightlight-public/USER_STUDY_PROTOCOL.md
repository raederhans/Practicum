# Nightlight Public App: Small Real-User Study Protocol

## Evidence status

This file is a protocol and blank instrument, not a findings report. **No sessions have been run, no participants have been recruited, and no usability or comprehension result is claimed here.** Add results only after a participant has consented and completed a recorded session under this protocol.

## Study purpose

Test whether first-time users can:

1. find the five existing routes and understand which route is active;
2. use the Atlas with a mouse, touch-equivalent viewport, or keyboard;
3. explain that an Evidence Passport describes **analysis admission/readiness**, not community recovery, event quality, resilience, severity, fairness, or policy performance;
4. explain that **Not assessed in v1** means no reviewed public Evidence Passport is available, not zero evidence, bad data, or worse recovery; and
5. distinguish the in-sample descriptive R² from the held-out-event damage-ranking AUC without turning either value into a recovery forecast.

The study evaluates understanding of the public interface. It does not validate the underlying scientific method, reproduce restricted inputs, measure real recovery outcomes, or authorize release.

## Participant and session criteria

- Planned small sample: 5 adults who have not worked on this application.
- Recruit 3 people without GIS/modeling expertise and 2 people who regularly read data or analytical products.
- Run at least 1 session keyboard-only. If a screen-reader participant is recruited, record the assistive technology and version; moderator simulation does not count as participant evidence.
- Required language: enough English to read the public interface and explain it in the participant's own words.
- Exclude the developer, supervisor, prior reviewers, and anyone already taught the intended interpretation.
- Obtain consent before timing or note-taking. Do not collect names, contact details, health details, or unrelated personal data in the results ledger.
- Target duration: 25–35 minutes. Compensation and recruitment method must be recorded before the first session, not filled in retrospectively.

## Moderator setup

Before each session:

1. Record the tested URL, deployed commit or release identifier, date, browser, browser version, viewport/device, input method, and any assistive technology.
2. Confirm the URL opens without console-visible failure. Start on the Overview route with browser zoom at 100% unless the participant normally uses another zoom level.
3. Clear only this site's session state if any exists. Do not change production settings or expose private analytical files.
4. Use the same task order unless the study lead pre-registers a counterbalanced order before recruitment.
5. Start the timer when the task is read. Stop it when the participant states an answer or asks to stop.
6. Ask the participant to think aloud, but do not define “recovery,” “analysis admission,” “readiness,” “Evidence Passport,” R², or AUC before the relevant task.

Read this opening script verbatim:

> We are testing the website, not you. Some labels may be unclear. Please say what you expect, what you notice, and what you think the page means. You may stop at any time. I will not teach the intended answer during a task, but I can repeat the task wording.

## Tasks and prompts

### Task 1 — Orientation and route navigation

Prompt: “Starting here, show me where you would go to understand (a) which events are included, (b) the main findings, and (c) the method. Tell me which page you are on after each move.”

Follow-up questions:

- What visual or announced cue tells you which route is active?
- After changing routes, where is keyboard focus?
- On a 1–5 scale, how confident are you that you can return to the Overview without using the browser Back button?

### Task 2 — One event and its Evidence Passport

Prompt: “In the Study Atlas, find Hurricane Maria. Explain what ‘Sensitivity-only’ means here and name one conclusion the page says you must not draw.”

Follow-up questions:

- Does the label describe the community, the disaster outcome, or the project's ability to analyze the event? Why?
- What does the ‘Unavailable’ observation-quality component mean in this interface?
- On a 1–5 scale, how confident are you in your explanation?

### Task 3 — Not assessed is not zero

Prompt: “Find Hurricane Matthew in Jacksonville. It is ‘Not assessed in v1.’ Explain what is known and unknown, and whether this label means worse recovery.”

Follow-up questions:

- What evidence would you need before making an analysis-admission claim?
- Did any wording make ‘Not assessed’ feel like a score or failure?
- On a 1–5 scale, how confident are you in your explanation?

### Task 4 — Compare evidence, not outcomes

Prompt: “Switch to Compare events. Compare Hurricane Maria with Hurricane Irma in Miami. Tell me what can be compared, what cannot be compared, and whether the page computes a new overall score or ranking.”

Follow-up questions:

- What does a same v1 rule-bin value establish?
- Is cross-event measurement comparability established?
- What would you change first if the boundary is hard to understand?

### Task 5 — Findings and metric roles

Prompt: “Open Findings. Explain the difference between R² 0.7603 and AUC 0.4814 to a classmate. What does each number support, and what does neither number prove about community recovery?”

Follow-up questions:

- Is the R² future-event accuracy?
- Is the AUC a calibrated recovery prediction?
- Which sentence or table most influenced your answer?
- On a 1–5 scale, how confident are you in your explanation?

## Success and error measures

Record measures per task, without correcting the participant until the task is coded:

| Measure | Coding rule |
| --- | --- |
| Completion | `independent`, `one neutral repeat`, `assisted`, `failed`, or `stopped` |
| Time | Seconds from the complete prompt to the participant's final answer; exclude consent and setup |
| Navigation errors | Wrong route, repeated backtrack, missed horizontally scrollable navigation, or lost focus; count observable instances |
| Interpretation accuracy | `correct`, `partial`, `incorrect`, or `no answer`, using the task-specific rules below |
| Confidence | Participant's 1–5 response; never infer it from behavior |
| Critical error | Record the exact task and paraphrased claim; do not “repair” it in the results ledger |

Task-specific success rules:

- Task 1 succeeds when all 3 destinations are found, the active route is identified, and keyboard focus is not lost after navigation.
- Task 2 succeeds only when the participant says the Passport concerns the project's analysis admission/readiness and rejects at least 1 recovery/outcome interpretation.
- Task 3 succeeds only when the participant treats Not assessed as missing reviewed public assessment, not a zero, failure, or recovery rank.
- Task 4 succeeds only when the participant keeps component rows separate, rejects a new total/rank/similarity score, and states that measurement comparability is not established.
- Task 5 succeeds only when the participant identifies R² as in-sample description, AUC as held-out damage ranking, and neither as a community-recovery result.

Critical interpretation errors:

- treating Observation-ready, Sensitivity-only, Repair-first, or Not assessed as a recovery outcome, resilience grade, disaster severity, event quality, fairness result, or policy score;
- interpreting Not assessed or Unavailable as zero;
- claiming Compare Mode creates a total, average, similarity score, or event rank;
- treating same component points as proof that events or measurement frames are equivalent;
- treating R² as future-event accuracy or AUC as recovery transport/calibration; or
- claiming the public app reproduces withheld private inputs.

Pre-registered review trigger: prioritize a wording or interaction repair when the same critical error appears independently in 2 or more completed sessions, or when a keyboard/assistive-technology blocker prevents 1 participant from completing a task. Do not convert this trigger into a success claim; it only starts review.

## Stop and escalation rules

Stop a task or session when:

- the participant withdraws consent, asks to stop, or shows distress;
- the tested build, route, or network fails for more than 5 minutes;
- the moderator accidentally teaches the intended answer before the participant responds;
- private/restricted material appears; close the page and notify the study owner without copying it into notes; or
- keyboard or assistive technology cannot proceed after 1 neutral retry.

Mark the affected task `stopped` or `invalidated`, state the reason, and do not score it as success or failure. A replacement session must use a new row; never overwrite or silently discard the stopped session.

## Moderator closing questions

Ask after all tasks:

1. In one sentence, what is this site for?
2. What is the most important limit a reader should remember?
3. Which label or number was hardest to interpret?
4. Where did you feel unsure about what to do next?
5. What is the smallest change that would make the site clearer?

Only after these questions may the moderator explain the intended analysis-admission and recovery-outcome boundary.

## Blank results structure

**Current results status: No sessions have been run. All tables below are intentionally blank.** Add one row only after a consented session. Use an anonymous session code; do not enter a participant's name.

### Session ledger

| Anonymous session code | Date | Tested URL + release | Participant group | Browser / viewport | Input / assistive technology | Consent recorded | Session status | Stop reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |

### Task observations

| Session code | Task | Completion | Time (s) | Navigation errors | Interpretation accuracy | Confidence (1–5) | Critical error? | Neutral observation / participant paraphrase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |

### Closing-question notes

| Session code | Site purpose | Most important limit | Hardest label/number | Navigation uncertainty | Smallest requested change |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

### Synthesis after sessions

Do not complete this section before sessions exist. Report counts with denominators, not unsupported percentages, and separate observation from inference.

| Item | Blank result |
| --- | --- |
| Completed / stopped / invalidated sessions |  |
| Independent task completions by task |  |
| Critical errors by type and task |  |
| Keyboard or assistive-technology blockers |  |
| Repeated navigation errors |  |
| Median task time by task |  |
| Confidence distribution by task |  |
| Evidence-backed repair candidates |  |
| Conflicting or inconclusive observations |  |
| Unverified risks and next study step |  |
