# P2 User Understanding and Accessibility Plan

Archive status: integrated as `e7e95be` and deployed in release `2a47f36`.

## Goal

Verify the deployed and local Nightlight public application for interpretation, keyboard, focus, navigation, semantic-alternative, contrast, and responsive risks; repair only reproducible issues inside `project/nightlight-public/**`; and provide a small executable real-user study instrument without inventing participants or findings.

## Scope

- Product and tests: `project/nightlight-public/**` only.
- Lane-local records: this directory only.
- Read-only evidence may include the deployed GitHub Pages application, repository history, workflow configuration, and archived prior validation records.

## Non-goals

- No new primary route unless current evidence proves it necessary.
- No fabricated participants, observations, quotes, success rates, or findings.
- No modeling or data-manifest changes, dashboard changes, workflow changes, production settings, deployment, or Git integration.
- No edits to the worktree registry, umbrella task, or `DOCS/archive/personal-project-evolution-research/`.

## Phases

1. Establish repository, deployment, and process-ownership evidence.
2. Audit deployed and local source for interpretation and accessibility risks.
3. Reproduce any material issue in a real browser at 320, 373, 375, and 768 CSS pixels.
4. Add focused regression coverage before each product repair where practical.
5. Implement the smallest evidence-backed repair and an executable blank real-user study protocol/instrument.
6. Run targeted tests, complete public validation, production preview browser checks, cleanup, review, and handoff.

## Acceptance criteria

- Every product change maps to a reproducible source or browser finding.
- Keyboard users can identify the active route and reach the new route content with visible focus after SPA navigation.
- Complex visuals retain semantic text/table alternatives and Atlas comparison states remain understandable without color alone.
- The application has no page-level horizontal overflow at 320, 373, 375, or 768 CSS pixels on all existing routes.
- Analysis admission/readiness is never presented as recovery outcome, resilience, event quality, or a newly computed overall score/rank.
- The study protocol defines participants, setup, tasks, prompts, success/error measures, stop rules, note-taking, and a genuinely blank results structure.
- Fresh scoped tests and the supported public validation pass, or any gap is recorded exactly.
- All lane-owned browser/server processes are stopped and generated artifacts are removed before handoff.

## Risks

- Automated/browser inspection cannot substitute for a real screen-reader session or actual participant evidence.
- The deployed site can differ from this detached local base; findings must distinguish live behavior from local-source behavior.
- Long-running preview/browser processes must remain single-owner and use a verified-free dedicated port. Port 5174 was already owned by an unrelated Hackathon process, so this lane uses 43189.
