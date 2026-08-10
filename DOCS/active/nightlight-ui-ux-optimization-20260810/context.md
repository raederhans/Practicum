# Nightlight UI/UX Optimization Context

## Current facts

- Primary checkout: `C:\Users\raede\Desktop\essay help master\Practicum`.
- Pre-baseline state: `main == origin/main == d0ef4c33a837b2e7a04b59f0d2d24ee24d304f9d`.
- Existing retained worktree `C:\Users\raede\.codex\worktrees\591a\Practicum` is unrelated and excluded from all inspection, testing, integration, and cleanup.
- Protected untracked WIP: `DOCS/archive/personal-project-evolution-research/`.
- Current owned changes before baseline commit: development CSP repair, matching static-shell tests, and the UI/UX audit report.
- Primary task is the only integration owner. Child tasks own implementation slices and must stop at `ready-for-integration`.

## Live-process ownership

| Owner | Command/workdir | Resource | Log/output | Rule |
| --- | --- | --- | --- | --- |
| Primary task | `npm run dev -- --host 127.0.0.1 --port 5173 --strictPort` from `project/nightlight-public` | `127.0.0.1:5173` | `C:/Users/raede/.codex/visualizations/2026/08/10/019fec05-dfa3-7861-a020-c504de44cf5f/nightlight-public-runtime/vite-dev-fixed.log` | Remains running for the user; children must not poll/stop/retry/interpret it |
| Phase A | lane-owned dev/preview/test command | port 5181 and isolated temp/log root | lane context/final response | A is sole owner |
| Phase B | lane-owned dev/preview/test command | port 5182 and isolated temp/log root | lane context/final response | B is sole owner |
| Phase C | lane-owned dev/preview/test command | port 5183 and isolated temp/log root | lane context/final response | C is sole owner |
| Phase D | lane-owned dev/preview/test command | port 5184 and isolated temp/log root | lane context/final response | D is sole owner |

Each child must verify its port is free and its process identity before starting. It must stop only its own process and preserve final commands/exit codes. Full validation, shared browser matrix, final build/dist interpretation, and port 5173 remain primary-owner duties.

## Evidence already available

- Development CSP runtime HTTP 200 with nonce and without `unsafe-inline`.
- Targeted compatibility tests: 26/26.
- Production build and `verify:public -- --require-dist`: passed before parallel dispatch.
- Full `npm run validate`: 165/169 tests reached pass; four proxy-evaluation tests blocked by missing `DOCS/active/p2-p3-solo-evidence-performance-20260810/p2-evidence.md`.
- UI audit: 15 screenshots, five routes plus Atlas interactions, 391 px measurements, contrast samples, source review, and current official guidance.

## Decisions

- Use four project worktrees from one exact local baseline so independent tasks do not mutate the saved project checkout.
- Give `main.css` exclusively to Phase D; A-C own markup/logic and publish class-hook requirements.
- Integration order is sequential even though implementation is parallel.
- No child task receives Git integration authority; the primary task owns staging, commits after the baseline, refs, merges, pushes, registry, and cleanup.

## Dispatch identities

Exact shared baseline: `codex/nightlight-ui-ux-base@962431b680845a91b0b5b96807c77630dc82dd89`.

| Phase | Thread/client identity | Worktree | Dispatch state |
| --- | --- | --- | --- |
| A | `019fec40-8293-7eb3-b03c-224ddcd85e6f` | `C:\Users\raede\.codex\worktrees\f00e\Practicum` | active; confirmed initial repository and browser-verification work |
| B | `019fec40-8293-7eb3-b03c-226d54ef8486` | `C:\Users\raede\.codex\worktrees\fd8e\Practicum` | active; confirmed initial hierarchy and disclosure-contract work |
| C | `019fec42-d0b0-7c92-9ed3-15176c8c3785` | `C:\Users\raede\.codex\worktrees\2a26\Practicum` | active; confirmed initial Atlas workflow, URL-state, and mobile-view work |
| D | `019fec40-b5ad-72b3-8be4-212c26dd35eb` | `C:\Users\raede\.codex\worktrees\feae\Practicum` | active; confirmed initial CSS-contract and baseline work |

## Handoff requirements for every child

Return: exact base/HEAD, changed files, diff summary, new class hooks, behavior evidence, test commands and exit codes, browser evidence, live-process cleanup state, unverified risks, file/semantic overlap, and recommended integration method. Do not return only “done.”
