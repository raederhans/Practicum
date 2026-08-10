# Authorized acquisition and solo evidence phase

## Goal

Convert the previous P1/P2/P3 closeout into two independently executable lanes while preserving scientific and public-claim boundaries:

1. Acquire or precisely receipt the authorized upstream inputs needed by the fail-closed reproducibility preflight.
2. Execute the solo-feasible public evidence plan, measure throttled runtime behavior, and resolve or document the GitHub Actions runtime annotation.

## Fixed baseline

- Integration branch: `main`
- Baseline: `ca8292040a402eae1d2e461708a4cc912867efcb`
- Remote baseline: `origin/main` at the same SHA when this phase started.
- Product release remains `e18b91a32cd92d379e4328889df1c0139f43ccee`; Pages run `31322447014` already succeeded.
- The three untracked files under `DOCS/archive/personal-project-evolution-research/` are user-owned and excluded.
- The registered `591a` worktree belongs to another task and is excluded.

## Lanes

### Lane A — P1 authorized source acquisition

- Task: `019fe950-a575-7b33-8f9b-dd65cca53bee` in `C:\Users\raede\.codex\worktrees\aefe\Practicum`.
- Owned task record: `DOCS/active/p1-authorized-source-acquisition-20260810/**`.
- Product ownership: acquisition manifests/scripts, reproducibility acquisition interfaces, and directly corresponding tests.
- Live ownership: Earth Engine/API calls, downloads, raw caches, export outputs, and acquisition logs.
- Required outcome: each of the seven full-upstream blockers becomes either verified evidence or a precise remaining blocker with one minimal next action.

### Lane B — P2/P3 solo evidence and delivery maintenance

- Task: `019fe950-adf8-7600-b531-7ae219bc3df0` in `C:\Users\raede\.codex\worktrees\0313\Practicum`.
- Owned task record: `DOCS/active/p2-p3-solo-evidence-performance-20260810/**`.
- Product ownership: `project/nightlight-public/**`, `project/nightlight-dashboard/**`, the dashboard deployment workflow, and directly corresponding tests.
- Live ownership: public/dashboard servers, browsers, ports, `dist`, performance logs, and runtime measurements.
- Required outcome: honest non-human P2 proxy evidence, a measured P3 change or no-op, and an official-evidence Actions decision.

## Sequence

1. Let both isolated worktrees start from the fixed baseline and record their process/data contracts before mutation.
2. Run the lanes in parallel because their files and live resources do not overlap.
3. Require both lanes to stop without staging, committing, pushing, deploying, or changing worktree topology.
4. Review each ready-for-integration package against claim, data, license, secret, and generated-artifact boundaries.
5. Integrate in dependency order: P1 evidence first, then P2/P3 public and workflow changes.
6. Run targeted tests, then aggregate Python/public/dashboard/build/boundary checks.
7. Create Lore-formatted commits, push `main`, observe CI/Pages only if product paths trigger deployment, update records, and clean task-owned resources.

## Data and authorization boundary

- User authorization permits attempting the acquisition, but it does not prove that local credentials, a Google Cloud project, billing, quota, or export destination already exist.
- Reuse only existing valid authentication after a non-secret preflight. Never print or commit tokens, private keys, account data, or credential files.
- Interactive login, terms acceptance, billing activation, cloud-project selection, or expanded cloud permissions remains a user-action gate.
- Use official and primary sources. Before a download/export, record identifier, license, spatial/temporal scope, expected size, quota/cost risk, and stop condition.
- Raw and large derived assets remain outside Git unless current repository policy explicitly requires and safely supports them. Commit receipts, checksums, metadata, small scripts, and tests instead.

## Acceptance criteria

- Both tasks run in isolated worktrees and respect their disjoint ownership.
- No protected WIP, unrelated worktree, credential, private asset, or unbounded raw download is touched.
- P1 does not claim full-upstream readiness without passing the fail-closed preflight.
- P2 does not claim participant, usability, or assistive-technology validation.
- P3 reports reproducible samples and noise boundaries; code changes require a measured benefit.
- The main task retains sole integration, Git mutation, remote synchronization, and production deployment authority.

## Integration closeout amendment — 2026-08-10

The supervising task subsequently delivered two additional exact-base lanes. The closeout therefore integrates four disjoint working-tree packages in this order:

1. UI lane `fa7d`: five-route public UI, responsive/text-spacing/forced-colors repairs, and matching static-shell tests.
2. P2/P3 lane `0313`: bounded proxy evidence, Dashboard preview deferral/readiness instrumentation, performance probe/test, and official immutable Actions pins.
3. P1 lane `aefe`: small acquisition/receipt code, source manifests, tests, and evidence only; raw caches and credentials are excluded.
4. Platform lane `1335`: static architecture/error contract, exact source/dist allowlist, release-manifest schema v2, security/data policy, and final platform gates.

All four packages started from exact base `ca8292040a402eae1d2e461708a4cc912867efcb`, are detached and uncommitted, and have zero same-file overlap. UI and platform are semantically related, so platform remains last and its final allowlist/schema/security checks run against the combined candidate.

The release candidate is not admitted by isolated evidence. Admission requires fresh targeted checks after each package, aggregate Python/public/Dashboard/build/security/claim gates on one fixed commit, one owner for browser/performance verification, a normal push, immutable CI `head_sha`, successful Pages deployment, and independent live artifact checks.
