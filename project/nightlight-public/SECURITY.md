# Security and Platform Maintenance

## Current architecture decision

The public application remains a static build. Its release contract is `static-only`, `local-assets-only`, `aggregate-only`, with no analytics and no runtime external requests.

The four conditions that could justify a server were reviewed against the current product:

| Possible trigger | Current evidence | Decision |
| --- | --- | --- |
| Data must become fresh outside the build cycle | Public content is a reviewed, versioned aggregate snapshot. No live-data promise exists. | Not triggered; rebuild a reviewed artifact. |
| Authorized writing or collaboration is required | The public app has no account, edit, comment, or shared-workflow requirement. | Not triggered; do not add login or a database. |
| Queries are too large or sensitive for static distribution | Only a small allowlisted aggregate artifact is admitted. Restricted and reversible inputs must stay private. | Not triggered; withholding is safer than server-side exposure. |
| License or access control must execute on a server | Published assets are already approved for public distribution; restricted sources are not served. | Not triggered; access control is not a substitute for the public boundary. |

If one of these conditions is later proven, design review must precede implementation. The smallest acceptable service contract would use a versioned schema, return aggregate-only records, define cache and invalidation behavior, log auditable administrative changes without sensitive payloads, minimize retained personal data, distinguish load/validation failures from valid zero, publish explicit degraded behavior, and expose operational health without user analytics. This repository does not implement that service.

## Code responsibilities

- `src/router/`: static route names, paths, and lazy view entry points.
- `src/views/`: route-level composition and presentation. Platform work does not place data validation here.
- `src/components/`: reusable presentation components when a real repeated UI responsibility exists.
- `src/content/`: reviewed public snapshots and artifact-specific lineage validators.
- `src/domain/`: deterministic product rules such as filtering, selection, projection, and comparisons.
- `src/lib/`: UI-independent cross-artifact contracts, currently the aggregate value/error-state contract.
- `scripts/`: build-time release manifest and public-boundary enforcement; never runtime data services.
- `tests/`: unit and integration contracts. Platform tests must use isolated temporary roots and clean them after each test.

The current review found no evidence for a new state manager, TypeScript migration, generic service layer, or large architecture rewrite. Those changes remain no-ops until a reproduced problem requires them.

## Release gates

| Gate | Command | Failure means | Cleanup / owner |
| --- | --- | --- | --- |
| Platform contract | `npm run test:platform` | Data/error, dependency, CSP, public allowlist, or manifest contract is invalid. | One command owner; tests remove their temporary roots. |
| Unit/integration suite | `npm test` | A deterministic source or product contract regressed. | No port or browser; Vitest exits after the run. |
| Source boundary | `npm run verify:public` | Source is not an allowlisted static public candidate. | Read-only apart from normal process output. |
| Build and manifest | `npm run build` | Vite could not produce the static bundle or manifest generation rejected the build contract. | One owner for this worktree's `dist`; Vite replaces it. |
| Release candidate | `npm run validate` | At least one test, build, hash, CSP, dependency, or public source/dist gate failed. | Run serially; remove candidate `dist` if it will not be handed off. |

Browser visual, route/focus, zoom, and text-spacing evidence is owned by the UI lane. Claim/proxy evidence and throttled performance are owned by their existing lanes. One live-process owner must be named before any shared build, preview, browser, port, cache, log, or checkpoint is started.

## Public boundary and browser policy

The verifier uses exact source paths and a narrow generated-asset pattern. It rejects restricted file types and names, private paths, credential-shaped text, source maps, files over 1.5 MB, runtime request entry points, non-reviewed runtime dependencies, weakened CSP/referrer metadata, unsafe new-tab links, and files not represented by the release manifest.

GitHub Pages can carry the CSP and referrer policy embedded in HTML. An HTML meta element is not a general response-header mechanism, so this project does not claim that Pages supplies `X-Content-Type-Options`, `Permissions-Policy`, `frame-ancestors`, or other custom response headers. Host-specific headers require separate live evidence.

## Release evidence levels

1. A local test/build pass proves only a local candidate in one worktree.
2. `dist/release-manifest.json` proves the listed local files, byte lengths, hashes, base path, and static build contract agree. It does not prove a Git commit, merge, CI run, upload, or deployment.
3. CI admission requires an immutable GitHub Actions `head_sha` plus the artifact produced by that run.
4. Publication requires a successful Pages deployment tied to that run and independent verification of the served artifact. Repository notes cannot self-certify this state.

Rollback means redeploying the last admitted immutable artifact or reverting the product commit and running the full admission sequence. Editing a manifest or documentation does not roll back a deployment.

## Dependency and schema maintenance

- Keep direct dependency versions exact and the lockfile committed. Runtime dependencies remain limited to Vue and Vue Router; build/test dependencies remain limited to the reviewed Vite/Vitest toolchain unless a separate review proves a need.
- Run `npm ci`, `npm audit`, `npm run test:platform`, and `npm run validate` after dependency or Node changes. An audit finding is evidence to triage, not permission for an unreviewed major upgrade.
- Node 20 is the lower supported boundary. Action-runtime upgrades belong to the workflow owner; this policy only requires the local and CI Node boundary to remain compatible.
- Aggregate value/error records use an integer schema version. Additive changes still require negative tests and verifier allowlist updates. Breaking changes require a new version, a bounded migration reader only when an admitted old artifact still exists, and removal after the documented compatibility window.
- Deprecations must name the replacement, last supported schema, and removal trigger. Do not keep silent fallbacks.

Review these controls quarterly and whenever dependencies, Node, schema, public artifact paths, CSP, hosting base path, licensing, or deployment workflow changes. Manual screen-reader/assistive-technology work, multi-browser studies, and research with people are conditional product-evidence upgrades, not automated release gates.

## Reporting a vulnerability

Avoid placing secrets or restricted source data in a public issue. Use the repository host's private vulnerability-reporting channel when available, and include the affected exact commit, public path, reproduction steps, and impact without attaching private datasets.
