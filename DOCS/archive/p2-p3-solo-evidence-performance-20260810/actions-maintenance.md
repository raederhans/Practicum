# GitHub Actions Node 24 Maintenance Evidence

## Decision

Update the four existing official GitHub actions to official Node-24-capable releases pinned by immutable full commit SHA. Keep `setup-node` input `node-version: 20` unchanged because that input selects the application's build/test Node version; it does not select the JavaScript runtime embedded in the action itself. No workflow or Pages deployment was triggered.

## Official evidence

All sources were accessed 2026-08-10. No third-party action, blog, or package source was used.

| Official source | Relevant evidence | Applicability boundary |
| --- | --- | --- |
| [GitHub Changelog: Deprecation of Node 20 on GitHub Actions runners](https://github.blog/changelog/2025-09-19-deprecation-of-node-20-on-github-actions-runners/) | Node 20 reached EOL in April 2026; GitHub-hosted runners began using Node 24 by default on 2026-06-16; action users should update to action versions that run on Node 24 | This concerns the runtime used by JavaScript actions. It does not automatically require changing the project runtime passed to `setup-node` |
| [actions/checkout v5.0.0](https://github.com/actions/checkout/releases/tag/v5.0.0) and [v5.0.1](https://github.com/actions/checkout/releases/tag/v5.0.1) | v5 moved the action to Node 24 and requires runner `v2.327.1+`; v5.0.1 is the selected compatible patch | The workflow uses GitHub-hosted `ubuntu-latest`, not an owner-managed old/self-hosted runner |
| [actions/setup-node v5.0.0](https://github.com/actions/setup-node/releases/tag/v5.0.0) | Official release explicitly upgrades the action to Node 24 and requires runner `v2.327.1+` | Its `node-version` input still controls the Node installed for later `npm` commands |
| [actions/upload-pages-artifact v5.0.0](https://github.com/actions/upload-pages-artifact/releases/tag/v5.0.0) | Official release updates its nested `actions/upload-artifact` to v7 | The action is composite; the pinned action file delegates to immutable `actions/upload-artifact@bbbca2d...` v7.0.0 |
| [actions/deploy-pages v5.0.0](https://github.com/actions/deploy-pages/releases/tag/v5.0.0) | Official release explicitly updates Node.js to 24.x | No deployment was run in this task |

Git's read-only remote tag lookup returned the exact release SHAs selected below. Reading each pinned official `action.yml` at the immutable SHA showed `node24` for checkout, setup-node, and deploy-pages; upload-pages-artifact is composite and pins upload-artifact v7.

## Immutable pin update

| Action | Previous pin | Candidate pin |
| --- | --- | --- |
| `actions/checkout` | `11d5960a326750d5838078e36cf38b85af677262` (`v4.4.0`) | `93cb6efe18208431cddfb8368fd83d5badbf9bfd` (`v5.0.1`) |
| `actions/setup-node` | `49933ea5288caeca8642d1e84afbd3f7d6820020` (`v4.4.0`) | `a0853c24544627f65ddf259abe73b1d18a591444` (`v5.0.0`) |
| `actions/upload-pages-artifact` | `56afc609e74202658d3ffba0e8f6dda462b719fa` (`v3.0.1`) | `fc324d3547104276b827a68afc52ff2a11cc49c9` (`v5.0.0`) |
| `actions/deploy-pages` | `d6db90164ac5ed86f2b6aed7e0febac5b3c0c03e` (`v4.0.5`) | `cd2ce8fcbc39b97be8ca5fce6e763baed58fa128` (`v5.0.0`) |

The workflow's triggers, permissions, concurrency, build commands, artifact path, environment, and `node-version: 20` are unchanged. No insecure Node opt-out environment variable was added.

## Validation and limitation

- exact full-SHA pins were resolved through `git ls-remote` against the four official `github.com/actions/*` repositories;
- pinned action metadata was retrieved from `raw.githubusercontent.com/actions/*/<sha>/action.yml` and inspected for `runs.using` / nested `uses`;
- a local static gate checks that all four expected immutable pins occur once and that no branch/tag shorthand replaced them;
- the public application clean install, tests, public-boundary checks, and production build are run locally as the workflow's non-deploy equivalent;
- the GitHub-hosted workflow itself was not dispatched, so GitHub runner admission, artifact upload, environment protection, and deployment remain integration-owner/CI verification risks.
