# Nightlight Recovery Science Foundation Plan

## Goal

Restore a versioned, testable scientific foundation for recovery outcomes, source and rights feasibility, Evidence Passport composite sensitivity, and a bounded label pilot without changing current Public App or Dashboard claims.

## Scope

- `NL-R01`: freeze the recovery construct and machine-readable outcome contract.
- `NL-D01`: inventory label-source identity, version, grain, rights, access, receipts, missingness, and rebuildability.
- `NL-C01`: evaluate research-only composite sensitivity and define a fail-closed admission decision.
- `NL-L01`: build a one-to-two-event label/censoring pilot only if the earlier source and rights gates pass; otherwise preserve an evidence-backed blocked handoff.
- `NL-P01`: freeze the versioned facility-probability producer contract, legacy-v0 migration boundary, and missing-value preprocessing evidence after the Dashboard ambiguity handoff.

## Sources of truth

- User-delegated four-phase contract in source task `019febfc-6511-7051-89c9-60970113a4ea`.
- Exact implementation baseline `6b3de4ee97c5391084538bec84db3b1a1f4e05ed`.
- Current tracked modeling configs, support logic, source manifests, reproducibility receipts, and directly overlapping tests.
- Official upstream source metadata for source identity, version, access, and publication or license terms.

## Stages

- [x] Phase 1 — NL-R01 recovery construct freeze.
- [x] Phase 2 — NL-D01 source, rights, and label feasibility.
- [x] Phase 3 — NL-C01 Evidence Passport composite sensitivity research.
- [x] Phase 4 — NL-L01 bounded label pilot or evidence-backed blocked decision.
- [x] Verify, document handoff, and create reversible Lore commit(s).
- [x] Phase 5 — NL-P01 facility-probability producer contract and legacy-v0 migration boundary.
- [ ] Re-run the affected gates, record the Dashboard/Public handoff, and create a reversible continuation Lore commit.

## Acceptance criteria

- A versioned machine-readable outcome dictionary defines phenomenon, spatial and temporal units, prediction moment, horizon, T50/T90/burden outcomes, censoring, missing/unavailable, ground truth, proxy status, and prohibited interpretations.
- Deterministic tests keep readiness/admission, observed recovery outcome, forecast, and probability as separate semantic classes and preserve all existing `recovery_days`, R2, AUC, and Passport names.
- A source feasibility manifest covers EAGLE-I, VNP46A2/Black Marble, utility alternatives, and already-authorized sources without adding raw data, credentials, caches, or restricted bytes to Git.
- Composite research covers normalization, weighting, missingness, rank stability, leave-one-component-out, Monte Carlo sensitivity, and same-schema/window-family comparability; its output cannot authorize a Public score, rank, or outcome label.
- Phase 4 runs only if source identity, rights, event time, denominator, independent ground truth, and rebuildability gates all pass. A failed gate produces an exact blocker and executable handoff, not mock labels.
- A versioned producer contract distinguishes an available modeled value of `0.5` from no eligible pixels, all eligible probabilities missing, computation failure, not assessed, and validation failure. It requires `schemaVersion`, value/null, status/reason, source/version, and count/aggregation provenance.
- Legacy-v0 values are never silently rewritten or inferred by a consumer. Migration is regeneration from pinned inputs and model lineage, with a bounded milestone-based dual-read window and explicit retirement gates.
- The existing `fillna(0)` path is classified from repository evidence without changing values: implementation consistency, current-panel missingness, and scientific admissibility are reported separately.
- Narrow tests and related modeling/reproducibility/source gates pass, or every unrun or blocked gate is recorded precisely.

## Non-goals

- No Public App, Dashboard, workflow, registry, raw/cache, credential, main, remote, deployment, or worktree-topology changes.
- No forecast/probability training or publication, headline metric changes, or renaming of existing scientific/public fields.
- No claim that a contract, deterministic test, reviewed output, or CI check establishes scientific or participant validation.

## Risks and constraints

- Existing tracked derivatives may lack parent-release and deterministic transform receipts even when the upstream source is public.
- Earth Engine authentication, Cloud project selection, export receipts, and bounded dates may be unavailable in this worktree.
- Source terms and publication metadata can drift; current official evidence must be recorded with an access date and unresolved ambiguity must fail closed.
- Public and Dashboard worktrees are concurrent semantic dependents; this lane must not absorb their files or weaken their no-score/no-rank/no-outcome boundary.
