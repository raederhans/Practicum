# Nightlight Recovery Science Foundation Task

## Current status

`ready-for-integration` — all phase decisions, available validation, implementation commit, and post-commit scope verification are complete. Integration remains owned by the future integration owner.

## Checklist

- [x] Verify exact baseline, clean worktree, detached ownership, and prohibited paths.
- [x] Read applicable task-record, Lore-commit, and worktree-delivery rules.
- [x] Implement and test NL-R01 recovery outcome contract.
- [x] Implement and test NL-D01 source/rights/label feasibility manifest.
- [x] Implement and test NL-C01 composite sensitivity protocol and decision.
- [x] Run or evidence-block NL-L01 label pilot under the Phase 1/2 gates.
- [x] Run narrow and related gates; review diff, secrets, large/generated files, and prohibited paths.
- [x] Update handoff evidence and create authorized Lore commit(s).

## Validation evidence

| Command or check | Result |
| --- | --- |
| `git rev-parse HEAD` | Exact required baseline `6b3de4ee97c5391084538bec84db3b1a1f4e05ed`. |
| `git status --short --branch` | Clean detached worktree at startup. |
| `git worktree list --porcelain` | Concurrent worktrees identified; integration and topology changes excluded. |
| `rg --files -g AGENTS.md` | No repository-local `AGENTS.md`; user-provided workspace instructions apply. |
| `py -m pytest project/tests/test_recovery_outcome_contract.py -q` | 6 passed; semantic separation, missingness, sustained thresholds, censoring, and compatibility names verified. |
| `py -m pytest project/tests/test_recovery_source_feasibility.py -q` | 5 passed after correcting a test-only negated-status assertion; inventory, rights/access boundaries, rebuildability, blockers, and handoff verified. |
| `py -m pytest project/tests/test_evidence_passport_composite_sensitivity.py -q` | 5 passed; bounded snapshot identity, normalization/weighting/LOO/Monte Carlo reproducibility, missingness and Public boundary verified. Decision is `no_go`. |
| `py -m pytest project/tests/test_recovery_label_pilot.py -q` | 3 passed; blocked upstream gate produces zero events and labels with matched blockers/handoff and no mock, training, publication, or headline changes. |
| Combined new tests plus `test_event_readiness_contract.py` | 21 passed, 1 skipped; the skip is the existing clean-clone absence of private readiness outputs. |
| Reproducibility/source/modeling pytest group | 37 passed across reproducibility inputs, authorized acquisition, source acquisition, and modeling entrypoint contracts. |
| Stage requirements and artifacts pytest group | 6 passed. |
| `reproducibility.py --scope reviewed-modeling --json` | Exit 0 and `ready`; this proves reviewed-output consistency only. |
| `reproducibility.py --scope full-upstream --json` | Exit 1 and `blocked` on seven existing source/lineage gaps; full upstream reproduction is not established. |
| Python compile plus five JSON parses | Passed for all new support modules, configs, and manifests. |
| Scope, prohibited paths, secret-pattern filenames, changed-file size, generated/data suffix, and tracked diff checks | Passed; no prohibited path, secret pattern, file over 1 MiB, or untracked generated/data artifact found. |
| Implementation Lore commit | `46dfab512e24402d70afce88729137281a7e44fa`, parent `6b3de4ee97c5391084538bec84db3b1a1f4e05ed`; 16 authorized paths and 1,866 insertions; post-commit worktree clean before this closeout-only record update. |

## Open risks and remaining work

- Official source evidence is inventoried; source publicity does not establish derivative lineage, and the exact parent/transform receipts remain blocked.
- Composite sensitivity is evaluated against a bounded tracked snapshot; the research-only `no_go` result is not exported to Public artifacts.
- Label pilot is evidence-backed blocked until source rights, receipts, event-time, denominator, missingness, independence, and rebuildability gates are satisfied.
- `project/tests/test_analysis_contracts.py` was not run: the Windows pytest environment lacks `statsmodels`, while the existing WSL modeling venv has `statsmodels` but lacks `pytest`; no dependency was installed for this task.
- `full-upstream` remains blocked by seven pre-existing external/lineage conditions and was not represented as ready.
