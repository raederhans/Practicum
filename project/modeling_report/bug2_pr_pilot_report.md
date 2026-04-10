# BUG2 Puerto Rico Pilot Report

## Objective
- Move the BUG line from proxy refinement to official inventory validation for the Puerto Rico pilot jurisdiction.

## Pilot Scope
- pilot_state: PR
- pilot_events: earthquake_sanjuan;maria_sanjuan
- status: awaiting_inventory

## Acquisition Backlog
- PR: availability=, notes=Extreme outage case and the clearest current ground-truth pilot candidate for remote sensing validation.
- FL: availability=, notes=Second priority after PR because multiple current events map to Florida counties.
- LA: availability=, notes=High-value gap because current mainline events already include Louisiana but the permitting path is still unresolved.

## QA Gate
- records_n: 0
- geo_coverage: 0.000
- attribute_coverage: 0.000
- confidential_usage_share: 0.000
- hours_outlier_share: 0.000
- inactive_permit_share: 0.000
- stale_usage_share: 0.000
- gate_pass: 0

## Feature Coverage
- Official inventory features not attached yet.

## Proxy vs Official Overlay
- Overlay not available yet.

## Recommendation
- Expand beyond Puerto Rico only if the QA gate passes and the official BUG features show a clear local increment over the baseline.
- If the pilot remains blocked on data, keep BUG2 as an acquisition-and-validation track rather than a main modeling branch.

## Outputs
- `project/modeling/output/bug2_pilot_acquisition_backlog_v1.csv`
- `project/modeling/output/bug2_pr_canonical_field_mapping_v1.csv`
- `project/modeling/output/bug2_pr_proxy_overlay_v1.csv`
- `project/modeling/output/bug2_pr_pilot_qa_v1.csv`
- `project/modeling/output/bug2_pr_feature_audit_v1.csv`
- `project/modeling/output/bug2_pr_pilot_aggregate_metrics_v1.csv`
- `project/modeling_report/bug2_pr_acquisition_memo_v1.md`
