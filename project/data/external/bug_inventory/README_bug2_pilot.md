# BUG2 Pilot Inventory Layout

This directory is reserved for the official-inventory validation line.

## Priority order

1. `PR`
2. `FL`
3. `LA`

## Puerto Rico pilot

- Put raw source files in `project/data/external/bug_inventory/raw/`
- Put the canonical table in `project/data/external/bug_inventory/canonical/bug_inventory_pr_pilot_v1.csv`
- Keep the canonical schema exactly aligned with `project/modeling/config/bug2_pilot_plan_v1.json`

## Expected canonical columns

- `source_dataset`
- `jurisdiction`
- `state`
- `county_or_district`
- `record_id`
- `facility_name`
- `facility_type_raw`
- `facility_type_std`
- `fuel_type`
- `capacity_kw`
- `operating_hours_annual`
- `address_raw`
- `lat`
- `lon`
- `geo_quality_flag`
- `attribute_quality_flag`
- `source_url`

## Notes

- `earthquake_sanjuan` and `maria_sanjuan` are the only pilot events for this first official-inventory test.
- If no official PR inventory is available yet, keep the template file in place and use the acquisition backlog report to track progress.
- Supporting workflow artifacts now live in:
  - `project/modeling_report/bug2_pr_acquisition_memo_v1.md`
  - `project/modeling/output/bug2_pr_canonical_field_mapping_v1.csv`
  - `project/modeling/output/bug2_pr_proxy_overlay_v1.csv`
