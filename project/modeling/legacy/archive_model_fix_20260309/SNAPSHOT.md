# Pre-Fix Modeling Snapshot

- Archive created: 2026-03-09
- Source commit: `6a0a96874f7cdcdfc1a9b10a64262d11c3cd2c83`
- Purpose: preserve the pre-fix working copies of the active modeling pipelines and documentation entrypoints before the March 2026 reproducibility / leakage cleanup.

## Archived files

- `pipelines/01_in_sample_pipeline.py`
- `pipelines/03_exploration_pipeline.py`
- `modeling/pipeline_lib.py`
- `modeling/run_pipeline.py`
- `pipelines/README.md`
- `modeling_report/index.md`

## Notes

- The archive is a copy-based snapshot. Original files remain in place so imports, wrappers, and existing scripts continue to resolve during the fix round.
- The archived `03_exploration_pipeline.py` includes the uncommitted plotting changes present in the working tree at snapshot time.
- Numerical outputs were not copied into the archive. Regenerated outputs will remain in the normal output directories with their current filenames and timestamps.

## Working tree context at snapshot time

- Modified: `project/modeling/experimental/intl_stage_repair_v1.py`
- Modified: `project/modeling/pipelines/03_exploration_pipeline.py`
- Modified figures under `project/modeling_report/figures/event_increment/`
- Modified figures under `project/modeling_report/figures/intl_stage_repair_v1/`
- Untracked: `BUG dataset tracker.xlsx`
- Untracked: `README_modeling_progress.md`
- Untracked: `README_modeling_progress_notebook.ipynb`
- Untracked: `document/`
- Untracked: `extra data/`
- Untracked: `tmp/`
