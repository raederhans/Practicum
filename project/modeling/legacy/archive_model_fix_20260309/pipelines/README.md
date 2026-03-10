# Modeling Pipelines

Active entrypoints are now the real implementations, not script-dispatch wrappers:

1. `01_in_sample_pipeline.py`
   - Baseline / NLCD / feature-upgrade / strict-v2 / reports / figures
2. `02_cross_event_pipeline.py`
   - Cross-event V3 and V3 stabilization
3. `03_exploration_pipeline.py`
   - Exploration V2 and future sensitivity experiments

Compatibility shims remain at the modeling root for `run_pipeline.py` and `15-19`.
Historical stage-specific scripts are under `project/modeling/legacy/`, and the pre-merge `15-19` implementations are under `project/modeling/legacy/archive_premerge/`.
