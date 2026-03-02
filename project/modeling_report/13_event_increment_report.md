# Event Increment Report / 新事件增量接入报告

## Objective
- 只同步 teammate 新增事件文件，逐步扩展事件集合并比较 strict-v2、hazard-mainline、quality-matched 三条评估线是否改善。

## Stages
- `baseline_6`: event_count=6, new_event=`baseline`, group=`baseline`
- `stage_7_ian_fortmyers`: event_count=7, new_event=`ian_fortmyers`, group=`us_only`
- `stage_8_ian_charlotteharbor`: event_count=8, new_event=`ian_charlotteharbor`, group=`us_only`
- `stage_9_earthquake_hatay`: event_count=9, new_event=`earthquake_hatay`, group=`intl_addition`
- `stage_10_dorian_freeport`: event_count=10, new_event=`dorian_freeport`, group=`intl_addition`

## Sync & Acquisition
- synced rows: 719
- acquisition records: 6
- generated/new POI files: 0

## Hazard Mainline Logit AUC by Stage
- baseline_6: auc=0.6001, delta_prev=nan, delta_baseline=0.0000
- stage_7_ian_fortmyers: auc=0.4914, delta_prev=-0.1087, delta_baseline=-0.1087
- stage_8_ian_charlotteharbor: auc=0.4856, delta_prev=-0.0058, delta_baseline=-0.1145
- stage_9_earthquake_hatay: auc=0.4726, delta_prev=-0.0130, delta_baseline=-0.1275
- stage_10_dorian_freeport: auc=0.4762, delta_prev=0.0036, delta_baseline=-0.1239

## Survival Best by Stage
- baseline_6: survival_best=0.5341
- stage_7_ian_fortmyers: survival_best=0.5312
- stage_8_ian_charlotteharbor: survival_best=0.5125
- stage_9_earthquake_hatay: survival_best=0.5219
- stage_10_dorian_freeport: survival_best=0.5161

## Figures
- `project/modeling_report/figures/event_increment/logit_auc_by_stage.png`
- `project/modeling_report/figures/event_increment/survival_best_by_stage.png`
- `project/modeling_report/figures/event_increment/strict_v2_in_buffer_by_stage.png`
- `project/modeling_report/figures/event_increment/matched_logit_or_by_stage.png`
- `project/modeling_report/figures/event_increment/event_gap_coverage_map.png`
