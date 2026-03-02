# International Stage Repair Report / Stage 9/10 国际协变量修补报告

## Objective
- 用 WorldPop 栅格采样替换 Stage 9/10 国际事件的旧事件级常数人口值，并用更瘦的 HZ2 transport 规格重跑。

## Legacy Issue Confirmed
- stage_9_earthquake_hatay | earthquake_hatay: legacy_issue_detected=1, old_missing_mean=1.000, v2_missing_mean=0.001, v2_unique=1571, status=ok
- stage_10_dorian_freeport | earthquake_hatay: legacy_issue_detected=1, old_missing_mean=1.000, v2_missing_mean=0.001, v2_unique=1571, status=ok
- stage_10_dorian_freeport | dorian_freeport: legacy_issue_detected=1, old_missing_mean=1.000, v2_missing_mean=0.059, v2_unique=353, status=repair_first

## HZ1 vs HZ2 Comparison
- stage_9_earthquake_hatay Logit auc: HZ1=0.4726, HZ2=0.5048, delta=0.0323, status=improved
- stage_10_dorian_freeport Logit auc: HZ1=0.4762, HZ2=0.4994, delta=0.0232, status=improved
- stage_9_earthquake_hatay survival_best: HZ1=0.5219, HZ2=0.4995, delta=-0.0224, status=worse
- stage_10_dorian_freeport survival_best: HZ1=0.5161, HZ2=0.5000, delta=-0.0161, status=worse
- stage_9_earthquake_hatay Logit brier: HZ1=0.3808, HZ2=0.4820, delta=0.1012, status=worse
- stage_10_dorian_freeport Logit brier: HZ1=0.3748, HZ2=0.5198, delta=0.1450, status=worse

## Strict-V2 Reference (not rerun)
- Stage 9 MixedLM coef(in_buffer): 0.0167
- Stage 10 MixedLM coef(in_buffer): 0.0215

## Matched Reference (Stage 10, not rerun)
- Matched Logit OR(in_buffer): 0.8170

## Readiness Ranking (top)
- ian_charlotteharbor: score=94.0, band=mainline_ready
- ian_fortmyers: score=94.0, band=mainline_ready
- earthquake_sanjuan: score=86.0, band=mainline_ready
- ida_neworleans: score=86.0, band=mainline_ready
- irma_miami: score=86.0, band=mainline_ready

## Repair-First Events
- dorian_freeport: covariate or observation readiness is not high enough for mainline training
- earthquake_hatay: covariate or observation readiness is not high enough for mainline training
- maria_sanjuan: covariate or observation readiness is not high enough for mainline training
- michael_panamacity: covariate or observation readiness is not high enough for mainline training

## Verdict
- Partially improved
- If HZ2 does not recover Stage 9/10, the evidence favors a structural transport issue rather than just a bad international population proxy.

## Figures
- `project/modeling_report/figures/intl_stage_repair_v1/stage9_10_hz1_vs_hz2.png`
- `project/modeling_report/figures/intl_stage_repair_v1/event_readiness_score_v1.png`
