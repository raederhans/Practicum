# Hazard Readiness-Filtered Transport Report

## Objective
- Re-run the HZ1 hazard/exposure transport line on the current mainline-ready event subset instead of the full mixed event pool.

## Mainline Event Allowlist
- ian_charlotteharbor: source=config, readiness_band=mainline_ready
- earthquake_sanjuan: source=config, readiness_band=mainline_ready
- ida_neworleans: source=config, readiness_band=mainline_ready
- irma_miami: source=config, readiness_band=mainline_ready
- laura_lakecharles: source=config, readiness_band=mainline_ready

## Metric Comparison
- QT1 Logit AUC: 0.4973
- full-event HZ1 Logit AUC: 0.6025
- readiness-filtered HZ1_READY Logit AUC: 0.5630
- HZ1_READY vs HZ1 AUC delta: -0.0396
- full-event HZ1 Logit Brier: 0.4424
- HZ1_READY Logit Brier: 0.3554
- HZ1_READY vs HZ1 Brier delta: -0.0870

## Top Hazard Features (Logit)
- island_local_urban: mean_coef=2.8161, abs=2.8161, sign_consistency=1.00
- island_local_water: mean_coef=-0.8546, abs=0.8546, sign_consistency=1.00
- event_slope_milli: mean_coef=-0.0526, abs=0.6459, sign_consistency=0.50
- hazard_cloud_water: mean_coef=-0.5616, abs=0.5929, sign_consistency=0.75
- event_cloud_shift: mean_coef=0.5548, abs=0.5569, sign_consistency=0.75

## Recommendation
- HZ1_READY improves calibration but loses ranking power versus the full-event HZ1 line, so it should remain a robustness subset rather than the main predictive anchor.
- Treat the readiness subset as a cleaner benchmark, not as a replacement for explanatory models.

## Outputs
- `project/modeling/output/hazard_transport_readiness_aggregate_metrics_v1.csv`
- `project/modeling/output/hazard_transport_readiness_feature_summary_v1.csv`
- `project/modeling/output/hazard_transport_readiness_events_v1.csv`
- `project/modeling_report/figures/exploration_v2/hazard_transport_readiness_compare_v1.png`
