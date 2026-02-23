# Modeling Issue Log

All issues here correspond to `project/modeling/output/model_issue_log.csv`.

## Update 2026-02-23 16:26:46 UTC
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions; skip random-intercept export | impact=random-effect chart unavailable for this variant | status=monitor
- [all] maria_sanjuan | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] michael_panamacity | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] earthquake_sanjuan | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] ida_neworleans | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] laura_lakecharles | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] irma_miami | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] all | insufficient_nlcd_coverage | symptom=land_use coverage <= 20% | fix=skip with_nlcd model reruns | impact=only no_nlcd model variant available | status=open

## Update 2026-02-23 16:28:39 UTC
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions; skip random-intercept export | impact=random-effect chart unavailable for this variant | status=monitor
- [all] maria_sanjuan | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] michael_panamacity | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] earthquake_sanjuan | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] ida_neworleans | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] laura_lakecharles | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] irma_miami | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] all | insufficient_nlcd_coverage | symptom=land_use coverage <= 20% | fix=skip with_nlcd model reruns | impact=only no_nlcd model variant available | status=open

## Update 2026-02-23 16:29:32 UTC
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions; skip random-intercept export | impact=random-effect chart unavailable for this variant | status=monitor
- [all] maria_sanjuan | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] michael_panamacity | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] earthquake_sanjuan | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] ida_neworleans | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] laura_lakecharles | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] irma_miami | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] all | insufficient_nlcd_coverage | symptom=land_use coverage <= 20% | fix=skip with_nlcd model reruns | impact=only no_nlcd model variant available | status=open

## Update 2026-02-23 17:51:57 UTC
- [all] maria_sanjuan | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] michael_panamacity | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] earthquake_sanjuan | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] ida_neworleans | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] laura_lakecharles | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] irma_miami | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] all | panel_build_failed_use_cached | symptom=No valid event data available to build pixel panel. | fix=load existing all_events_pixel_panel_v1.parquet | impact=pipeline continues without raw pre/post rebuild | status=resolved
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions; skip random-intercept export | impact=random-effect chart unavailable for this variant | status=monitor
- [Cox] maria_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] michael_panamacity | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] earthquake_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] ida_neworleans | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] laura_lakecharles | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] irma_miami | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] all | recovery_build_failed_use_cached | symptom=No recovery rows built. | fix=load existing recovery_daily_panel_v1.parquet | impact=cox baseline runs on cached recovery panel | status=resolved
- [all] maria_sanjuan | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [all] michael_panamacity | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [all] earthquake_sanjuan | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [all] ida_neworleans | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [all] laura_lakecharles | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [all] irma_miami | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [Cox] maria_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] michael_panamacity | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] earthquake_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] ida_neworleans | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] laura_lakecharles | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] irma_miami | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] all | recovery_build_failed_use_cached_with_land_use | symptom=No recovery rows built. | fix=merge cached recovery panel with panel_nlcd land_use by pixel_id | impact=cox with_nlcd runs without rebuilding post-event stacks | status=resolved
- [Cox] maria_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] michael_panamacity | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] earthquake_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] ida_neworleans | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] laura_lakecharles | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] irma_miami | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] all | recovery_panel_unavailable | symptom=recovery_threshold=0.8: No recovery rows built. | fix=skip this recovery-threshold scenario | impact=partial robustness table for Cox recovery thresholds | status=open
- [Cox] maria_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] michael_panamacity | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] earthquake_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] ida_neworleans | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] laura_lakecharles | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] irma_miami | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] all | recovery_panel_unavailable | symptom=recovery_threshold=0.9: No recovery rows built. | fix=skip this recovery-threshold scenario | impact=partial robustness table for Cox recovery thresholds | status=open
- [Cox] maria_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] michael_panamacity | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] earthquake_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] ida_neworleans | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] laura_lakecharles | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] irma_miami | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] all | recovery_panel_unavailable | symptom=recovery_threshold=0.95: No recovery rows built. | fix=skip this recovery-threshold scenario | impact=partial robustness table for Cox recovery thresholds | status=open

## Update 2026-02-23 17:53:26 UTC
- [all] maria_sanjuan | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] michael_panamacity | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] earthquake_sanjuan | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] ida_neworleans | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] laura_lakecharles | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] irma_miami | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] all | panel_build_failed_use_cached | symptom=No valid event data available to build pixel panel. | fix=load existing all_events_pixel_panel_v1.parquet | impact=pipeline continues without raw pre/post rebuild | status=resolved
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions; skip random-intercept export | impact=random-effect chart unavailable for this variant | status=monitor
- [Cox] maria_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] michael_panamacity | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] earthquake_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] ida_neworleans | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] laura_lakecharles | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] irma_miami | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] all | recovery_build_failed_use_cached | symptom=No recovery rows built. | fix=load existing recovery_daily_panel_v1.parquet | impact=cox baseline runs on cached recovery panel | status=resolved
- [all] maria_sanjuan | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [all] michael_panamacity | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [all] earthquake_sanjuan | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [all] ida_neworleans | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [all] laura_lakecharles | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [all] irma_miami | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions; skip random-intercept export | impact=random-effect chart unavailable for this variant | status=monitor
- [Cox] maria_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] michael_panamacity | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] earthquake_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] ida_neworleans | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] laura_lakecharles | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] irma_miami | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] all | recovery_build_failed_use_cached_with_land_use | symptom=No recovery rows built. | fix=merge cached recovery panel with panel_nlcd land_use by pixel_id | impact=cox with_nlcd runs without rebuilding post-event stacks | status=resolved
- [Cox] maria_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] michael_panamacity | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] earthquake_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] ida_neworleans | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] laura_lakecharles | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] irma_miami | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] all | recovery_panel_unavailable | symptom=recovery_threshold=0.8: No recovery rows built. | fix=skip this recovery-threshold scenario | impact=partial robustness table for Cox recovery thresholds | status=open
- [Cox] maria_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] michael_panamacity | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] earthquake_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] ida_neworleans | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] laura_lakecharles | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] irma_miami | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] all | recovery_panel_unavailable | symptom=recovery_threshold=0.9: No recovery rows built. | fix=skip this recovery-threshold scenario | impact=partial robustness table for Cox recovery thresholds | status=open
- [Cox] maria_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] michael_panamacity | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] earthquake_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] ida_neworleans | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] laura_lakecharles | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] irma_miami | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] all | recovery_panel_unavailable | symptom=recovery_threshold=0.95: No recovery rows built. | fix=skip this recovery-threshold scenario | impact=partial robustness table for Cox recovery thresholds | status=open

## Update 2026-02-23 23:02:08 UTC
- [all] maria_sanjuan | missing_cloud_screening | symptom=No cloud screening CSV found under project/script | fix=set missing_weather_flag=1 and continue | impact=weather controls unavailable for this event | status=monitor
- [all] michael_panamacity | missing_cloud_screening | symptom=No cloud screening CSV found under project/script | fix=set missing_weather_flag=1 and continue | impact=weather controls unavailable for this event | status=monitor
- [all] earthquake_sanjuan | missing_cloud_screening | symptom=No cloud screening CSV found under project/script | fix=set missing_weather_flag=1 and continue | impact=weather controls unavailable for this event | status=monitor
- [all] ida_neworleans | missing_cloud_screening | symptom=No cloud screening CSV found under project/script | fix=set missing_weather_flag=1 and continue | impact=weather controls unavailable for this event | status=monitor
- [all] laura_lakecharles | missing_cloud_screening | symptom=No cloud screening CSV found under project/script | fix=set missing_weather_flag=1 and continue | impact=weather controls unavailable for this event | status=monitor
- [all] irma_miami | missing_cloud_screening | symptom=No cloud screening CSV found under project/script | fix=set missing_weather_flag=1 and continue | impact=weather controls unavailable for this event | status=monitor
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions; skip random-intercept export | impact=random-effect chart unavailable for this variant | status=monitor
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions; skip random-intercept export | impact=random-effect chart unavailable for this variant | status=monitor

## Update 2026-02-23 23:03:16 UTC
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions; skip random-intercept export | impact=random-effect chart unavailable for this variant | status=monitor
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions; skip random-intercept export | impact=random-effect chart unavailable for this variant | status=monitor
