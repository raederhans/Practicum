# Feature Upgrade Report / 复用优先与云量特征升级报告

## Objective
本报告对应本轮“复用优先（teammate/main）+ VNP46A2 云量特征接入”实施结果，回答三个问题：  
1. 数据是否按“缺失补齐、非覆盖”完成复用；  
2. 云量特征是否成功写入六事件像素主表；  
3. 模型结果相对上一轮是否发生可解释变化。

## What Was Implemented
- 新增复用同步脚本：`project/modeling/12_reuse_teammate_assets.py`
- 新增云量特征接入：`attach_cloud_features()`（集成进 `project/modeling/pipeline_lib.py`）
- 新增图表脚本：`project/modeling/14_generate_feature_upgrade_figures.py`
- 重跑主链路：`project/modeling/run_pipeline.py`

## Data Reuse (teammate/main)
- 清单文件：`project/modeling/output/teammate_reuse_manifest.csv`
- 差集文件：`project/modeling/output/teammate_reuse_gap.csv`
- 同步日志：`project/modeling/output/teammate_reuse_sync_log.csv`
- 门禁报告：`project/modeling/output/input_data_gate_report.csv`

关键结果：
- Manifest 总计 `333` 项：`ntl_tif=319`，`poi=6`，`cloud_screening=7`，`download_script=1`
- 同步前缺失 `7` 项，均已从 `teammate/main` 复制（`copied=7`）
- 六事件数据门禁全部通过（`6/6`）
- 因门禁通过，未触发重下载：`project/modeling/output/download_trigger_plan.csv`

## Cloud Features Attached
来源：`project/script/*cloud_screening*.csv`（VNP46A2 QA screening）  
输出：`project/modeling/output/cloud_feature_summary.csv`

写入像素主表字段：
- `pre_valid_ratio`
- `post_valid_ratio`
- `cloud_pre_mean`
- `cloud_post_mean`
- `cloud_window_mean`
- `missing_weather_flag`

覆盖结果：
- 六事件 `missing_weather_flag=0`（无缺失事件）
- 全事件平均 `pre_valid_ratio=0.6475`，`post_valid_ratio=0.5795`

## Model Comparison (Before/After Land-use Controls)
来源：`project/modeling/output/model_summary_for_report.csv`

| Model | Metric | no_nlcd | with_nlcd | Change |
|---|---|---:|---:|---:|
| OLS | `coef_in_buffer` | 0.0283 (p=0.0701) | -0.0238 (p=0.1220) | sign flip |
| MixedLM | `coef_in_buffer` | 0.0283 (p=0.0196) | -0.0238 (p=0.0453) | sign flip |
| Logit | `odds_ratio_in_buffer` | 0.6831 (p=1.02e-07) | 1.1777 (p=0.1049) | <1 -> >1 |
| Logit | `AUC` | 0.7192 | 0.7490 | +0.0298 |
| Cox | `hazard_ratio_in_buffer` | 1.1261 (p=4.53e-07) | 1.0536 (p=0.0692) | effect attenuated |

简要解读：
- 引入 land-use 后，`in_buffer` 在 OLS/MixedLM/Logit 上出现方向翻转；
- Cox 仍保持 `HR>1`，但显著性弱化；
- Logit AUC 提升，说明分类性能略改善，但主解释变量方向冲突仍在。

## Figures
- 模型对比图：`project/modeling_report/figures/feature_upgrade/model_compare_before_after.png`
- 云量可用率图：`project/modeling_report/figures/feature_upgrade/cloud_usable_ratio_by_event.png`
- 复用同步汇总图：`project/modeling_report/figures/feature_upgrade/teammate_sync_summary.png`

## Problems & Fixes
1. 症状：云量特征初次接入全部缺失。  
原因：manifest 里 cloud 文件匹配规则过严（正则漏匹配）。  
修复：放宽 regex 并重跑复用同步 + 主流水线。  
影响：云量特征成功覆盖六事件。

2. 症状：执行环境缺少 parquet 与建模依赖。  
原因：系统 Python 缺少 `pyarrow/rasterio/statsmodels/lifelines/matplotlib/seaborn`。  
修复：创建本地虚拟环境 `.venv_modeling` 并安装依赖后运行。  
影响：主流程可复现执行，仓库代码逻辑未改动。

## Verdict
本轮结论：`Partially Improved`
- 工程层面：显著改进（复用链路、门禁、云量特征全部打通）
- 统计一致性：未收敛（跨模型 `in_buffer` 方向冲突仍存在）
- 可追溯性：改进（manifest/gap/sync/gate/cloud 全链路可审计）

## Next Step
1. 保持同样本锁定前提下，引入 OSM 设施特征并做增量对照（仅新增控制，不改样本）。  
2. 对 Logit 做正则化/降维，缓解 land-use 引入后的不稳定。  
3. 对 MixedLM + Cox 做统一规格收敛（主文双主结果口径）。
