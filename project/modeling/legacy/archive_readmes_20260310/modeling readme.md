# 基于NASA Black Marble夜光遥感数据的关键基础设施备用发电机使用检测

## 项目概述

本项目利用NASA VIIRS Black Marble（VNP46A2）逐日夜光（NTL）数据，检测自然灾害期间关键基础设施的备用发电机使用情况。核心思路：配备备用发电机的医院、机场、电厂等关键设施，在停电期间应比周边区域保持更高的夜光亮度。通过分析这些设施缓冲区与非缓冲区的NTL保留差异，可以识别出作为备用发电机信号的"韧性优势"（resilience advantage）。

研究覆盖了6个灾害事件，涵盖不同地区、灾害类型、城市规模和恢复时长，形成了稳健的跨事件比较框架。

**指导教授**：李晓江

---

## 六个灾害事件

| 事件ID | 事件名称 | 地点 | 灾害日期 | 边界框 (W, S, E, N) | 投影 | 恢复时长 |
|---|---|---|---|---|---|---|
| `maria_sanjuan` | 飓风Maria | 圣胡安, 波多黎各 | 2017-09-20 | [-66.20, 18.35, -65.95, 18.48] | EPSG:32619 | 数月 |
| `michael_panamacity` | 飓风Michael | 巴拿马城, 佛州 | 2018-10-10 | [-85.80, 30.10, -85.55, 30.25] | EPSG:32616 | 数周 |
| `earthquake_sanjuan` | 波多黎各地震 | 圣胡安, 波多黎各 | 2020-01-07 | [-66.20, 18.35, -65.95, 18.48] | EPSG:32619 | 数天 |
| `ida_neworleans` | 飓风Ida | 新奥尔良, 路州 | 2021-08-29 | [-90.20, 29.87, -89.90, 30.08] | EPSG:32615 | 数周 |
| `laura_lakecharles` | 飓风Laura | 查尔斯湖, 路州 | 2020-08-27 | [-93.35, 30.15, -93.10, 30.30] | EPSG:32615 | 数周–月 |
| `irma_miami` | 飓风Irma | 迈阿密, 佛州 | 2017-09-10 | [-80.40, 25.70, -80.10, 25.90] | EPSG:32617 | 1–2周 |

---

## 仓库文件结构

```
project/
├── README.md                          ← 本文件
│
├── data/
│   ├── processed/                     ← VNP46A2逐日GeoTIFF（从GEE导出）
│   │   ├── Maria-VNP46A2-pre/         ← 命名格式: {event_id}_{period}_{YYYY-MM-DD}.tif
│   │   ├── Maria-VNP46A2-post/
│   │   ├── Michael-VNP46A2-pre/
│   │   ├── Michael-VNP46A2-post/
│   │   ├── Earthquake-VNP46A2-pre/
│   │   ├── Earthquake-VNP46A2-post/
│   │   ├── Ida-VNP46A2-pre/
│   │   ├── Ida-VNP46A2-post/
│   │   ├── Laura-VNP46A2-pre/
│   │   ├── Laura-VNP46A2-post/
│   │   ├── Irma-VNP46A2-pre/
│   │   └── Irma-VNP46A2-post/
│   ├── nlcd/                          ← ★ NLCD土地覆盖数据（从GEE导出，500m）
│   │   ├── nlcd_maria_sanjuan.tif     ← PR用2016版
│   │   ├── nlcd_earthquake_sanjuan.tif
│   │   ├── nlcd_michael_panamacity.tif ← 美国本土用2021版
│   │   ├── nlcd_ida_neworleans.tif
│   │   ├── nlcd_laura_lakecharles.tif
│   │   └── nlcd_irma_miami.tif
│   └── raw/                           ← （可选）VNP46A3月度.h5瓦片
│
├── script/
│   ├── multi_event_ntl_download.ipynb ← GEE下载脚本（6事件统一）
│   ├── multi_event_eda.ipynb          ← 统一EDA分析流水线
│   └── MariaEDAV2.ipynb               ← 原始单事件EDA（参考用）
│
├── result/
│   ├── plots/
│   │   ├── maria_sanjuan/             ← 各事件可视化输出
│   │   │   ├── ntl_timeseries.png
│   │   │   ├── buffer_vs_nonbuffer_timeseries.png
│   │   │   ├── resilience_curve_by_type.png
│   │   │   ├── ntl_change_ratio_map.png
│   │   │   └── brighter_pixel_investigation.png
│   │   ├── michael_panamacity/
│   │   ├── earthquake_sanjuan/
│   │   ├── ida_neworleans/
│   │   ├── laura_lakecharles/
│   │   ├── irma_miami/
│   │   └── cross_event/               ← 跨事件对比图
│   │       ├── cross_event_resilience_heatmap.png
│   │       ├── cross_event_bar_chart.png
│   │       └── cross_event_dumbbell.png
│   │
│   ├── all_events_resilience_summary.csv      ← 主表：事件×设施类型韧性汇总
│   ├── cross_event_resilience_by_type_stats.csv
│   ├── {event_id}_resilience_by_facility_type.csv
│   ├── {event_id}_ntl_by_type.csv             ← 各设施类型逐日时间序列
│   ├── {event_id}_critical_infra_poi.csv      ← OSM关键设施POI
│   └── {event_id}_ntl_change_ratio.tif        ← 像素级变化率GeoTIFF
│
└── modeling/                          ← ★ 待建：回归建模部分
    ├── 01_build_pixel_dataset.py
    ├── 02_ols_baseline.py
    ├── 03_mixed_effects.py
    ├── 04_logistic_regression.py
    ├── 05_survival_analysis.py
    └── pixel_data/
        └── all_events_pixel_panel.parquet
```

### GeoTIFF命名规范

```
{event_id}_{period}_{YYYY-MM-DD}.tif
```
- `event_id`：如 `maria_sanjuan`、`ida_neworleans`
- `period`：`pre`（灾前）或 `post`（灾后）
- 波段：`Gap_Filled_DNB_BRDF_Corrected_NTL`（已乘以缩放因子×0.1，单位：nW/cm²/sr）
- 空间分辨率：500m（VNP46A2原生分辨率）
- 坐标系：EPSG:4326（WGS84）

### 关键数据文件说明

| 文件 | 说明 |
|---|---|
| `all_events_resilience_summary.csv` | 跨事件主对比表。每行 = 一个事件×一个设施类型。包含 `pre_buffer_ntl`, `pre_nonbuffer_ntl`, `resilience_buffer`, `resilience_nonbuffer`, `resilience_advantage` |
| `{event_id}_ntl_by_type.csv` | 各事件逐日时间序列。列：`date`, `period`, `facility_type`, `buffer_ntl`, `nonbuffer_ntl`。是BEAST分解的输入 |
| `{event_id}_critical_infra_poi.csv` | 从OpenStreetMap Overpass API查询的关键设施POI。设施类型包括：`hospital`, `fire_station`, `police`, `government`, `water_works`, `power_plant`, `substation`, `aerodrome` |
| `{event_id}_ntl_change_ratio.tif` | 像素级 `(post - pre) / pre` 变化率，遵循Zhang et al. (2023)方法 |

---

## 目前研究进展

### 第一阶段：数据获取 ✅ 已完成

- 通过Google Earth Engine下载VNP46A2逐日NTL数据，**下载前预筛选云量**（阈值：30%云覆盖率）
- 每个事件约20天灾前数据 + 不等数量灾后数据
- 导出时已做QA掩膜：QF_Cloud_Mask位提取、Snow_Flag过滤、缩放因子（×0.1）
- GEE项目ID：`deductive-tempo-485113-n8`

### 第二阶段：探索性数据分析（EDA）✅ 已完成

统一流水线（`multi_event_eda.ipynb`）对6个事件执行相同分析：

1. **影像扫描** → 每个事件的整体NTL时间序列
2. **POI下载** → 通过Overpass API获取关键基础设施位置
3. **差异化缓冲区构建**：机场1250m，其他设施750m（在投影坐标系下计算）
4. **缓冲区 vs 非缓冲区逐日统计** → 按设施类型分组的每日buffer/non-buffer平均NTL
5. **韧性曲线** → NTL归一化到灾前基线（Business-As-Usual）
6. **NTL变化率制图** → 像素级 `(post均值 - pre均值) / pre均值`，遵循Zhang et al. (2023)
7. **跨事件对比** → 热力图 + 分组柱状图，展示事件×设施类型的韧性优势

### 核心EDA发现

**跨事件韧性模式——"地板效应"（Floor Effect）**：

| 事件 | Buffer/Non-buffer亮度比 | 韧性优势 |
|---|---|---|
| Maria（圣胡安） | 1.64（buffer更亮） | **+5.1%** |
| Ida（新奥尔良） | 1.16 | **+3.0%** |
| Irma（迈阿密） | 1.31 | **+2.5%** |
| 地震（圣胡安） | 1.70 | −1.3%（弱事件） |
| Laura（查尔斯湖） | 1.15 | −2.3% |
| Michael（巴拿马城） | 0.42（buffer反而更暗！） | −16.2% |

**解读**：在城区密集的大城市（圣胡安、新奥尔良、迈阿密），关键设施所在区域灾前就较亮，原始buffer vs non-buffer对比能看到清晰的正向韧性信号。但在小城市（巴拿马城、查尔斯湖），关键设施位于较暗的城区核心，而非缓冲区反而包含了更亮的郊区——形成了**基线亮度混淆**，掩盖了真实的备用发电机信号。**这正是为什么需要像素级回归控制基线亮度的原因。**

**设施类型信号**：
- **电厂（power_plant）**：信号最强（跨事件均值 +4.6%），与物理预期一致
- **机场（aerodrome）**：第二强（+2.7%），符合FAA备用电力要求
- **医院、政府、警察、消防**：整体均值略为负，但完全是被Michael和Laura拉低的；在三个大城市中这些类型也显示小幅正向信号

---

## 第三阶段：像素级回归建模（待完成）

> **这是论文的核心贡献。** EDA表明原始buffer/non-buffer对比受基线亮度混淆。回归方法控制该混淆因素，将6个事件合并分析，提供备用发电机检测的统计证据。

### 3.1 三层建模策略总览

用三个互补的模型从不同角度问同一个问题。如果三层结果方向一致，论文说服力极强。

| 层级 | 模型类型 | 因变量 | 回答的问题 |
|---|---|---|---|
| 第一层 | OLS → 混合效应 | ΔNTL（连续值） | 缓冲区像素的NTL下降是否更少？ |
| 第二层 | Logistic回归 | is_damaged（二分类） | 缓冲区像素"受损"的概率是否更低？ |
| 第三层 | 生存分析 / Cox PH | recovery_duration（时间） | 缓冲区像素恢复是否更快？ |

---

### 3.2 第一步：构建像素级面板数据集

**目标**：将各事件的GeoTIFF + 缓冲区几何体转换为一个扁平dataframe，每行 = 一个像素。

**每个事件的输入文件**：
- `data/processed/{Event}-VNP46A2-pre/*.tif`（灾前逐日GeoTIFF）
- `data/processed/{Event}-VNP46A2-post/*.tif`（灾后逐日GeoTIFF）
- `result/{event_id}_critical_infra_poi.csv`（设施位置）

**处理逻辑**：

```
对每个事件:
  0. ★ 加载POI CSV → 筛选设施类型（去掉无关设施，详见下方）
  1. 加载所有灾前GeoTIFF → 堆叠 → 计算每像素 pre_mean_ntl
  2. 加载所有灾后GeoTIFF → 堆叠 → 计算每像素 post_mean_ntl
  3. 计算 delta_ntl = (post_mean_ntl - pre_mean_ntl) / pre_mean_ntl（像素级变化率）
  4. 筛选后的POI → 在投影坐标系下创建缓冲区（机场1250m，其他750m）
  5. 将缓冲区栅格化到与GeoTIFF相同的网格上
  6. 对每个像素提取：
     - in_buffer (0/1)：是否落在任何设施缓冲区内
     - nearest_facility_type：最近设施类型
     - distance_to_nearest：到最近设施的距离（米）
  7. 过滤：仅保留 pre_mean_ntl > 0.5 nW/cm²/sr 的像素（排除水体/无人区）
  8. 添加 event_id 列
```

**★ 设施筛选规则（Step 0）**：

Overpass API查回来的POI包含各种大小设施，但并非所有设施都有备用发电机。筛选原则：**这个设施在停电的夜晚有没有可能还亮着灯？** 白天运营晚上关门的设施（如学校、小诊所），灾后夜晚不会亮，NTL看不到信号，放进去只增加噪声。

| 保留 ✅ | 原因 |
|---|---|
| `hospital` | 法律强制要求备用发电机（CMS认证） |
| `aerodrome` | FAA强制要求备用电力 |
| `power_plant` | 本身就是发电设施 |
| `substation` | 电网关键节点，通常有备用电力 |
| `fire_station` | 24小时运营，应急核心设施 |
| `police` | 24小时运营，应急核心设施 |
| `government`（大型，如city hall） | 可能有备用电力，保留观察 |
| `water_works` | 供水关键设施，通常有备用电力 |

| 去掉 ❌ | 原因 |
|---|---|
| `clinic`（社区诊所/私人门诊） | 面积小，晚上关门，无强制发电机要求 |
| `school` | 晚上无人，停电后夜光信号本来就是零 |
| 其他小型office类建筑 | 非24小时运营，无备用电力需求 |

```python
# 设施筛选代码
KEEP_TYPES = ['hospital', 'aerodrome', 'power_plant', 'substation',
              'fire_station', 'police', 'government', 'water_works']
poi = poi[poi['facility_type'].isin(KEEP_TYPES)].copy()
```

**子模型分组**（建模阶段使用）：

| 分组 | 设施类型 | 预期信号强度 |
|---|---|---|
| 第一组：强制有发电机 | `hospital`, `aerodrome`, `power_plant` | 最强 |
| 第二组：可能有发电机 | `fire_station`, `police`, `government` | 中等 |
| 第三组：不确定 | `substation`, `water_works` | 较弱 |

如果β₁呈现"第一组 > 第二组 > 第三组"的梯度，**信号强度与设施拥有发电机的先验概率吻合，本身就是最好的验证。**

**★ 土地利用（Land Use）数据获取**：

每个像素需要附加NLCD土地覆盖类型，作为回归中的控制变量。不同用地类型（商业区、住宅区、工业区）的夜光行为本质上不同——商业区晚上关灯可能本来就暗，住宅区一直有人在。加入land use可以控制这种差异，使 `in_buffer` 的系数更干净。

数据源：**USGS National Land Cover Database (NLCD)**，30m分辨率，免费。
- 美国本土：GEE上 `USGS/NLCD_RELEASES/2021_REL/NLCD`（2021版）
- 波多黎各：GEE上 `USGS/NLCD_RELEASES/2016_REL/NLCD`（2016版，PR最新只有这个，但土地利用类型几年内基本不变，完全够用）

GEE获取代码：
```javascript
// 美国本土事件（Michael, Ida, Laura, Irma）
var nlcd = ee.Image('USGS/NLCD_RELEASES/2021_REL/NLCD/2021').select('landcover');

// 波多黎各事件（Maria, Earthquake）
var nlcd_pr = ee.Image('USGS/NLCD_RELEASES/2016_REL/NLCD/2016').select('landcover');

// 导出到与NTL相同的500m网格（用 mode 重采样，取30m像素中的众数类别）
Export.image.toDrive({
  image: nlcd.reduceResolution({reducer: ee.Reducer.mode(), maxPixels: 1024}).reproject({crs: 'EPSG:4326', scale: 500}),
  description: 'nlcd_landcover_500m',
  region: roi,
  scale: 500,
  crs: 'EPSG:4326'
});
```

关注的NLCD类别：

| NLCD代码 | 类别 | 说明 |
|---|---|---|
| 21 | Developed, Open Space | 开放开发用地（公园、高尔夫球场） |
| 22 | Developed, Low Intensity | 低密度开发（大地块住宅） |
| 23 | Developed, Medium Intensity | 中密度开发（小地块住宅/公寓） |
| 24 | Developed, High Intensity | 高密度开发（商业/工业/密集城区） |

建模时可以直接用这4类作为分类变量，或简化为二元变量 `is_high_density`（23+24 vs 21+22），降低自由度。非开发用地（森林、水体、农田等）的像素会被 `pre_mean_ntl > 0.5` 过滤掉，通常不会进入模型。

**输出表结构**（`all_events_pixel_panel.parquet`）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `pixel_id` | str | `{event_id}_{row}_{col}` 唯一标识 |
| `event_id` | str | 6个事件ID之一 |
| `row`, `col` | int | 像素在栅格中的行列位置 |
| `lon`, `lat` | float | 像素中心点经纬度 |
| `pre_mean_ntl` | float | 灾前所有天的NTL均值（nW/cm²/sr） |
| `post_mean_ntl` | float | 灾后所有天的NTL均值 |
| `delta_ntl` | float | `(post - pre) / pre`，归一化变化率 |
| `in_buffer` | int | 1=在任意设施缓冲区内，0=不在 |
| `nearest_facility_type` | str | 最近关键设施的类型 |
| `distance_to_nearest` | float | 到最近设施的距离（米） |
| `n_facilities_in_buffer` | int | 覆盖该像素的设施缓冲区数量 |
| `land_use` | int | NLCD土地覆盖类别代码（21/22/23/24） |
| `is_damaged` | int | 1 if `delta_ntl < -0.10`（超过10%下降），否则0 |

**实现要点**：
- 用 `rasterio` 读取GeoTIFF并提取仿射变换矩阵
- 用 `geopandas` 做缓冲区几何操作（**必须先转到投影坐标系再做buffer！**）
- 用 `rasterio.features.rasterize()` 将缓冲区矢量"烧录"到栅格网格
- 用 `scipy.spatial.cKDTree` 做快速最近邻查找
- NLCD重采样：30m → 500m用众数（mode）聚合，确保每个NTL像素获得主导土地类型
- 保存为 `.parquet` 格式（比CSV快很多）
- 预计数据量：6个事件共约 50,000–200,000 个像素

**所需库**：
```python
import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from rasterio.features import rasterize
from scipy.spatial import cKDTree
```

**像素过滤阈值**：`pre_mean_ntl > 0.5` 排除水体、森林和无人居住区。这是NTL文献中常用的城区/非城区分界值。建议同时尝试 0.3 和 1.0 作为敏感性检验。

---

### 3.3 第二步（可选）：BEAST时间序列分解

**目的**：将NTL时间序列分离为趋势、季节和噪声分量。BEAST提供比原始NTL更干净的回归因变量（趋势分量而非原始噪声值）。

**是否必须做**：这是稳健性增强，不是前置条件。如果时间紧，可以跳过BEAST，直接用原始的灾前灾后NTL差值作为因变量。BEAST结果可以后续作为robustness check补充。

**如果做BEAST**：

现有EDA notebook已经包含了聚合层面（buffer/non-buffer区域均值）的BEAST代码。对像素级BEAST，计算量很大（每个像素×6个事件 = 一次BEAST拟合）。实际操作建议：

1. 对每个事件的**聚合** buffer/non-buffer 时间序列跑BEAST（EDA中已做）
2. 利用BEAST的**changepoint检测**精确定义停电起止时间（而非用固定日期）
3. 用这些精确窗口重新计算像素级 `pre_mean_ntl` 和 `post_mean_ntl`

**BEAST参数**（EDA中已调优）：
```python
import Rbeast as rb

opt = rb.args()
opt.period             = 1.0       # 年周期（以年为单位）
opt.minSeasonOrder     = 1
opt.maxSeasonOrder     = 3
opt.minTrendOrder      = 0
opt.maxTrendOrder      = 1
opt.maxKnotNum_Trend   = 5         # 最多5个趋势突变点
opt.maxKnotNum_Season  = 3
opt.mcmc_samples       = 8000
opt.mcmc_burnin        = 2000
opt.mcmc_thin          = 3
opt.hasOutlier         = True
```

---

### 3.4 第三步：第一层 — OLS和混合效应回归

#### 3.4.1 OLS基线模型

**模型公式**：
```
delta_ntl_i = β₀ + β₁·in_buffer_i + β₂·pre_ntl_i + β₃·(in_buffer_i × pre_ntl_i) + β₄·land_use_i + Σ αⱼ·event_j + ε_i
```

| 变量 | 说明 | 预期符号 |
|---|---|---|
| `β₁`（in_buffer） | **核心系数**：控制基线亮度和用地类型后，缓冲区的韧性优势 | **正**（缓冲区NTL下降更少） |
| `β₂`（pre_ntl） | 基线亮度对NTL变化的影响 | 正（越亮的像素可下降空间越大） |
| `β₃`（交互项） | buffer效应如何随亮度变化——**这个就是解决floor effect的关键** | 捕捉巴拿马城 vs 圣胡安的差异 |
| `β₄`（land_use） | 不同用地类型的NTL变化差异（每类一个系数） | 高密度开发区下降较少 |
| `αⱼ`（事件固定效应） | 控制不同灾害/地区的系统性差异 | 各不相同 |

**代码实现**：
```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 加载数据
df = pd.read_parquet('pixel_data/all_events_pixel_panel.parquet')

# OLS + 事件固定效应 + 土地利用控制
model_ols = smf.ols(
    'delta_ntl ~ in_buffer * pre_mean_ntl + C(land_use) + C(event_id)',
    data=df
).fit(cov_type='HC1')  # HC1 = 异方差稳健标准误

print(model_ols.summary())
```

**关键参数说明**：
- 用 `cov_type='HC1'`（稳健标准误），因为像素级方差几乎必然是异方差的
- `C(event_id)` 自动为每个事件创建虚拟变量（参考组：字母序第一个事件）
- 如果要加设施类型：`delta_ntl ~ in_buffer * pre_mean_ntl + C(land_use) + C(event_id) + C(nearest_facility_type)`

**结果解读**：
- `β₁` 应该**显著为正**（p < 0.05）→ 缓冲区下降更少
- `β₃`（交互项）的符号反映floor effect方向
- R²会很低（大概 0.01–0.10）—— 这对像素级回归是正常的；关键看系数显著性，不看R²

#### 3.4.2 混合效应（层级）模型 — 主要结果

**为什么用混合效应**：像素嵌套在设施中，设施嵌套在事件中。这种层级结构违反OLS的独立性假设。混合效应处理组内相关性，并允许备用发电机效应在不同事件/设施间有变异。

**模型公式**：
```
delta_ntl_ij = β₀ + β₁·in_buffer_ij + β₂·pre_ntl_ij + β₃·(in_buffer × pre_ntl)_ij + β₄·land_use_ij + u_j + ε_ij

其中：
  i = 像素索引
  j = 事件索引
  u_j ~ N(0, σ²_u)   ← 事件级随机截距
  ε_ij ~ N(0, σ²)    ← 像素级残差
```

**代码实现**：
```python
import statsmodels.formula.api as smf

# 事件随机截距
model_mixed = smf.mixedlm(
    'delta_ntl ~ in_buffer * pre_mean_ntl + C(land_use)',
    data=df,
    groups=df['event_id'],    # 按事件分组的随机截距
    # re_formula='~in_buffer'  # 取消注释可加随机斜率（需收敛）
).fit(method='lbfgs')

print(model_mixed.summary())
```

**参数调优建议**：
- 先只加随机截距（`groups=event_id`）；等价于事件固定效应但有正确的方差分解
- 如果收敛，尝试取消注释 `re_formula='~in_buffer'` 加随机斜率——允许buffer效应因事件而异，更符合现实
- 如果随机斜率不收敛，保持只有随机截距
- 优化方法：先试 `'lbfgs'`；不行试 `'powell'` 或 `'nm'`
- 替代库：`pymer4`（R的lme4的Python包装）——优化器更好但需要安装R

**带设施类型的扩展模型**：
```python
model_extended = smf.mixedlm(
    'delta_ntl ~ in_buffer * pre_mean_ntl + C(land_use) + C(nearest_facility_type)',
    data=df,
    groups=df['event_id']
).fit()
```

**论文报告**：将混合效应模型作为主要结果，OLS放附录作为稳健性检验。需报告的关键数字：β₁（buffer效应）、β₃（交互项）、随机截距方差（σ²_u）、ICC（组内相关系数 = σ²_u / (σ²_u + σ²)）。

---

### 3.5 第四步：第二层 — Logistic回归

**目的**：问一个二分类问题——"缓冲区像素受损的概率是否更低？"

**因变量**：`is_damaged = 1 if delta_ntl < -0.10, else 0`

-10%的阈值来自Zhang et al. (2023)的灾害损害评估方法。将连续的ΔNTL转化为二分类结果。

**模型公式**：
```
logit(P(is_damaged_i = 1)) = β₀ + β₁·in_buffer_i + β₂·pre_ntl_i + β₃·(in_buffer × pre_ntl)_i + β₄·land_use_i + Σ αⱼ·event_j
```

**代码实现**：
```python
import statsmodels.formula.api as smf
import numpy as np

# 创建二分类因变量
df['is_damaged'] = (df['delta_ntl'] < -0.10).astype(int)

# 检查类别平衡
print(df['is_damaged'].value_counts(normalize=True))

# Logistic回归
model_logit = smf.logit(
    'is_damaged ~ in_buffer * pre_mean_ntl + C(land_use) + C(event_id)',
    data=df
).fit()

print(model_logit.summary())

# 优势比（更直观）
odds_ratios = np.exp(model_logit.params)
print("\n优势比 (Odds Ratios):")
print(odds_ratios)

# 边际效应（概率变化量）
marginal = model_logit.get_margeff()
print(marginal.summary())
```

**结果解读**：
- `β₁` 应该**为负**（缓冲区像素受损概率更低）
- 优势比 < 1.0 → 缓冲区受损的几率（odds）更低
- 报告：优势比 + 95%置信区间、边际效应（如"位于缓冲区使受损概率降低X个百分点"）

**敏感性分析**：除了-10%，还要试 -5%、-15%、-20%。如果buffer效应在不同阈值下都稳健，说明结论可靠。

**混合效应Logistic**（如果计算资源允许）：
```python
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial

model_gee = GEE.from_formula(
    'is_damaged ~ in_buffer * pre_mean_ntl + C(land_use)',
    groups='event_id',
    data=df,
    family=Binomial()
).fit()
```

---

### 3.6 第五步：第三层 — 生存分析

**目的**：问一个时间维度的问题——"缓冲区像素恢复是否更快？"

**因变量**：`recovery_duration` = 灾后NTL恢复到灾前水平≥90%所需的天数。

**构建恢复变量**（需要灾后逐日数据，不是仅灾前灾后均值）：

```python
def compute_recovery_duration(pixel_pre_mean, pixel_post_daily, threshold=0.90):
    """
    对一个像素，找到灾后NTL首次达到 threshold * pre_mean 的天数。
    
    参数：
        pixel_pre_mean: float, 灾前平均NTL
        pixel_post_daily: list of (day_number, ntl_value)元组, 按天排序
        threshold: 恢复阈值（默认0.90 = 灾前水平的90%）
    
    返回：
        recovery_days: int 或 NaN（如果观测窗口内未恢复）
        is_censored: bool（True = 未观测到恢复，即右删失）
    """
    target = threshold * pixel_pre_mean
    for day_num, ntl in pixel_post_daily:
        if ntl >= target:
            return day_num, False  # 已恢复
    return len(pixel_post_daily), True  # 删失（未恢复）
```

**注意**：这需要逐张加载灾后GeoTIFF（而非取均值）。对每个像素追踪逐日NTL，找到首次超过恢复阈值的那一天。

**Cox比例风险模型**：
```python
from lifelines import CoxPHFitter

# 准备生存数据
surv_df = df[['recovery_days', 'is_censored', 'in_buffer',
              'pre_mean_ntl', 'land_use', 'event_id']].copy()
surv_df['event_observed'] = ~surv_df['is_censored']  # lifelines惯例：1=事件发生

# 对event_id和land_use做哑变量编码
event_dummies = pd.get_dummies(surv_df['event_id'], prefix='event', drop_first=True)
lu_dummies = pd.get_dummies(surv_df['land_use'], prefix='lu', drop_first=True)
surv_df = pd.concat([surv_df, event_dummies, lu_dummies], axis=1)
surv_df = surv_df.drop(columns=['event_id', 'land_use', 'is_censored'])

cph = CoxPHFitter()
cph.fit(
    surv_df,
    duration_col='recovery_days',
    event_col='event_observed'
)
cph.print_summary()

# in_buffer的风险比
# HR > 1 → 缓冲区像素恢复更快（"恢复"的风险率更高 = 好事）
print(f"\nin_buffer的风险比 (Hazard Ratio): {np.exp(cph.params_['in_buffer']):.3f}")
```

**关键参数**：
- 恢复阈值：先用90%，同时尝试80%和95%作为敏感性检验
- `is_censored` 标志非常关键——部分像素（尤其Maria）可能在观测窗口内未恢复；生存分析通过右删失正确处理这种情况
- 报告：`in_buffer` 的风险比（HR）。HR > 1 表示缓冲区像素恢复更快

**Kaplan-Meier生存曲线**（可视化）：
```python
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()

fig, ax = plt.subplots(figsize=(10, 6))
for label, group in surv_df.groupby('in_buffer'):
    name = '缓冲区' if label == 1 else '非缓冲区'
    kmf.fit(group['recovery_days'], group['event_observed'], label=name)
    kmf.plot_survival_function(ax=ax)

ax.set_xlabel('灾后天数')
ax.set_ylabel('尚未恢复的比例')
ax.set_title('恢复生存曲线：缓冲区 vs 非缓冲区')
plt.tight_layout()
plt.savefig('result/plots/cross_event/survival_curve_buffer_vs_nonbuffer.png', dpi=200)
```

---

### 3.7 所有模型规格汇总

| 模型 | 公式 | Python库 | 核心系数 | 预期方向 |
|---|---|---|---|---|
| OLS基线 | `delta_ntl ~ in_buffer * pre_mean_ntl + C(land_use) + C(event_id)` | `statsmodels.ols` | β₁ (in_buffer) | 正 |
| 混合效应 | `delta_ntl ~ in_buffer * pre_mean_ntl + C(land_use)`, groups=event_id | `statsmodels.mixedlm` | β₁ (in_buffer) | 正 |
| Logistic | `is_damaged ~ in_buffer * pre_mean_ntl + C(land_use) + C(event_id)` | `statsmodels.logit` | β₁ (in_buffer) | 负 |
| Cox PH | `recovery_days ~ in_buffer + pre_mean_ntl + land_use_dummies + event_dummies` | `lifelines.CoxPHFitter` | HR(in_buffer) | > 1.0 |

---

### 3.8 推荐执行顺序

```
第一步：构建像素数据集
    → all_events_pixel_panel.parquet（约50K–200K行）
    → 验证：检查每事件像素数、in_buffer比例、delta_ntl分布

第二步：OLS基线模型
    → 快速检查：β₁是否为正？是否显著？
    → 如果β₁不显著：检查数据质量，尝试不同的像素过滤条件
    → 如果β₁显著：进入混合效应

第三步：混合效应模型
    → 论文的主要结果
    → 报告：β₁、β₃、随机效应方差、ICC

第四步：Logistic回归
    → 二分类损害分析
    → 报告：in_buffer的优势比 + 95%置信区间

第五步：生存分析
    → 恢复速度分析
    → 报告：风险比、Kaplan-Meier曲线图

第六步：稳健性检验
    → 不同像素过滤阈值（pre_ntl > 0.3, 0.5, 1.0）
    → 不同损害阈值（-5%, -10%, -15%, -20%）
    → 不同恢复阈值（80%, 90%, 95%）
    → 按设施类型子样本（仅power_plant + aerodrome）
    → 去掉最弱事件（地震）重新跑
```

---

### 3.9 扩展分析建议（可选）

1. **距离衰减模型**：用连续的 `distance_to_nearest`（取对数）替代二元的 `in_buffer`。测试备用发电机信号是否随距离衰减：`delta_ntl ~ log(distance) + pre_ntl + C(land_use) + C(event_id)`。如果距离系数显著为负，说明离设施越近的像素韧性越强。

2. **设施类型子模型**：对 `power_plant` + `aerodrome`（信号最强）单独跑回归，对 `hospital` + `fire_station` + `police`（信号较弱）单独跑。比较β₁的大小。

3. **空间自相关检验（Moran's I）**：用 `pysal.esda.Moran` 检验OLS残差是否有空间自相关。如果显著，考虑空间误差模型或按设施聚类的标准误。

4. **BEAST增强因变量**：用BEAST趋势分量的变化替代原始 `delta_ntl` 作为因变量。这去除了季节和噪声混淆，应该产生更精确的估计。

---

## 关键参考文献

- **Zhang et al. (2023)**：NTL变化率方法用于灾害损害评估，损害阈值（10%）
- **Wang et al. (2018)**：NASA Black Marble团队，恢复指标（起始日、时长、速率），停电监测
- **Zheng et al. (2025, RSE 319)**：BEAST分解用于NTL分析，偏相关分析，恢复不平等——像素级分析的方法论模板
- **Zhao et al. (2019)**：BEAST算法（贝叶斯突变/季节/趋势估计器）

---

## 运行环境

```
Python 3.10+
numpy, pandas, geopandas, rasterio, shapely
matplotlib, contextily, seaborn
statsmodels          # OLS、混合效应、logistic
lifelines            # 生存分析
scipy                # cKDTree最近邻
Rbeast               # BEAST时间序列分解（可选）
pyarrow              # parquet读写
```

安装：
```bash
pip install numpy pandas geopandas rasterio shapely matplotlib contextily seaborn statsmodels lifelines scipy Rbeast pyarrow
```

---

## 联系方式

数据获取与EDA流水线相关问题：Zhiyuan
建模相关问题：[同学名字]
指导教授：李晓江
