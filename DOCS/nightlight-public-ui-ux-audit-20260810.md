# Nightlight Public UI/UX 全面审查与优化方案

审查日期：2026-08-10

审查对象：`project/nightlight-public` 五条公开路由

运行环境：Vite 开发服务器、Chromium 内置浏览器、桌面与 391 × 844 CSS px 移动视口
结论边界：这是技术、视觉和启发式审查，不是正式 WCAG 认证、辅助技术全覆盖测试或真实用户理解度研究。

## 1. 执行摘要

当前界面不是“设计方向错误”，而是一个有辨识度的科研编辑型界面被以下问题拖累：

1. 开发模式 CSP 与 Vite 的动态样式/HMR 不兼容，导致样式可能无法正常加载。该问题已修复，生产 CSP 保持严格。
2. 移动端主导航仍依赖横向滚动条；当前项会被脚本居中，但用户只能看到约 3 个入口，导航发现性和观感都较差。
3. 每次路由进入后，整块 H1 获得高亮外框。焦点确实可见，但视觉面积过大，像调试框而不是产品级焦点反馈。
4. 页面标题和引导文案占据过多首屏空间。Atlas 的主要操作、Findings 的关键数字、Methods 的工作流都出现得太晚。
5. Atlas 的选择状态全部是内存状态，URL 不反映 Explore/Compare、筛选条件或比较对象，因此无法刷新保留、分享或用浏览器前进/后退恢复任务上下文。
6. Findings、Methods、Credits 和 Atlas Compare 都是超长页面，却没有页内目录、当前位置提示或紧凑的渐进展开结构。
7. Skip link 与 `createWebHashHistory` 争用 URL hash。实测激活 `href="#main-content"` 后，URL 仍停在 `#/` 且焦点仍在 `BODY`，没有跳到主内容。

建议保留现有的深绿、琥珀、酸性色、衬线标题、等宽元数据和“观测站/研究档案”气质；优化重点应是信息架构、任务路径、移动导航、焦点反馈和内容密度，而不是换一套通用 Dashboard 皮肤。

## 2. 已完成的兼容性修复

### 2.1 根因

开发模式下，Vite 会把导入的 CSS 作为动态 `<style>` 注入页面，并通过 WebSocket 建立 HMR 连接；原始 CSP 同时要求 `style-src 'self'` 和 `connect-src 'none'`，因此会拦截开发样式与 HMR。Vite 官方支持用 `html.cspNonce` 给注入的 script/style/link 添加 nonce。

### 2.2 修复方式

- `vite.config.js` 仅在 `command === 'serve' && !isPreview` 时生成随机 nonce。
- 开发 HTML 将 `style-src 'self'` 收窄扩展为 `style-src 'self' 'nonce-…'`。
- 开发 HTML 将 `connect-src 'none'` 改为 `connect-src 'self' ws:`，只允许本地开发 HMR。
- 不使用 `'unsafe-inline'`。
- `build` 和 `preview` 不注入开发 nonce，不放宽生产 CSP。
- CSP 文本如果发生漂移，转换函数 fail closed，而不是静默放宽错误策略。

代码证据：

- `project/nightlight-public/vite.config.js:12`
- `project/nightlight-public/vite.config.js:28`
- `project/nightlight-public/tests/static-shell.test.js:21`
- `project/nightlight-public/tests/static-shell.test.js:50`

### 2.3 验证结果

- 针对性 Vitest：26/26 通过。
- 开发运行时：nonce 与 Vite nonce meta 一致；动态样式生效；页面背景、排版和组件样式恢复；没有本次 CSP 问题产生的报错。
- 独立的现有 warning：Vite 指出 Findings 的 `<figcaption>` 位于 `<div>` 内，不符合 HTML 内容模型；这不是 CSP 回归，但应在 UI 修复阶段处理。
- 生产构建：通过，40 个模块完成转换。
- 公共发布边界：`npm run verify:public -- --require-dist` 通过。
- 完整 `npm run validate` 的已知缺口：169 项测试中 165 项通过，4 项在测试前置阶段因仓库现有文件 `DOCS/active/p2-p3-solo-evidence-performance-20260810/p2-evidence.md` 缺失而出现 `ENOENT`。该缺失与本次 CSP 修复无关，本次没有伪造或补写证据文件。

官方依据：

- [Vite shared options: html.cspNonce](https://vite.dev/config/shared-options.html)
- [Vite features: CSP nonce and development CSS injection](https://vite.dev/guide/features.html)
- [Vite plugin API: command and isPreview](https://vite.dev/guide/api-plugin.html)
- [Vite CLI: preview serves a production build](https://vite.dev/guide/cli)

## 3. 审查范围与方法

### 3.1 用户流程

审查按真实访客的理解路径展开：

1. 在 Overview 理解项目是什么、能看什么、不能看什么。
2. 进入 Study Atlas 搜索或筛选事件。
3. 选择一个事件并读取 Evidence Passport。
4. 切换 Compare，选择预设或自定义两个事件。
5. 在 Findings 区分 descriptive fit、damage ranking 和 recovery transport。
6. 在 Methods 核对输入、处理、建模、准入和公开产物边界。
7. 在 Credits / Policy 核对作者、协作者、来源、权利、运行时和发布边界。

### 3.2 检查维度

- 桌面和移动端视觉层级、布局、密度、断点和溢出。
- 导航、筛选、单事件选择、Compare 预设、交换和 live region。
- 键盘入口、Skip link、路由焦点、可见焦点、语义元素和目标尺寸。
- 图表文本替代、表格、空状态、`Unavailable` / `Not assessed` 语义。
- URL 状态、刷新/分享/历史恢复能力。
- CSS token、字体尺寸、动效、forced-colors、reduced-motion 和维护性。
- 当前官方 W3C、USWDS、GOV.UK、Vite 和 Vercel Web Interface Guidelines。

### 3.3 截图证据目录

`C:/Users/raede/.codex/visualizations/2026/08/10/019fec05-dfa3-7861-a020-c504de44cf5f/nightlight-public-runtime/ui-audit-2026-08-10`

主要步骤与截图：

| 步骤 | 状态 | 截图 |
| --- | --- | --- |
| 1 | Overview 桌面首屏 | `02-overview-desktop.png` |
| 2 | Overview 边界说明展开 | `03-overview-disclosure-desktop.png` |
| 3 | Atlas Explore 桌面 | `04-atlas-explore-desktop.png` |
| 4 | Atlas Compare 桌面 | `05-atlas-compare-desktop.png` |
| 5 | Compare 选择 “Same storm, two places” | `06-atlas-guided-comparison-desktop.png` |
| 6 | Findings 桌面 | `07-findings-desktop.png` |
| 7 | Methods 桌面 | `08-methods-desktop.png` |
| 8 | Credits 桌面 | `09-credits-desktop.png` |
| 9 | Overview 移动首屏 | `10-overview-mobile.png` |
| 10 | Atlas Explore 移动首屏 | `11-atlas-explore-mobile.png` |
| 11 | Atlas Compare 移动状态 | `12-atlas-compare-mobile.png` |
| 12 | Findings 移动首屏 | `13-findings-mobile.png` |
| 13 | Methods 移动首屏 | `14-methods-mobile.png` |
| 14 | Credits 移动首屏 | `15-credits-mobile.png` |

## 4. 当前设计的优势：必须保留

### 4.1 独特且适合项目的视觉语言

- 深绿黑底、暖琥珀、酸性黄绿、衬线标题和等宽元信息形成了清晰的“夜间观测 / 科研档案”身份。
- 首页的 signal illustration 与项目题目高度一致；没有依赖 stock photo 或通用卡片模板。
- 网格、细边框、编号和 `Field note` 语汇使五条路由具有统一的文档系统感。

### 4.2 科学边界表达成熟

- `Unavailable`、`Not assessed` 与零值严格分离。
- Evidence Passport 明确不是 recovery measure、event grade 或 ranking。
- Findings 明确区分 R²、AUC、描述性拟合、damage ranking 和未建立的 recovery transport。
- Compare 不生成相似度、排名或综合得分，且把 measurement comparability 的未知条件显式保留。
- 这些边界是项目最有辨识度的内容资产，任何压缩都只能改变呈现顺序，不能删除其语义。

### 4.3 语义与响应式基础良好

- 主要交互使用原生 `a`、`button`、`input`、`select`、`details` 和 `table`。
- Atlas 的 SVG 点只负责展示，真正选择入口是可键盘操作的原生按钮。
- 图表包含 `<title>` / `<desc>`，关键图表还有文本表格替代。
- Compare 和选择读数使用 polite live region。
- 五条路由在 391 px 测试下均无文档级横向溢出。
- `prefers-reduced-motion` 和 `forced-colors` 已有专门分支。
- 抽样对比度结果：普通 muted 文本约 8.34:1；最弱的导航序号样本约 4.65:1。当前主要风险是字号过小，而不是抽样颜色本身不达 4.5:1。

### 4.4 运行时轻量且边界清楚

- 当前构建约 204,624 bytes：JS 161,266 bytes、CSS 39,912 bytes、HTML 980 bytes。
- 无外部字体、地图瓦片、分析脚本或后台数据请求。
- 这个性能和隐私优势应设为后续 redesign 的硬约束。

## 5. 主要问题与优先级

### P0-1：Skip link 在 hash router 中失效

**验证事实**

- `App.vue:55` 使用 `href="#main-content"`。
- `src/main.js` 使用 `createWebHashHistory`，同一个 URL hash 已被路由占用。
- 冷启动页面激活 Skip link 后，地址仍为 `#/`，`#main-content` 没有成为可用锚点，焦点仍在 `BODY`。

**影响**

- 键盘用户无法可靠跳过品牌和五项主导航。
- 静态测试只检查了 markup 存在，没有证明激活后的焦点结果。

**修复**

- 保留可发现的 Skip link，但用 `@click.prevent` 调用单独的 `focusMainContent()`。
- 只在这个显式用户操作中调用 `main.focus()` 和滚动，不影响初始 Tab 顺序或路由自动焦点。
- 增加真实浏览器断言：冷启动首次 Tab 聚焦 Skip link；Enter 后 `document.activeElement === main`；route hash 保持不变。

### P0-2：移动端导航依赖横向滚动条

**验证事实**

- 391 px 下 nav `clientWidth = 369`、`scrollWidth = 640`、`overflow-x = auto`。
- 当前项会被 `App.vue:19-30` 居中，因此“当前项不可见”的旧问题已缓解。
- 但首屏只显示约 3 个入口，Windows 风格横向滚动条始终可见，Methods / Credits 等入口的存在需要滚动后才知道。

**影响**

- 导航发现性差；滚动条与精致的编辑型视觉系统冲突。
- 超长页下移动 header 不是 sticky，离开顶部后无法快速换页。

**修复**

- 小于 900 px 时使用明确的 `Menu` 按钮和纵向导航列表；按钮需有 `aria-expanded`、`aria-controls`，Escape 可关闭。
- 保留所有 5 个文本标签和 `aria-current="page"`。
- 菜单可以在文档流中展开，避免不必要的焦点陷阱和 overlay 复杂度。
- 将 `Public / aggregate-only` 状态放进移动菜单或品牌区，不再完全隐藏。

USWDS 建议主导航使用短、清晰标签、逻辑 Tab 顺序和可靠 Skip navigation；GOV.UK 的 service navigation 在移动端默认把多项导航折叠到带可访问标签的菜单中。

### P0-3：路由 H1 焦点反馈过强

**验证事实**

- `App.vue:38-46` 在路由进入后聚焦整块 H1。
- 全局 `:focus-visible` 给 H1 绘制 2 px 外框；移动 Credits 截图中外框覆盖 337 × 223 px，桌面中则横跨大块内容区域。
- 焦点语义正确，问题是视觉反馈的形态。

**影响**

- 新页面打开后首先看到像调试边框一样的大框，破坏标题排版。
- 用户很容易把它理解为错误/选中状态，而不是键盘定位。

**修复**

- 保留路由焦点，但为 `.focus-target:focus-visible` 定义专门的低干扰指示器，例如 3–4 px 左侧强调线和轻微背景，而不是整个 block outline。
- 所有真正可操作元素继续使用清晰的全局 outline。
- `focus({ preventScroll: true })` 与 router 的 `scrollBehavior` 分工，避免程序性焦点二次滚动。

### P1-1：首屏信息优先级失衡

391 × 844 稳定测量：

| 页面 | 文档高度 | 首要内容位置 | 结论 |
| --- | ---: | ---: | --- |
| Overview | 3,174 px | CTA 在首屏内 | 基本可用，但插图和指标顺序可更紧凑 |
| Atlas Explore | 4,745 px | mode 690；filter 949；map 1,164；Passport 2,217 | 任务控件出现太晚 |
| Atlas Compare | 7,706 px | mode 690；Compare 区 949；preset 1,333；selectors 1,709 | 核心选择器超过 2 个屏高 |
| Findings | 6,289 px | 第一分析段 849；主要图表 1,214 | 首屏没有关键数字 |
| Methods | 4,480 px | 第一流程步骤 660 | 尚可，但步骤卡片过高 |
| Credits | 3,793 px | 第一 policy card 661 | 尚可，标题仍过大 |

**修复原则**

- 首页可以保留展示型 hero；内页不应继续使用同样大的展示标题。
- 移动端内页 H1 从当前约 66 px 调整到 44–52 px，line-height 约 0.95–1.05。
- Atlas 的模式切换和任务控件应在首屏可见；Findings 的核心结论应在 1–1.5 屏内出现。
- 标题下的解释文案保持 55–70 字符行宽；长免责声明通过“短结论 + 展开详细边界”呈现。

### P1-2：Atlas 操作顺序和反馈距离过长

**Explore 当前顺序**

`H1/lede → mode → definitions → filters → map → 25-item index → Evidence Passport`

用户选择事件后，最重要的 Evidence Passport 位于地图和列表之后。移动端即使事件按钮可用，结果与操作之间仍有很长的视觉距离。

**推荐顺序**

`紧凑标题 → mode → filters → 当前选择摘要 → Map / Event list 切换 → Evidence Passport → 完整边界`

- 移动端默认显示 Event list，地图作为并列 tab；该站的主要任务是证据检查，不是空间导航。
- 选择事件后，在控制区立即显示 sticky/compact selection summary，并提供 “View Passport” 锚点。
- Passport 放在列表前或以 drawer/inline panel 紧邻选择结果；不要要求读完 25 项列表才看到结果。
- 空筛选状态保留当前清晰文案，并增加 “Clear filters” 动作。

**Compare 推荐顺序**

`mode → A/B selectors → compatibility headline → guided presets → key component differences → full component ledger → detailed measurement boundary`

- 自定义选择器应比 editorial presets 更早出现。
- 结果先展示“能比较什么 / 不能比较什么”和 2–3 个关键组件差异，再展开完整 ledger。
- 详细的 measurement boundary 不能删除，但可放入默认展开的 summary box + 可展开完整条件。

### P1-3：Atlas 状态不可分享、不可恢复

**验证事实**

- `AtlasView.vue:20-25` 的 mode、query、hazard、selected event、A/B pair 全部使用本地 `ref`。
- 实测输入搜索词并切到 Compare 后 URL 仍只为 `#/atlas`。

**推荐 URL 契约**

```text
#/atlas?mode=explore&q=ian&hazard=Tropical%20cyclone&event=hurricane-ian-charlotte
#/atlas?mode=compare&a=hurricane-ian-charlotte&b=hurricane-ian-fort-myers&preset=same-storm
```

- 解析时严格校验枚举和 event ID；无效值回退到安全默认值。
- 用户操作用 router replace/push 更新 URL，避免每次字符输入污染历史；搜索输入可 debounce 150–250 ms。
- reload、复制链接、浏览器 back/forward 都应恢复同一界面。
- URL 只包含公开 event ID 和界面状态，不包含受限数据。

### P1-4：长页缺乏页内定位与渐进披露

Findings、Methods、Credits 和 Atlas Compare 均超过 3,700 px，Compare 达 7,706 px。当前只能线性滚动。

**修复**

- Findings、Methods、Credits 加页内目录；桌面为 sticky rail，移动为紧凑 “On this page” disclosure。
- 使用真实 heading anchor，并将 `scroll-margin-top` 与 header 实际高度对齐。
- Findings 首部增加 3 个角色卡：`Description`、`Damage ranking`、`Recovery transport unavailable`，避免先读长段落才理解结论。
- Methods 将 5 个超高步骤压缩成可扫描 timeline：标题、1 句说明、输入/输出 badge；完整说明按步骤展开。
- Credits 将 Authorship、Sources & rights、Runtime & privacy、Known limits、License 分组，并在顶部加摘要目录。

USWDS 的 in-page navigation 专门用于长页，并要求键盘导航、当前段落提示和移动适配；GOV.UK 也建议当内容过多时先考虑拆页或更清晰的内容结构，而不是把所有内容塞进一个 accordion。

### P1-5：功能性小字过多

当前存在 0.54–0.67 rem 的导航序号、状态、图表元数据、footer 和组件标签；在本次 391 px 环境中约为 8.64–10.72 px。抽样颜色对比度尚可，但可读性和扫描速度不足。

**修复**

- 功能性/可操作标签最低 12 px，正文辅助信息最低 13–14 px。
- 只有非必要编号或纯装饰 metadata 可降到 11 px；不能用极小字号承载限制条件或状态含义。
- 关键数字和比较列使用 `font-variant-numeric: tabular-nums`。
- 搜索和 hazard select 的当前可见高度约 30–31 px；提升到至少 44 px 作为产品目标。WCAG 2.2 AA 的最低 pointer target 是 24 × 24 px，44 × 44 px 是更稳妥的增强目标。

### P1-6：表单与交互细节仍可补齐

- Atlas search、hazard select、A/B selects 缺少稳定 `name`；搜索 placeholder 应使用省略号 `…`。
- 为触控交互增加 `touch-action: manipulation`。
- `html` 明确声明 `color-scheme: dark`，而不是只依赖 `:root`。
- 移动全宽/贴边布局应考虑 `env(safe-area-inset-*)`。
- active/hover/focus 状态已存在，但 radio mode 的 focus 容器可以用 `:focus-within` 强化，让焦点落在 16 px 原生 radio 时整个选项块都清晰反馈。

### P1-7：Findings 图表的 figcaption 结构无效

**验证事实**

- 开发服务器在加载 Findings 时输出 Vite/Vue warning：`<figcaption> cannot be child of <div>`。
- `FindingsView.vue:60-76` 的外层是 `<figure>`，但 `<figcaption>` 被放进内部 `.finding-hero__chart` `<div>`。

**修复**

- 将 `<figcaption>` 移为 `<figure>` 的直接末尾子元素，并用 CSS 让它跨越所需 grid column；或者把真正包含图和 caption 的内部容器改成独立 `<figure>`，但不要形成嵌套语义混乱。
- 增加开发构建/浏览器 smoke 的 warning gate；完成后本项目自身的 Vite warning 应为 0。

### P2-1：永久自动动效可进一步克制

当前 status pulse、signal scan 和 event pulse 是无限循环；reduced-motion 分支能停止它们，这是优点。但对科研内容站而言，永久扫描线和呼吸灯没有任务价值。

建议：

- status 采用静态高对比圆点；只在状态发生变化时做一次短暂过渡。
- signal scan 默认静态，若保留，只运行一次或限制在 hover/focus 场景。
- 保留用户可关闭/系统 reduced-motion 适配。

### P2-2：源码维护边界过于集中

- `main.css` 约 2,638 行，承载 token、shell、五页布局、响应式、forced-colors 和 motion。
- `AtlasView.vue` 同时承担状态、筛选、地图、索引、Passport、Compare presets、selectors、summary 和 ledger。

建议分阶段而不是先重构：

1. 先完成 P0/P1 行为修复和浏览器测试。
2. 再提取已有稳定边界：`SiteNavigation`、`PageHeader`、`InPageNav`、`AtlasModeSwitch`、`EventFilters`、`EventSelectionSummary`、`ComparisonSelectors`。
3. CSS 使用 `@layer tokens, base, components, pages, utilities, accessibility` 或按现有构建允许的文件拆分。
4. 新文件必须同步更新公共发布 allowlist，并通过 fail-closed verifier；不能绕过现有边界。

## 6. 逐页优化方案

### 6.1 Overview

**保留**

- “Reading recovery in the dark.” 主叙事。
- 恢复曲线插图和“illustration only”边界。
- Atlas 与 Findings 两个直接 CTA。

**调整**

- 桌面 hero 标题缩小约 10–15%，避免文字压过图表；移动保持 3 行以内。
- 移动顺序改为：标题 → 1 句价值说明 → CTA → 3 个紧凑事实 → 插图 → 详细边界。
- 在首屏加入一句明确任务说明：`Explore events, compare evidence states, and inspect where the model does not travel.`
- metric strip 由 3 个高卡片改为紧凑数据条，保留单位/样本/限制。

### 6.2 Study Atlas / Explore

**调整**

- mode switch 紧跟 page title，不再被大段 lede 推到首屏底部。
- 搜索、hazard、数量反馈放在一个 sticky filter bar；移动堆叠但每个控件 44 px 高。
- 移动使用 `List / Map` 两态切换，List 默认。
- 当前选择 summary 永远靠近控件，包含 event、place、year、Passport band 和直达 Passport 的动作。
- 25 项列表支持 hazard grouping、清除筛选和当前项明显标识；不需要虚拟化，因为总数仅 25。
- map 继续保持 schematic，绝不升级为“精确轨迹”或“安全/风险地图”。

### 6.3 Study Atlas / Compare

**调整**

- A/B selects 上移到 Compare header 之后；Swap 置于两个字段之间/下方并保持 48 px 目标。
- preset 作为“Try a guided pair”，不是主入口。
- 顶部结果 summary 只回答三件事：公开分类是否对齐、Passport 是否齐全、测量可比性是否建立。
- 每个 component row 移动端默认显示差异标题和 A/B 状态，详细定义按 row 展开。
- 保留 `No similarity score is computed` 和 outcome boundary，放在 summary 下的 persistent banner。

### 6.4 Findings

**调整**

- 首屏直接给出角色矩阵摘要，而不是先给 2 个长段落。
- 将 `0.7603 R²` 和 `0.4814 AUC` 分成不同任务卡，视觉上禁止共用同一“高低好坏”色阶。
- `Recovery transport` 以明确 unavailable 卡出现，避免用户把 damage ranking AUC 误读为恢复迁移能力。
- 长证据卡改为定义列表 + 展开 lineage，默认保留 metric、value/unit、role、supports/does not support。
- 加页内目录：Attractive result / Harder test / Role matrix / Evidence cards。

### 6.5 Methods

**调整**

- 五步 timeline 每步首层只显示：编号、标题、1 句动作、输入/输出状态。
- “Private boundary / Public result / Published boundary” 使用稳定 badge 体系。
- 完整说明在 details 中展开；默认页面高度目标下降 25–35%。
- 页首给出 `Private inputs → processed signals → place-level model → admission → public artifact` 文字流程，不用新图形资产也能快速理解。

### 6.6 Credits / Policy

**调整**

- 内页 H1 缩小，避免 337 × 223 px 的标题占据半个移动首屏。
- 第一屏显示四个 trust facts：Aggregate-only / Local assets / No analytics / User-activated external links only。
- Author、Collaborator、Source、No endorsement 保持完整，但用分组标题提高扫描。
- LICENSE、CREDITS、DATA_POLICY、THIRD_PARTY_NOTICES 使用可访问的文件链接或明确说明其仓库位置，而不是只用粗体文件名。

## 7. 目标信息架构

### 全局导航

```text
Overview
Study Atlas
  Explore
  Compare
Findings
Methods
Credits / Policy
```

不新增第六条一级路由；Explore / Compare 继续属于 Atlas，但其状态必须可深链。

### 页面模板

```text
Global header
Compact page header
Task controls or page summary
Primary evidence/result
Interpretation boundary
Detailed evidence / methods
In-page navigation on long pages
Global footer
```

## 8. 设计系统建议

### 8.1 保留的 token

- `--ink`, `--ink-raised`, `--paper`, `--amber`, `--acid`, `--rust`。
- serif / sans / mono 三字体角色。
- 1 px scientific grid 和细边框。

### 8.2 新增语义 token

```css
--text-primary
--text-secondary
--text-metadata
--surface-page
--surface-panel
--surface-active
--state-available
--state-limited
--state-unavailable
--state-unassessed
--space-page-inline
--header-height
--focus-interactive
--focus-route
```

避免直接在组件中不断重复 rgba；状态颜色必须同时配合文字/边框，不仅靠颜色。

### 8.3 类型比例

| 角色 | 桌面 | 移动 |
| --- | ---: | ---: |
| Overview display | 88–120 px | 52–64 px |
| Inner-page H1 | 64–88 px | 44–52 px |
| H2 | 36–56 px | 30–38 px |
| Body lead | 18–22 px | 17–19 px |
| Body | 16–18 px | 16–18 px |
| Metadata | 12–14 px | 12–14 px |

## 9. 实施路线图

### Phase A — 稳定性与可达性（1–2 个开发日）

- 修复 Skip link 的 hash-router 冲突。
- 将移动横向导航改为可访问 Menu。
- 重新设计 route-focus 指示器。
- 增加 sticky offset / scroll-margin、44 px controls、最小字号、safe-area、touch-action。
- 补真实浏览器测试，不再只用 regex 证明焦点行为。

### Phase B — 信息层级与长页导航（2–4 个开发日）

- 压缩所有内页 hero。
- Overview 事实前移。
- Findings / Methods / Credits 增加页内目录和 compact summary。
- Methods 与证据卡采用渐进披露。

### Phase C — Atlas 工作流重排与 URL 状态（3–5 个开发日）

- URL 同步 mode、filters、event 和 A/B pair。
- Explore 移动 List / Map 模式。
- 当前选择 summary 与 Passport 前移。
- Compare selectors 前移、结果 summary 分层、component row 展开。

### Phase D — 视觉精修与维护性（2–3 个开发日）

- 收敛 token、字号、间距和边框。
- 移除或限制永久动效。
- 提取已稳定且有多个消费者的组件边界。
- 保持零新依赖优先；若增加文件，显式更新并验证发布 allowlist。

### Phase E — 验证与用户证据（2–3 个开发日 + 用户研究）

- Chromium + Edge/Firefox 最小矩阵。
- 320 / 375 / 391 / 768 / 1024 / 1440 px。
- 200% zoom/reflow、WCAG text spacing、forced colors、reduced motion。
- Windows NVDA + 键盘 smoke；必要时 VoiceOver 补充。
- 5 名目标读者的理解度测试：能否正确回答 R²、AUC、Passport、Unavailable、aggregate-only 分别意味着什么。

单人实现预计 10–17 个开发日；真实用户招募和迭代另计。

## 10. 验收标准

### 导航与焦点

- 冷启动首次 Tab 聚焦 Skip link；Enter 后焦点落在 `main`，route hash 不改变。
- 320–391 px 不显示主导航横向滚动条；全部 5 条路由在一次明确 Menu 操作内可达。
- active route 同时有文字/结构状态和 `aria-current="page"`。
- 路由进入后有清晰但不包围整块 H1 的焦点提示。
- sticky header 不遮挡任何获得焦点的控件或 heading anchor。

### 布局与可读性

- 五条路由及 Atlas Compare 在 320 px 无文档横向溢出。
- 移动端内页 H1 不超过约 52 px，且不因硬换行产生单字/窄列。
- 功能性文字最低 12 px；输入与选择控件高度至少 44 px。
- Findings 的关键角色和至少一个关键数字在 1.5 个移动视口内可见。
- Atlas mode 在首屏内，Explore filters / Compare selectors 最多一次短滚动可达。

### 状态与交互

- Atlas URL 可复现 mode、filters、selected event 和 comparison pair。
- reload、复制链接、back/forward 恢复同一公开状态。
- Event 选择后立即看到 selection summary，并能直接到 Passport。
- Compare 更新只有一个简明 live announcement，不重复朗读整页。
- 空状态提供清除筛选动作；Unavailable / Not assessed 永不表示为 0。

### 科学与发布边界

- 不新增 recovery score、safety score、risk rank、相似度分数或预测性暗示。
- 地图仍是 broad orientation，不暗示 GPS matching 或精确灾害轨迹。
- 生产 CSP 仍为 local-only；无 analytics、外部 font/map/runtime data。
- 公共构建与 release verifier 通过；无受限细粒度记录进入 dist。
- 初始 JS/CSS 体积不显著高于当前基线；建议以未压缩总构建 ≤ 250 KB 作为第一阶段预算。

## 11. 验证缺口与未决风险

- 本次没有进行 NVDA、JAWS、VoiceOver、speech input 或 switch access 的人工全流程测试。
- 浏览器自动化与视觉审查不能证明读者真正理解统计角色和研究边界。
- 自动对比度抽样不是全站每种状态的正式 WCAG 审计。
- 当前开发日志仍有 Findings `figcaption` 内容模型 warning；它不阻塞 CSP 修复，但在 UI 重排前应先清零。
- 完整 validation 当前被仓库现有、与本次修复无关的缺失证据文件阻塞；不能把 targeted pass 表述为完整发布批准。
- Atlas URL 状态、移动菜单和页内导航尚是优化方案，当前尚未实现。

## 12. 官方参考

- [W3C WCAG 2.2](https://www.w3.org/TR/WCAG22/)
- [W3C Understanding Target Size (Minimum)](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum)
- [W3C Understanding Reflow](https://www.w3.org/WAI/WCAG22/Understanding/reflow)
- [W3C Page Structure Tutorial](https://www.w3.org/WAI/tutorials/page-structure/)
- [W3C Understanding Animation from Interactions](https://www.w3.org/WAI/WCAG22/Understanding/animation-from-interactions.html)
- [USWDS Header](https://designsystem.digital.gov/components/header/)
- [USWDS In-page Navigation](https://designsystem.digital.gov/components/in-page-navigation/)
- [GOV.UK Service Navigation](https://design-system.service.gov.uk/components/service-navigation/)
- [GOV.UK Accordion](https://design-system.service.gov.uk/components/accordion/)
- [Vercel Web Interface Guidelines](https://github.com/vercel-labs/web-interface-guidelines/blob/main/command.md)
- [Vite CSP nonce documentation](https://vite.dev/config/shared-options.html)

## 13. 最终建议

采用“保留品牌、重排任务、先修可达性”的方案：

1. 立即完成 Skip link、移动菜单、route focus 三项 P0。
2. 将内页标题压缩并加入长页页内导航。
3. 把 Atlas 的 controls、selection summary 和 Compare selectors 前移，同时实现 URL 状态。
4. 再进行组件拆分、token 收敛和动效减法。
5. 最后用人工辅助技术测试和真实理解度测试决定文案是否还需进一步简化。

这样可以保留当前最有价值的科研编辑气质、科学克制和轻量运行边界，同时解决用户第一眼觉得“像坏了/像调试界面/不知道先点哪里”的核心问题。
