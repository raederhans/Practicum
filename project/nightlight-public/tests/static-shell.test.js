import { readFile } from 'node:fs/promises'
import { describe, expect, it } from 'vitest'

import createViteConfig, { applyDevelopmentCsp } from '../vite.config.js'

describe('offline-first HTML shell', () => {
  it('sets a local-only CSP and no-referrer policy', async () => {
    const html = await readFile(new URL('../index.html', import.meta.url), 'utf8')

    expect(html).toMatch(/http-equiv=["']Content-Security-Policy["']/i)
    expect(html).toMatch(/default-src 'self'/i)
    expect(html).toMatch(/connect-src 'none'/i)
    expect(html).toMatch(/name=["']referrer["']\s+content=["']no-referrer["']/i)
  })

  it('does not reference external runtime resources', async () => {
    const html = await readFile(new URL('../index.html', import.meta.url), 'utf8')
    expect(html).not.toMatch(/(?:src|href)\s*=\s*["']https?:\/\//i)
  })

  it('adds nonce-authorized Vite styles and HMR connections only for development', async () => {
    const html = await readFile(new URL('../index.html', import.meta.url), 'utf8')
    const developmentHtml = applyDevelopmentCsp(html, 'unit-test-nonce')
    const developmentConfig = createViteConfig({
      command: 'serve',
      isPreview: false,
      mode: 'development',
    })
    const buildConfig = createViteConfig({
      command: 'build',
      isPreview: false,
      mode: 'production',
    })
    const previewConfig = createViteConfig({
      command: 'serve',
      isPreview: true,
      mode: 'production',
    })

    expect(developmentHtml).toContain("style-src 'self' 'nonce-unit-test-nonce'")
    expect(developmentHtml).toContain("connect-src 'self' ws:")
    expect(developmentHtml).not.toContain("'unsafe-inline'")
    expect(developmentConfig.html.cspNonce).toMatch(/^[A-Za-z0-9+/]{24}$/)
    expect(buildConfig.html).toBeUndefined()
    expect(previewConfig.html).toBeUndefined()
    expect(html).toContain("style-src 'self'")
    expect(html).toContain("connect-src 'none'")
  })

  it('fails closed when either production CSP directive drifts', () => {
    expect(() => applyDevelopmentCsp("style-src 'self'", 'unit-test-nonce'))
      .toThrow(/could not find the production policy/i)
    expect(() => applyDevelopmentCsp("connect-src 'none'", 'unit-test-nonce'))
      .toThrow(/could not find the production policy/i)
  })
})

describe('atlas keyboard contract', () => {
  it('keeps SVG points presentational and exposes selection on native buttons', async () => {
    const atlas = await readFile(new URL('../src/views/AtlasView.vue', import.meta.url), 'utf8')

    expect(atlas).not.toMatch(/class="atlas-point"[\s\S]*?tabindex="0"/)
    expect(atlas).not.toMatch(/class="atlas-point"[\s\S]*?role="button"/)
    expect(atlas).toMatch(/:aria-pressed="selectedEvent\?\.id === event\.id"/)
  })

  it('uses native radios, selects, and buttons for Compare Mode keyboard access', async () => {
    const atlas = await readFile(new URL('../src/views/AtlasView.vue', import.meta.url), 'utf8')

    expect(atlas.match(/type="radio"/g)).toHaveLength(2)
    expect(atlas.match(/name="atlas-evidence-view"/g)).toHaveLength(2)
    expect(atlas.match(/<select/g)?.length).toBeGreaterThanOrEqual(3)
    expect(atlas).toMatch(/<button[^>]*type="button"[^>]*class="comparison-swap"/)
    expect(atlas).toMatch(/:aria-pressed="activePresetId === preset\.id"/)
    expect(atlas).toMatch(/:disabled="event\.id === comparisonRightId"/)
    expect(atlas).toMatch(/:disabled="event\.id === comparisonLeftId"/)
    expect(atlas.match(/aria-live="polite"/g)).toHaveLength(2)
    expect(atlas).toMatch(/class="event-selection-summary"[\s\S]{0,160}aria-live="polite"/)
    expect(atlas).toMatch(/class="comparison-live-summary" aria-live="polite"/)
    expect(atlas).not.toMatch(/class="comparison-compatibility"[\s\S]{0,180}role="status"/)
  })
})

describe('route-change focus contract', () => {
  it('moves focus to a programmatically focusable page heading after navigation', async () => {
    const app = await readFile(new URL('../src/App.vue', import.meta.url), 'utf8')
    const viewPaths = [
      '../src/views/OverviewView.vue',
      '../src/views/AtlasView.vue',
      '../src/views/FindingsView.vue',
      '../src/views/MethodsView.vue',
      '../src/views/CreditsView.vue',
    ]

    expect(app).toMatch(/querySelector\(['"]\[data-route-focus\]['"]\)/)
    expect(app).toMatch(/heading\.classList\.add\(['"]focus-target--route['"]\)/)
    expect(app).toMatch(/heading\.focus\(\{ preventScroll: true \}\)/)
    for (const viewPath of viewPaths) {
      const view = await readFile(new URL(viewPath, import.meta.url), 'utf8')
      expect(view).toMatch(/<h1[^>]*class="focus-target"[^>]*data-route-focus[^>]*tabindex="-1">/)
    }
  })

  it('does not suppress the shared focus indicator on route targets or atlas filters', async () => {
    const styles = await readFile(new URL('../src/styles/main.css', import.meta.url), 'utf8')

    expect(styles).not.toMatch(/main:focus\s*{\s*outline:\s*none/)
    expect(styles).not.toMatch(/\.atlas-controls\s+(?:input|select)[\s\S]{0,240}outline:\s*none/)
  })

  it('does not mistake the first route change for the first render', async () => {
    const app = await readFile(new URL('../src/App.vue', import.meta.url), 'utf8')

    expect(app).toMatch(/let enteredRoutePath = route\.path/)
    expect(app).toMatch(/if \(route\.path !== enteredRoutePath\)\s*{[\s\S]*?enteredRoutePath = route\.path[\s\S]*?focusRouteHeading\(\)/)
    expect(app).not.toMatch(/hasRenderedRoute/)
  })

  it('syncs document and navigation context after asynchronous route content enters', async () => {
    const app = await readFile(new URL('../src/App.vue', import.meta.url), 'utf8')

    expect(app).toMatch(/function handleRouteEnter\(\)[\s\S]*?updateRouteContext\(\)/)
    expect(app).toMatch(/@after-enter="handleRouteEnter"/)
  })

  it('preserves the Skip-link-first focus origin without scrolling a focusable route link into view', async () => {
    const app = await readFile(new URL('../src/App.vue', import.meta.url), 'utf8')
    const navigationReveal = app.match(/function revealActiveNavigation\(\) \{([\s\S]*?)\n\}/)?.[1]

    expect(app).toMatch(/<a class="skip-link" href="#main-content" @click\.prevent="focusMainContent">/)
    expect(app).toMatch(/navigation\.scrollLeft =/)
    expect(navigationReveal).toBeDefined()
    expect(navigationReveal).not.toMatch(/scrollIntoView/)
  })
})

describe('accessible shell navigation contract', () => {
  it('keeps the hash route stable while Skip explicitly focuses and scrolls main content', async () => {
    const app = await readFile(new URL('../src/App.vue', import.meta.url), 'utf8')
    const skipHandler = app.match(/function focusMainContent\(\) \{([\s\S]*?)\n\}/)?.[1]
    const routeFocusHandler = app.match(/function focusRouteHeading\(\) \{([\s\S]*?)\n\}/)?.[1]

    expect(skipHandler).toBeDefined()
    expect(skipHandler).toMatch(/main\.focus\(\{ preventScroll: true \}\)/)
    expect(skipHandler).toMatch(/main\.scrollIntoView\(\{ block: 'start' \}\)/)
    expect(skipHandler).not.toMatch(/(?:location|history|route|router)/)
    expect(routeFocusHandler).toBeDefined()
    expect(routeFocusHandler).toMatch(/focus\(\{ preventScroll: true \}\)/)
    expect(routeFocusHandler).not.toMatch(/scrollIntoView|scrollTo/)
  })

  it('exposes a controllable mobile menu that closes on Escape and route changes', async () => {
    const app = await readFile(new URL('../src/App.vue', import.meta.url), 'utf8')

    expect(app).toMatch(/class="mobile-nav-toggle"/)
    expect(app).toMatch(/:aria-expanded="isMobileNavigationOpen \? 'true' : 'false'"/)
    expect(app).toMatch(/aria-controls="primary-navigation"/)
    expect(app).toMatch(/id="primary-navigation"/)
    expect(app).toMatch(/:class="\{ 'site-nav--open': isMobileNavigationOpen \}"/)
    expect(app).toMatch(/@keydown\.esc="handleNavigationEscape"/)
    expect(app).toMatch(/watch\(\(\) => route\.fullPath, \(\) => closeMobileNavigation\(\)/)
    expect(app).toMatch(/:aria-current="route\.path === item\.to \? 'page' : undefined"/)
  })
})

describe('five-route task and interpretation contract', () => {
  it('gives Overview direct Atlas and Findings decisions while keeping proxy evidence separate from human validation', async () => {
    const overview = await readFile(new URL('../src/views/OverviewView.vue', import.meta.url), 'utf8')

    expect(overview).toMatch(/to="\/atlas"[^>]*>Open the study atlas/)
    expect(overview).toMatch(/to="\/findings"[^>]*>Read the bounded findings/)
    expect(overview).toMatch(/not human validation/i)
    expect(overview).toMatch(/descriptive R², unitless \[0–1\][^<]*not future-event accuracy/i)
  })

  it('makes Atlas modes, evidence-state definitions, and controlled regions discoverable without a score', async () => {
    const atlas = await readFile(new URL('../src/views/AtlasView.vue', import.meta.url), 'utf8')

    expect(atlas).toMatch(/id="atlas-mode-explore"[^>]*aria-controls="atlas-explore-panel"/)
    expect(atlas).toMatch(/id="atlas-mode-compare"[^>]*aria-controls="atlas-compare-panel"/)
    expect(atlas).toMatch(/id="atlas-view-help"[^>]*class="definition-disclosure"/)
    expect(atlas).toMatch(/Not assessed[\s\S]{0,80}means no reviewed public Passport/i)
    expect(atlas).toMatch(/Neither state means zero or worse recovery/i)
    expect(atlas).toMatch(/computes no new total, average, event rank/i)
    expect(atlas).not.toMatch(/recovery.?score|leaderboard/i)
  })

  it('states metric units, ranges, references, and outcome boundaries on Findings', async () => {
    const findings = await readFile(new URL('../src/views/FindingsView.vue', import.meta.url), 'utf8')

    expect(findings).toMatch(/R²[\s\S]{0,80}unitless \[0–1\]/i)
    expect(findings).toMatch(/AUC[\s\S]{0,80}unitless \[0–1\]/i)
    expect(findings).toMatch(/0\.50 is the no-ranking reference/i)
    expect(findings).toMatch(/Analysis admission\/readiness[\s\S]{0,160}recovery outcome/i)
    expect(findings).toMatch(/metric\.unit/)
    expect(findings).not.toMatch(/readiness[^\n]{0,80}recovery performance/i)
  })

  it('traces Methods from private inputs through admission to the local public artifact', async () => {
    const methods = await readFile(new URL('../src/views/MethodsView.vue', import.meta.url), 'utf8')

    expect(methods.match(/class="method-timeline__number">0[1-5]</g)).toHaveLength(5)
    expect(methods).toMatch(/Input boundary/)
    expect(methods).toMatch(/Analysis admission/)
    expect(methods).toMatch(/Public artifact/)
    expect(methods).toMatch(/Missing assessment remains Not assessed; unavailable components remain unavailable rather than becoming zero/)
    expect(methods).toMatch(/local artifacts only and makes no runtime data request/)
  })

  it('makes rights, aggregate grain, local runtime, analytics absence, and known limits scannable on Credits', async () => {
    const credits = await readFile(new URL('../src/views/CreditsView.vue', import.meta.url), 'utf8')

    expect(credits).toMatch(/Aggregate-only public content/)
    expect(credits).toMatch(/Local assets, no analytics/)
    expect(credits).toMatch(/does not request them in the background/)
    expect(credits).toMatch(/KNOWN LIMITS/)
    expect(credits).toMatch(/screen-reader, speech-input, switch-access, and multi-browser support/)
  })
})

describe('shared UI primitive and high-contrast contract', () => {
  it('uses each shared primitive in multiple real route consumers', async () => {
    const viewPaths = [
      '../src/views/OverviewView.vue',
      '../src/views/AtlasView.vue',
      '../src/views/FindingsView.vue',
      '../src/views/MethodsView.vue',
      '../src/views/CreditsView.vue',
    ]
    const views = (await Promise.all(viewPaths.map((viewPath) => readFile(new URL(viewPath, import.meta.url), 'utf8')))).join('\n')

    expect(views.match(/definition-disclosure/g)?.length).toBeGreaterThanOrEqual(2)
    expect(views.match(/data-table-wrap/g)?.length).toBeGreaterThanOrEqual(2)
    expect(views.match(/status-badge/g)?.length).toBeGreaterThanOrEqual(2)
    expect(views.match(/state-panel/g)?.length).toBeGreaterThanOrEqual(2)
    expect(views.match(/focus-target/g)).toHaveLength(5)
  })

  it('provides contained data tables and explicit Windows forced-colors mappings', async () => {
    const styles = await readFile(new URL('../src/styles/main.css', import.meta.url), 'utf8')

    expect(styles).toMatch(/\.data-table-wrap\s*{[^}]*overflow-x:\s*auto/s)
    expect(styles).toMatch(/@media \(forced-colors: active\)/)
    expect(styles).toMatch(/outline-color:\s*Highlight/)
    expect(styles).toMatch(/border-color:\s*CanvasText/)
    expect(styles).not.toMatch(/@media \(forced-colors: active\)[\s\S]*?\*\s*{[^}]*forced-color-adjust:\s*none/)
  })

  it('does not force a 320px root width after classic scrollbars reduce the layout viewport', async () => {
    const styles = await readFile(new URL('../src/styles/main.css', import.meta.url), 'utf8')

    expect(styles).not.toMatch(/html\s*{[^}]*min-width:\s*320px/s)
    expect(styles).not.toMatch(/body\s*{[^}]*min-width:\s*320px/s)
  })

  it('allows shared grid children and long evidence labels to shrink under user text spacing', async () => {
    const styles = await readFile(new URL('../src/styles/main.css', import.meta.url), 'utf8')

    expect(styles).toMatch(/\.page-heading--split > \*,[\s\S]{0,240}\.license-ledger > \*\s*{\s*min-width:\s*0/)
    expect(styles).toMatch(/\.finding-hero__number small[\s\S]{0,240}overflow-wrap:\s*anywhere/)
    expect(styles).toMatch(/\.license-ledger strong\s*{\s*overflow-wrap:\s*anywhere/)
  })
})

describe('dependency install policy', () => {
  it('approves only the pinned esbuild install script required by Vite', async () => {
    const packageJson = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'))
    const packageLock = JSON.parse(await readFile(new URL('../package-lock.json', import.meta.url), 'utf8'))

    expect(packageLock.packages['node_modules/esbuild'].version).toBe('0.25.12')
    expect(packageJson.allowScripts).toEqual({ 'esbuild@0.25.12': true })
  })
})

describe('Vercel upload boundary', () => {
  it('ignores only root entries before unignoring approved source directories', async () => {
    let contents
    try {
      contents = await readFile(new URL('../.vercelignore', import.meta.url), 'utf8')
    } catch (error) {
      expect(error.code).toBe('ENOENT')
      expect(process.env.VERCEL).toBe('1')
      return
    }

    const rules = contents.split(/\r?\n/).filter(Boolean)

    expect(rules[0]).toBe('/*')
    expect(rules).toEqual(expect.arrayContaining(['!src', '!public', '!scripts', '!tests']))
  })
})
