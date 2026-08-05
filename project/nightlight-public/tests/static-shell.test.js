import { readFile } from 'node:fs/promises'
import { describe, expect, it } from 'vitest'

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
})

describe('atlas keyboard contract', () => {
  it('keeps SVG points presentational and exposes selection on native buttons', async () => {
    const atlas = await readFile(new URL('../src/views/AtlasView.vue', import.meta.url), 'utf8')

    expect(atlas).not.toMatch(/class="atlas-point"[\s\S]*?tabindex="0"/)
    expect(atlas).not.toMatch(/class="atlas-point"[\s\S]*?role="button"/)
    expect(atlas).toMatch(/:aria-pressed="selectedEvent\?\.id === event\.id"/)
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
