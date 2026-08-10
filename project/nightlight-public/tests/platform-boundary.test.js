import { mkdir, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import { scanPublicTree } from '../scripts/verify-public.mjs'

import { makeTemporaryRoot } from './support/temporaryRoot.js'

async function writeFixture(root, relativePath, contents) {
  const path = join(root, relativePath)
  await mkdir(join(path, '..'), { recursive: true })
  await writeFile(path, contents)
}

describe('static platform boundary', () => {
  it('rejects runtime dependencies outside the reviewed static application allowlist', async () => {
    const root = await makeTemporaryRoot()
    await writeFixture(root, 'package.json', JSON.stringify({
      dependencies: { vue: '3.5.21', 'vue-router': '4.5.1', analytics: '1.0.0' },
      devDependencies: { '@vitejs/plugin-vue': '6.0.1', vite: '6.4.3', vitest: '3.2.7' },
      engines: { node: '>=20' },
    }))

    const result = await scanPublicTree(root)
    expect(result.ok).toBe(false)
    expect(result.violations.join('\n')).toMatch(/dependencies dependency analytics is not allowlisted/i)
  })

  it('rejects weakened CSP metadata and keeps Pages claims limited to enforceable HTML policy', async () => {
    const root = await makeTemporaryRoot()
    await writeFixture(root, 'index.html', [
      '<meta http-equiv="Content-Security-Policy" content="default-src \'self\'; script-src \'self\' \'unsafe-inline\'; style-src \'self\'; img-src \'self\'; font-src \'self\'; connect-src https:; object-src \'none\'; base-uri \'self\'; form-action \'none\'">',
      '<meta name="referrer" content="origin">',
    ].join(''))

    const result = await scanPublicTree(root)
    expect(result.ok).toBe(false)
    expect(result.violations.join('\n')).toMatch(/connect-src|unsafe|referrer/i)
  })

  it('requires safe rel attributes on external links that open a new tab', async () => {
    const root = await makeTemporaryRoot()
    await writeFixture(root, 'src/views/CreditsView.vue', '<template><a href="https://example.com" target="_blank" rel="noopener">Source</a></template>')

    const result = await scanPublicTree(root)
    expect(result.ok).toBe(false)
    expect(result.violations.join('\n')).toMatch(/noopener noreferrer/i)
  })

  it('requires lockfile and direct dependency notices whenever package metadata is present', async () => {
    const root = await makeTemporaryRoot()
    await writeFixture(root, 'package.json', JSON.stringify({
      dependencies: { vue: '3.5.21', 'vue-router': '4.5.1' },
      devDependencies: { '@vitejs/plugin-vue': '6.0.1', vite: '6.4.3', vitest: '3.2.7' },
      engines: { node: '>=20' },
      allowScripts: { 'esbuild@0.25.12': true },
    }))

    const result = await scanPublicTree(root)
    expect(result.ok).toBe(false)
    expect(result.violations.join('\n')).toMatch(/package-lock\.json.*required/i)
  })
})
