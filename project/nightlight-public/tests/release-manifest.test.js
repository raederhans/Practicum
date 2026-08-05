import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

import { createReleaseManifest, verifyReleaseManifest } from '../scripts/release-manifest.mjs'

import { makeTemporaryRoot } from './support/temporaryRoot.js'

describe('immutable release manifest', () => {
  it('records every dist file except itself with stable paths, bytes, and SHA-256', async () => {
    const root = await makeTemporaryRoot()
    const dist = join(root, 'dist')
    await mkdir(join(dist, 'assets'), { recursive: true })
    await writeFile(join(dist, 'index.html'), '<main>observatory</main>')
    await writeFile(join(dist, 'assets', 'app.js'), 'export const ready = true')

    const manifest = await createReleaseManifest(dist)
    const persisted = JSON.parse(await readFile(join(dist, 'release-manifest.json'), 'utf8'))

    expect(manifest).toEqual(persisted)
    expect(manifest.schemaVersion).toBe(1)
    expect(manifest.files.map((file) => file.path)).toEqual(['assets/app.js', 'index.html'])
    expect(manifest.files.every((file) => Number.isInteger(file.bytes) && file.bytes > 0)).toBe(true)
    expect(manifest.files.every((file) => /^[a-f0-9]{64}$/.test(file.sha256))).toBe(true)
    await expect(verifyReleaseManifest(dist)).resolves.toEqual({ ok: true, violations: [] })
  })

  it('detects modified, missing, and unlisted files', async () => {
    const root = await makeTemporaryRoot()
    const dist = join(root, 'dist')
    await mkdir(dist, { recursive: true })
    await writeFile(join(dist, 'index.html'), '<main>observatory</main>')
    await createReleaseManifest(dist)
    await writeFile(join(dist, 'index.html'), '<main>changed</main>')
    await writeFile(join(dist, 'extra.js'), 'unexpected')

    const result = await verifyReleaseManifest(dist)

    expect(result.ok).toBe(false)
    expect(result.violations.join('\n')).toMatch(/index\.html.*mismatch/i)
    expect(result.violations.join('\n')).toMatch(/extra\.js.*unlisted/i)
  })
})
