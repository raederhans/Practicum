import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

import {
  STATIC_RELEASE_CONTRACT,
  createReleaseManifest,
  verifyReleaseManifest,
} from '../scripts/release-manifest.mjs'

import { makeTemporaryRoot } from './support/temporaryRoot.js'

describe('immutable release manifest', () => {
  it('records every dist file except itself with stable paths, bytes, and SHA-256', async () => {
    const root = await makeTemporaryRoot()
    const dist = join(root, 'dist')
    await mkdir(join(dist, 'assets'), { recursive: true })
    await writeFile(join(dist, 'index.html'), '<script src="/assets/app.js"></script>')
    await writeFile(join(dist, 'assets', 'app.js'), 'export const ready = true')

    const manifest = await createReleaseManifest(dist)
    const persisted = JSON.parse(await readFile(join(dist, 'release-manifest.json'), 'utf8'))

    expect(manifest).toEqual(persisted)
    expect(manifest.schemaVersion).toBe(2)
    expect(manifest.buildContract).toEqual({ basePath: '/', ...STATIC_RELEASE_CONTRACT })
    expect(manifest.files.map((file) => file.path)).toEqual(['assets/app.js', 'index.html'])
    expect(manifest.files.every((file) => Number.isInteger(file.bytes) && file.bytes > 0)).toBe(true)
    expect(manifest.files.every((file) => /^[a-f0-9]{64}$/.test(file.sha256))).toBe(true)
    await expect(verifyReleaseManifest(dist)).resolves.toEqual({ ok: true, violations: [] })
  })

  it('detects modified, missing, and unlisted files', async () => {
    const root = await makeTemporaryRoot()
    const dist = join(root, 'dist')
    await mkdir(dist, { recursive: true })
    await writeFile(join(dist, 'index.html'), '<script src="/assets/app.js"></script>')
    await createReleaseManifest(dist)
    await writeFile(join(dist, 'index.html'), '<main>changed</main>')
    await writeFile(join(dist, 'extra.js'), 'unexpected')

    const result = await verifyReleaseManifest(dist)

    expect(result.ok).toBe(false)
    expect(result.violations.join('\n')).toMatch(/index\.html.*mismatch/i)
    expect(result.violations.join('\n')).toMatch(/extra\.js.*unlisted/i)
  })

  it('binds a repository base path to the built HTML resource paths', async () => {
    const root = await makeTemporaryRoot()
    const dist = join(root, 'dist')
    await mkdir(join(dist, 'assets'), { recursive: true })
    await writeFile(join(dist, 'index.html'), '<script src="/Practicum/assets/app.js"></script>')
    await writeFile(join(dist, 'assets', 'app.js'), 'export const ready = true')

    await createReleaseManifest(dist, { basePath: '/Practicum/' })
    await expect(verifyReleaseManifest(dist)).resolves.toEqual({ ok: true, violations: [] })

    const manifestPath = join(dist, 'release-manifest.json')
    const manifest = JSON.parse(await readFile(manifestPath, 'utf8'))
    manifest.buildContract.basePath = '/'
    await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`)

    const result = await verifyReleaseManifest(dist)
    expect(result.ok).toBe(false)
    expect(result.violations.join('\n')).toMatch(/resource.*base path/i)
  })

  it('rejects duplicate, traversing, malformed, and undeclared manifest metadata', async () => {
    const root = await makeTemporaryRoot()
    const dist = join(root, 'dist')
    await mkdir(dist, { recursive: true })
    await writeFile(join(dist, 'index.html'), '<main>observatory</main>')
    await createReleaseManifest(dist)

    const manifestPath = join(dist, 'release-manifest.json')
    const manifest = JSON.parse(await readFile(manifestPath, 'utf8'))
    manifest.files.push({ ...manifest.files[0] })
    manifest.files.push({ path: '../private.txt', bytes: 1, sha256: '0'.repeat(64) })
    manifest.files[0].unexpected = true
    await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`)

    const result = await verifyReleaseManifest(dist)
    expect(result.ok).toBe(false)
    expect(result.violations.join('\n')).toMatch(/duplicate manifest entry/i)
    expect(result.violations.join('\n')).toMatch(/invalid file path/i)
    expect(result.violations.join('\n')).toMatch(/unexpected or missing fields/i)
  })
})
