import { createHash } from 'node:crypto'
import { readdir, readFile, stat, writeFile } from 'node:fs/promises'
import { relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const manifestName = 'release-manifest.json'

export const STATIC_RELEASE_CONTRACT = Object.freeze({
  deploymentModel: 'static-only',
  runtimeData: 'local-assets-only',
  dataGrain: 'aggregate-only',
  analytics: 'local-opt-in-only',
  analyticsTransport: 'none',
  persistentIdentifier: false,
  externalRequests: false,
  sourceMaps: false,
})

function normalizePath(path) {
  return path.split(sep).join('/')
}

function stableCompare(left, right) {
  if (left === right) return 0
  return left < right ? -1 : 1
}

function normalizeBasePath(basePath) {
  if (typeof basePath !== 'string' || !basePath.startsWith('/') || !basePath.endsWith('/')) {
    throw new TypeError('release base path must start and end with a slash')
  }
  if (basePath.includes('\\') || basePath.includes('//') || basePath.split('/').includes('..')) {
    throw new TypeError('release base path must be a normalized absolute path')
  }
  return basePath
}

function expectedBuildContract(basePath) {
  return { basePath: normalizeBasePath(basePath), ...STATIC_RELEASE_CONTRACT }
}

async function collectReleaseFiles(distPath) {
  const dist = resolve(distPath)
  const files = []

  async function walk(directory) {
    const entries = await readdir(directory, { withFileTypes: true })
    for (const entry of entries) {
      const absolutePath = resolve(directory, entry.name)
      const relativePath = normalizePath(relative(dist, absolutePath))
      if (relativePath === manifestName) continue
      if (entry.isDirectory()) {
        await walk(absolutePath)
        continue
      }
      if (!entry.isFile()) continue

      const contents = await readFile(absolutePath)
      files.push({
        path: relativePath,
        bytes: contents.byteLength,
        sha256: createHash('sha256').update(contents).digest('hex'),
      })
    }
  }

  await walk(dist)
  return files.sort((left, right) => stableCompare(left.path, right.path))
}

export async function createReleaseManifest(
  distPath,
  { basePath = process.env.VITE_BASE_PATH || '/' } = {},
) {
  const dist = resolve(distPath)
  const manifest = {
    schemaVersion: 2,
    buildContract: expectedBuildContract(basePath),
    files: await collectReleaseFiles(dist),
  }
  await writeFile(resolve(dist, manifestName), `${JSON.stringify(manifest, null, 2)}\n`)
  return manifest
}

function hasExactKeys(value, expectedKeys) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const actualKeys = Object.keys(value).sort(stableCompare)
  const sortedExpected = [...expectedKeys].sort(stableCompare)
  return JSON.stringify(actualKeys) === JSON.stringify(sortedExpected)
}

async function verifyBuiltBasePath(dist, basePath) {
  const violations = []
  let html
  try {
    html = await readFile(resolve(dist, 'index.html'), 'utf8')
  } catch (error) {
    return [`index.html: missing or unreadable (${error.code ?? error.name})`]
  }

  const localResourcePattern = /\b(?:href|src)=["'](\/(?!\/)[^"']*)["']/g
  for (const match of html.matchAll(localResourcePattern)) {
    const assetMarker = match[1].indexOf('assets/')
    const resourceBasePath = assetMarker >= 0 ? match[1].slice(0, assetMarker) : null
    if (!match[1].startsWith(basePath) || (resourceBasePath !== null && resourceBasePath !== basePath)) {
      violations.push(`index.html: local resource ${match[1]} does not match manifest base path ${basePath}`)
    }
  }
  return violations
}

export async function verifyReleaseManifest(distPath) {
  const dist = resolve(distPath)
  const violations = []
  let manifest

  try {
    manifest = JSON.parse(await readFile(resolve(dist, manifestName), 'utf8'))
  } catch (error) {
    return { ok: false, violations: [`${manifestName}: missing or invalid (${error.code ?? error.name})`] }
  }

  if (!hasExactKeys(manifest, ['schemaVersion', 'buildContract', 'files'])) {
    return { ok: false, violations: [`${manifestName}: unexpected or missing top-level fields`] }
  }
  if (manifest.schemaVersion !== 2 || !Array.isArray(manifest.files)) {
    return { ok: false, violations: [`${manifestName}: unsupported schema`] }
  }

  let expectedContract
  try {
    expectedContract = expectedBuildContract(manifest.buildContract?.basePath)
  } catch (error) {
    violations.push(`${manifestName}: ${error.message}`)
  }
  if (
    !hasExactKeys(manifest.buildContract, ['basePath', ...Object.keys(STATIC_RELEASE_CONTRACT)])
    || (expectedContract && JSON.stringify(manifest.buildContract) !== JSON.stringify(expectedContract))
  ) {
    violations.push(`${manifestName}: build contract does not match the static release contract`)
  }

  const actualFiles = await collectReleaseFiles(dist)
  const declared = new Map()
  const seenDeclaredPaths = new Set()
  for (const file of manifest.files) {
    if (typeof file?.path === 'string') {
      if (seenDeclaredPaths.has(file.path)) violations.push(`${file.path}: duplicate manifest entry`)
      seenDeclaredPaths.add(file.path)
    }
    if (!hasExactKeys(file, ['path', 'bytes', 'sha256'])) {
      violations.push(`${manifestName}: file entry has unexpected or missing fields`)
      continue
    }
    if (
      typeof file.path !== 'string'
      || file.path === ''
      || file.path.startsWith('/')
      || file.path.includes('\\')
      || file.path.split('/').includes('..')
    ) {
      violations.push(`${manifestName}: invalid file path ${String(file.path)}`)
      continue
    }
    if (!Number.isInteger(file.bytes) || file.bytes < 0 || !/^[a-f0-9]{64}$/.test(file.sha256)) {
      violations.push(`${file.path}: invalid bytes or SHA-256 declaration`)
    }
    declared.set(file.path, file)
  }
  const actual = new Map(actualFiles.map((file) => [file.path, file]))

  for (const [path, file] of actual) {
    const expected = declared.get(path)
    if (!expected) {
      violations.push(`${path}: unlisted file`)
      continue
    }
    if (file.bytes !== expected.bytes || file.sha256 !== expected.sha256) {
      violations.push(`${path}: bytes or SHA-256 mismatch`)
    }
  }

  for (const path of declared.keys()) {
    if (!actual.has(path)) violations.push(`${path}: declared file is missing`)
  }

  const declaredPaths = manifest.files.map((file) => file.path)
  const sortedPaths = [...declaredPaths].sort(stableCompare)
  if (JSON.stringify(declaredPaths) !== JSON.stringify(sortedPaths)) {
    violations.push(`${manifestName}: entries are not stably sorted`)
  }

  if (expectedContract) {
    violations.push(...await verifyBuiltBasePath(dist, expectedContract.basePath))
  }

  return { ok: violations.length === 0, violations: [...new Set(violations)].sort() }
}

async function main() {
  if (!process.argv.includes('--write')) return
  const dist = resolve(process.cwd(), 'dist')
  await stat(dist)
  const manifest = await createReleaseManifest(dist)
  console.log(`Release manifest written for ${manifest.files.length} files.`)
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
  await main()
}
