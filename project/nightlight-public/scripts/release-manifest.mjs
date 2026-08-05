import { createHash } from 'node:crypto'
import { readdir, readFile, stat, writeFile } from 'node:fs/promises'
import { relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const manifestName = 'release-manifest.json'

function normalizePath(path) {
  return path.split(sep).join('/')
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
  return files.sort((left, right) => left.path.localeCompare(right.path, 'en'))
}

export async function createReleaseManifest(distPath) {
  const dist = resolve(distPath)
  const manifest = { schemaVersion: 1, files: await collectReleaseFiles(dist) }
  await writeFile(resolve(dist, manifestName), `${JSON.stringify(manifest, null, 2)}\n`)
  return manifest
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

  if (manifest.schemaVersion !== 1 || !Array.isArray(manifest.files)) {
    return { ok: false, violations: [`${manifestName}: unsupported schema`] }
  }

  const actualFiles = await collectReleaseFiles(dist)
  const declared = new Map(manifest.files.map((file) => [file.path, file]))
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
  const sortedPaths = [...declaredPaths].sort((left, right) => left.localeCompare(right, 'en'))
  if (JSON.stringify(declaredPaths) !== JSON.stringify(sortedPaths)) {
    violations.push(`${manifestName}: entries are not stably sorted`)
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
