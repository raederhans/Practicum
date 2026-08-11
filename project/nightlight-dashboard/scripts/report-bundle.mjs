import { readFile } from 'node:fs/promises'
import { gzipSync } from 'node:zlib'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const ROUTE_SOURCES = {
  home: 'src/views/HomeView.vue',
  map: 'src/views/MapView.vue',
}

/**
 * @typedef {{ file?: string, name?: string, isEntry?: boolean, css?: string[], imports?: string[] }} ManifestItem
 * @typedef {Record<string, ManifestItem>} ViteManifest
 */

/** @param {ViteManifest} manifest @param {string} key */
function requireManifestItem(manifest, key) {
  const item = manifest[key]
  if (!item) throw new Error(`Bundle manifest is missing ${key}`)
  return item
}

/** @param {ViteManifest} manifest @param {string[]} roots */
export function collectManifestFiles(manifest, roots) {
  const visitedKeys = new Set()
  const files = new Set()

  /** @param {string} key */
  const visit = key => {
    if (visitedKeys.has(key)) return
    visitedKeys.add(key)

    const item = requireManifestItem(manifest, key)
    if (item.file) files.add(item.file)
    for (const cssFile of item.css ?? []) files.add(cssFile)
    for (const importedKey of item.imports ?? []) visit(importedKey)
  }

  for (const root of roots) visit(root)
  return files
}

/** @param {string} indexHtml */
export function assertNoExternalMapLibreCss(indexHtml) {
  if (/<link\b[^>]*href=["']https?:\/\/[^"']*maplibre[^"']*\.css[^"']*["'][^>]*>/i.test(indexHtml)) {
    throw new Error('External MapLibre CSS leaked into the application shell')
  }
}

/** @param {ViteManifest} manifest */
export function resolveRouteBoundaries(manifest) {
  const entryKey = Object.keys(manifest).find(key => manifest[key].isEntry)
  if (!entryKey) throw new Error('Bundle manifest has no application entry')

  const maplibreEntries = Object.entries(manifest)
    .filter(([, item]) => item.name === 'maplibre')
  if (maplibreEntries.length !== 1) {
    throw new Error(`Expected one isolated MapLibre chunk, found ${maplibreEntries.length}`)
  }

  const [[, maplibreItem]] = maplibreEntries
  if (!maplibreItem.file) {
    throw new Error('The isolated MapLibre chunk has no emitted file')
  }
  const homeFiles = collectManifestFiles(manifest, [entryKey, ROUTE_SOURCES.home])
  const mapFiles = collectManifestFiles(manifest, [entryKey, ROUTE_SOURCES.map])

  if (homeFiles.has(maplibreItem.file)) {
    throw new Error('MapLibre leaked into the initial home-route payload')
  }
  if (!mapFiles.has(maplibreItem.file)) {
    throw new Error('The map route no longer loads the isolated MapLibre chunk')
  }

  return {
    entryKey,
    maplibreFile: maplibreItem.file,
    homeFiles,
    mapFiles,
    mapIncrementalFiles: new Set([...mapFiles].filter(file => !homeFiles.has(file))),
  }
}

/** @param {string} distDirectory @param {Set<string>} files */
async function measureFiles(distDirectory, files) {
  const details = await Promise.all([...files].sort().map(async file => {
    const contents = await readFile(resolve(distDirectory, file))
    return {
      file,
      rawBytes: contents.byteLength,
      gzipBytes: gzipSync(contents).byteLength,
    }
  }))

  return {
    rawBytes: details.reduce((total, file) => total + file.rawBytes, 0),
    gzipBytes: details.reduce((total, file) => total + file.gzipBytes, 0),
    files: details,
  }
}

export async function createBundleReport(distDirectory = 'dist') {
  const absoluteDistDirectory = resolve(distDirectory)
  const indexHtml = await readFile(resolve(absoluteDistDirectory, 'index.html'), 'utf8')
  assertNoExternalMapLibreCss(indexHtml)
  const manifest = JSON.parse(await readFile(
    resolve(absoluteDistDirectory, '.vite/manifest.json'),
    'utf8',
  ))
  const boundaries = resolveRouteBoundaries(manifest)
  const homeInitialFiles = new Set(['index.html', ...boundaries.homeFiles])
  const mapDirectFiles = new Set(['index.html', ...boundaries.mapFiles])

  const [homeInitial, mapDirect, mapIncrementalAfterHome, maplibre] = await Promise.all([
    measureFiles(absoluteDistDirectory, homeInitialFiles),
    measureFiles(absoluteDistDirectory, mapDirectFiles),
    measureFiles(absoluteDistDirectory, boundaries.mapIncrementalFiles),
    measureFiles(absoluteDistDirectory, new Set([boundaries.maplibreFile])),
  ])

  return {
    schemaVersion: 1,
    routes: {
      homeInitial,
      mapDirect,
      mapIncrementalAfterHome,
    },
    maplibre: {
      ...maplibre,
      isolatedFromHomeInitial: true,
      exceeds500kMinified: maplibre.rawBytes > 500_000,
    },
  }
}

async function main() {
  const report = await createBundleReport(process.argv[2] ?? 'dist')
  console.log(JSON.stringify(report, null, 2))
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(error => {
    console.error(error instanceof Error ? error.message : error)
    process.exitCode = 1
  })
}
