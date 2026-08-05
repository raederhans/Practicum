import { lstat, readdir, readFile } from 'node:fs/promises'
import { basename, extname, relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

import { verifyReleaseManifest } from './release-manifest.mjs'
import { PUBLIC_GENERALIZATION_ARTIFACT, validatePublicGeneralizationArtifact } from '../src/content/generalizationArtifact.js'

const ignoredRootDirectories = new Set(['node_modules', '.git', '.vercel'])
const allowedTopLevel = new Set([
  '.github',
  'src',
  'tests',
  'scripts',
  'public',
  'dist',
  'index.html',
  'package.json',
  'package-lock.json',
  'vite.config.js',
  'vercel.json',
  'LICENSE',
  'README.md',
  'CREDITS.md',
  'DATA_POLICY.md',
  'THIRD_PARTY_NOTICES.md',
  '.gitignore',
  '.vercelignore',
])
const prohibitedExtensions = new Set([
  '.csv', '.tsv', '.tif', '.tiff', '.geojson', '.parquet', '.feather',
  '.pkl', '.pickle', '.joblib', '.onnx', '.pt', '.pth', '.h5', '.hdf5',
  '.sav', '.rds', '.db', '.sqlite', '.sqlite3', '.zip', '.7z', '.tar', '.gz',
  '.xls', '.xlsx', '.doc', '.docx', '.ipynb', '.map', '.pem', '.key', '.p12',
  '.pfx', '.exe', '.dll', '.bat', '.cmd', '.ps1', '.sh', '.wasm',
])
const prohibitedNames = new Set(['events_config.js', 'results_summary.json'])
const prohibitedPrefixes = ['facilities_', 'prob_', 'ts_']
const prohibitedExactNames = new Set(['.env', '.npmrc', '.pypirc', '.netrc', '.ds_store'])
const allowedExtensions = new Set(['', '.js', '.mjs', '.vue', '.css', '.html', '.json', '.md', '.yml', '.yaml', '.txt', '.svg'])
const textExtensions = new Set(['.js', '.mjs', '.vue', '.css', '.html', '.json', '.md', '.yml', '.yaml', '.txt', '.svg'])
const allowedExactFiles = new Set([
  '.github/workflows/deploy-pages.yml',
  '.gitignore',
  '.vercelignore',
  'CREDITS.md',
  'DATA_POLICY.md',
  'LICENSE',
  'README.md',
  'THIRD_PARTY_NOTICES.md',
  'index.html',
  'package-lock.json',
  'package.json',
  'public/observatory-mark.svg',
  'scripts/release-manifest.mjs',
  'scripts/verify-public.mjs',
  'src/App.vue',
  'src/content/copy.js',
  'src/content/generalizationArtifact.js',
  'src/content/study.js',
  'src/domain/filterEvents.js',
  'src/domain/projectPoint.js',
  'src/domain/resolveSelectedId.js',
  'src/main.js',
  'src/router/routes.js',
  'src/styles/main.css',
  'src/views/AtlasView.vue',
  'src/views/CreditsView.vue',
  'src/views/FindingsView.vue',
  'src/views/MethodsView.vue',
  'src/views/OverviewView.vue',
  'tests/copy.test.js',
  'tests/generalization-artifact.test.js',
  'tests/public-boundary.test.js',
  'tests/release-manifest.test.js',
  'tests/routes.test.js',
  'tests/static-shell.test.js',
  'tests/study.test.js',
  'tests/support/temporaryRoot.js',
  'vercel.json',
  'vite.config.js',
])
const allowedDistFiles = new Set([
  'dist/index.html',
  'dist/observatory-mark.svg',
  'dist/release-manifest.json',
])
const credentialPatterns = [
  /gh[pousr]_[A-Za-z0-9]{30,}/,
  /github_pat_[A-Za-z0-9_]{50,}/,
  /AKIA[0-9A-Z]{16}/,
  /AIza[0-9A-Za-z_-]{30,}/,
  /-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----/,
  /(?:api[_-]?key|secret|access[_-]?token)\s*[=:]\s*['"][^'"]{12,}['"]/i,
]
const runtimeNetworkPatterns = [
  /\bfetch\s*\(/,
  /\bXMLHttpRequest\b/,
  /\bWebSocket\s*\(/,
  /\bEventSource\s*\(/,
  /\baxios\s*[.(]/,
  /<(?:script|img|source|video|audio|iframe|embed)\b[^>]*\bsrc\s*=\s*['"]https?:\/\//i,
  /<(?:link|object)\b[^>]*\b(?:href|data)\s*=\s*['"]https?:\/\//i,
  /url\(\s*['"]?https?:\/\//i,
  /@import\s+(?:url\()?\s*['"]?https?:\/\//i,
]
const restrictedFieldPattern = /(?:^|[{,]\s*)['"]?(?:facility|facilities|probability|probabilities|time_series|timeseries|zip_code|zip_event|grid_id|predicted_value|prediction|outage_duration|recovery_time)['"]?\s*:/gim
const rawSchemaPatterns = [
  new RegExp(['eaglei', 'outages'].join('_'), 'i'),
  new RegExp(['Outage', 'Dataset', 'R1'].join('_'), 'i'),
  new RegExp(['customers', 'out'].join('_'), 'i'),
  new RegExp(['county', 'fips'].join('_'), 'i'),
  new RegExp(['state', 'fips'].join('_'), 'i'),
]
const localAnalysisPathPatterns = [
  /[A-Za-z]:[\\/](?:Users|home)[\\/]/i,
  /\/home\//i,
  /project[\\/]data[\\/]raw/i,
  /cache[\\/]experiments/i,
]

function normalizePath(path) {
  return path.split(sep).join('/')
}

function isRuntimeSurface(relativePath) {
  return relativePath === 'index.html'
    || relativePath === 'vite.config.js'
    || relativePath.startsWith('src/')
    || relativePath.startsWith('public/')
    || relativePath.startsWith('dist/')
}

function isAllowlistedFile(relativePath) {
  return allowedExactFiles.has(relativePath)
    || allowedDistFiles.has(relativePath)
    || /^dist\/assets\/[A-Za-z0-9_-]+\.(?:js|css)$/.test(relativePath)
}

export async function scanPublicTree(rootPath, { requireDist = false } = {}) {
  const root = resolve(rootPath)
  const violations = []
  let distIndexFound = false

  async function walk(directory) {
    const entries = await readdir(directory, { withFileTypes: true })
    for (const entry of entries) {
      const absolutePath = resolve(directory, entry.name)
      const relativePath = normalizePath(relative(root, absolutePath))
      const topLevel = relativePath.split('/')[0]

      const isRootDirectory = entry.isDirectory() && !relativePath.includes('/')
      if (isRootDirectory && ignoredRootDirectories.has(entry.name)) continue
      if (!allowedTopLevel.has(topLevel)) {
        violations.push(`${relativePath}: top-level path is not allowlisted`)
      }

      if (entry.isDirectory()) {
        if (ignoredRootDirectories.has(entry.name)) {
          violations.push(`${relativePath}: nested ignored directory name is not permitted`)
        }
        if (entry.name.toLocaleLowerCase() === 'data') {
          violations.push(`${relativePath}: data directories are not permitted`)
        }
        await walk(absolutePath)
        continue
      }

      const stats = await lstat(absolutePath)
      if (stats.isSymbolicLink()) {
        violations.push(`${relativePath}: symbolic links are not permitted`)
        continue
      }

      const lowerName = basename(relativePath).toLocaleLowerCase()
      const extension = extname(lowerName)
      if (!isAllowlistedFile(relativePath)) {
        violations.push(`${relativePath}: file path is not allowlisted`)
      }
      if (!allowedExtensions.has(extension)) {
        violations.push(`${relativePath}: file type ${extension || '(none)'} is not allowlisted`)
      }
      if (prohibitedExtensions.has(extension)) {
        violations.push(`${relativePath}: prohibited artifact type ${extension}`)
      }
      if (prohibitedNames.has(lowerName) || prohibitedPrefixes.some((prefix) => lowerName.startsWith(prefix))) {
        violations.push(`${relativePath}: prohibited artifact name`)
      }
      if (prohibitedExactNames.has(lowerName) || lowerName.startsWith('.env.')) {
        violations.push(`${relativePath}: prohibited private configuration name`)
      }
      if (stats.size > 1_500_000) {
        violations.push(`${relativePath}: file exceeds the 1.5 MB public limit`)
      }

      if (relativePath === 'dist/index.html') distIndexFound = true
      if (!textExtensions.has(extension) && !['LICENSE'].includes(relativePath)) continue

      const contents = await readFile(absolutePath, 'utf8')
      for (const pattern of credentialPatterns) {
        if (pattern.test(contents)) {
          violations.push(`${relativePath}: credential-shaped content detected`)
          break
        }
      }

      for (const pattern of rawSchemaPatterns) {
        if (pattern.test(contents)) {
          violations.push(`${relativePath}: raw schema marker detected`)
          break
        }
      }
      for (const pattern of localAnalysisPathPatterns) {
        if (pattern.test(contents)) {
          violations.push(`${relativePath}: local analysis path detected`)
          break
        }
      }

      if (restrictedFieldPattern.test(contents)) {
        violations.push(`${relativePath}: restricted field detected`)
      }
      restrictedFieldPattern.lastIndex = 0

      if (isRuntimeSurface(relativePath)) {
        for (const pattern of runtimeNetworkPatterns) {
          if (pattern.test(contents)) {
            violations.push(`${relativePath}: runtime network request detected`)
            break
          }
        }
      }
    }
  }

  await walk(root)
  const artifactViolations = validatePublicGeneralizationArtifact(PUBLIC_GENERALIZATION_ARTIFACT)
  violations.push(...artifactViolations.map((violation) => `src/content/generalizationArtifact.js: ${violation}`))
  if (requireDist && !distIndexFound) {
    violations.push('dist/index.html: required production build is missing')
  }
  if (requireDist && distIndexFound) {
    const manifestResult = await verifyReleaseManifest(resolve(root, 'dist'))
    violations.push(...manifestResult.violations)
  }

  return { ok: violations.length === 0, violations: [...new Set(violations)].sort() }
}

async function main() {
  const requireDist = process.argv.includes('--require-dist')
  const result = await scanPublicTree(process.cwd(), { requireDist })
  if (!result.ok) {
    console.error('Public boundary verification failed:')
    for (const violation of result.violations) console.error(`- ${violation}`)
    process.exitCode = 1
    return
  }
  console.log(requireDist ? 'Public source and dist boundary verified.' : 'Public source boundary verified.')
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
  await main()
}
