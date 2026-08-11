import { createHash } from 'node:crypto'
import { lstat, readdir, readFile } from 'node:fs/promises'
import { basename, extname, relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

import { verifyReleaseManifest } from './release-manifest.mjs'
import { PUBLIC_EVIDENCE_PASSPORT_ARTIFACT, validatePublicEvidencePassportArtifact } from '../src/content/evidencePassportArtifact.js'
import { PUBLIC_GENERALIZATION_ARTIFACT, validatePublicGeneralizationArtifact } from '../src/content/generalizationArtifact.js'

const ignoredRootDirectories = new Set(['node_modules', '.git', '.vercel'])
const allowedTopLevel = new Set([
  '.github',
  'src',
  'tests',
  'scripts',
  'public',
  'DOCS',
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
  'SECURITY.md',
  'THIRD_PARTY_NOTICES.md',
  'USER_STUDY_PROTOCOL.md',
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
  'SECURITY.md',
  'DOCS/archive/proxy-evidence-phase-20260809/context.md',
  'DOCS/archive/proxy-evidence-phase-20260809/plan.md',
  'DOCS/archive/proxy-evidence-phase-20260809/proxy-evaluation-report.md',
  'DOCS/archive/proxy-evidence-phase-20260809/task.md',
  'LICENSE',
  'README.md',
  'THIRD_PARTY_NOTICES.md',
  'USER_STUDY_PROTOCOL.md',
  'index.html',
  'package-lock.json',
  'package.json',
  'public/observatory-mark.svg',
  'scripts/release-manifest.mjs',
  'scripts/verify-public.mjs',
  'src/App.vue',
  'src/components/LocalResearchLog.vue',
  'src/content/copy.js',
  'src/content/evidencePassportArtifact.js',
  'src/content/evidencePassportManifest.json',
  'src/content/generalizationArtifact.js',
  'src/content/study.js',
  'src/domain/compareEvents.js',
  'src/domain/filterEvents.js',
  'src/domain/localAnalyticsContract.js',
  'src/domain/projectPoint.js',
  'src/domain/resolveSelectedId.js',
  'src/lib/aggregateValueContract.js',
  'src/lib/bundledSourceAdapter.js',
  'src/lib/localResearchAnalytics.js',
  'src/lib/sourceFreshnessContract.js',
  'src/main.js',
  'src/router/routes.js',
  'src/styles/main.css',
  'src/views/AtlasView.vue',
  'src/views/CreditsView.vue',
  'src/views/FindingsView.vue',
  'src/views/MethodsView.vue',
  'src/views/OverviewView.vue',
  'tests/copy.test.js',
  'tests/compare-events.test.js',
  'tests/evidence-passport.test.js',
  'tests/error-contract.test.js',
  'tests/generalization-artifact.test.js',
  'tests/local-analytics.test.js',
  'tests/platform-boundary.test.js',
  'tests/public-boundary.test.js',
  'tests/proxy-evaluation.test.js',
  'tests/release-manifest.test.js',
  'tests/routes.test.js',
  'tests/source-freshness-contract.test.js',
  'tests/static-shell.test.js',
  'tests/study.test.js',
  'tests/support/temporaryRoot.js',
  'tests/user-study-protocol.test.js',
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
  /\bnavigator\.sendBeacon\s*\(/,
  /\bimport\s*\(\s*['"]https?:\/\//,
  /\bnew\s+(?:Request|Worker|SharedWorker)\s*\(\s*['"]https?:\/\//,
  /\b(?:window|document)\.location(?:\.href)?\s*=\s*['"]https?:\/\//,
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
const allowedRuntimeDependencies = new Set(['vue', 'vue-router'])
const allowedDevelopmentDependencies = new Set(['@vitejs/plugin-vue', 'vite', 'vitest'])
const dependencyNoticeNames = new Map([
  ['vue', 'Vue'],
  ['vue-router', 'Vue Router'],
  ['vite', 'Vite'],
  ['@vitejs/plugin-vue', '@vitejs/plugin-vue'],
  ['vitest', 'Vitest'],
])
const requiredCspDirectives = new Map([
  ['default-src', ["'self'"]],
  ['script-src', ["'self'"]],
  ['style-src', ["'self'"]],
  ['img-src', ["'self'"]],
  ['font-src', ["'self'"]],
  ['connect-src', ["'none'"]],
  ['object-src', ["'none'"]],
  ['base-uri', ["'self'"]],
  ['form-action', ["'none'"]],
])

function normalizePath(path) {
  return path.split(sep).join('/')
}

function canonicalSha256(contents) {
  const canonical = contents.replace(/\r\n?/g, '\n')
  return createHash('sha256').update(canonical, 'utf8').digest('hex')
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

function securityMetadataViolations(contents, relativePath) {
  const violations = []
  const cspTag = contents.match(/<meta\b[^>]*http-equiv=["']Content-Security-Policy["'][^>]*>/i)?.[0]
  const csp = cspTag?.match(/\bcontent=(["'])(.*?)\1/i)?.[2]
  if (!csp) {
    violations.push(`${relativePath}: Content-Security-Policy meta is missing or malformed`)
  } else {
    const directives = new Map()
    for (const declaration of csp.split(';').map((value) => value.trim()).filter(Boolean)) {
      const [name, ...values] = declaration.split(/\s+/)
      if (directives.has(name)) violations.push(`${relativePath}: duplicate CSP directive ${name}`)
      directives.set(name, values)
    }
    for (const [name, expectedValues] of requiredCspDirectives) {
      if (JSON.stringify(directives.get(name)) !== JSON.stringify(expectedValues)) {
        violations.push(`${relativePath}: CSP directive ${name} must be ${expectedValues.join(' ')}`)
      }
    }
    if (/['"]unsafe-(?:inline|eval)['"]/i.test(csp) || /\bhttps?:/i.test(csp)) {
      violations.push(`${relativePath}: CSP must not permit unsafe or external runtime sources`)
    }
  }

  const referrerTag = contents.match(/<meta\b[^>]*name=["']referrer["'][^>]*>/i)?.[0]
  const referrer = referrerTag?.match(/\bcontent=(["'])(.*?)\1/i)?.[2]
  if (referrer !== 'no-referrer') violations.push(`${relativePath}: referrer meta must be no-referrer`)
  return violations
}

function externalLinkViolations(contents, relativePath) {
  const violations = []
  for (const match of contents.matchAll(/<a\b[^>]*>/gi)) {
    const tag = match[0]
    const href = tag.match(/\bhref=["'](https?:\/\/[^"']+)["']/i)?.[1]
    const target = tag.match(/\btarget=["']([^"']+)["']/i)?.[1]
    if (!href || target?.toLocaleLowerCase() !== '_blank') continue
    const rel = new Set((tag.match(/\brel=["']([^"']*)["']/i)?.[1] || '').split(/\s+/).filter(Boolean))
    if (!rel.has('noopener') || !rel.has('noreferrer')) {
      violations.push(`${relativePath}: external target=_blank link must use rel="noopener noreferrer"`)
    }
  }
  return violations
}

async function optionalText(path) {
  try {
    return await readFile(path, 'utf8')
  } catch (error) {
    if (error?.code === 'ENOENT') return null
    throw error
  }
}

function validateDependencySection(packageJson, section, allowedNames, violations) {
  const dependencies = packageJson[section] || {}
  for (const [name, version] of Object.entries(dependencies)) {
    if (!allowedNames.has(name)) violations.push(`package.json: ${section} dependency ${name} is not allowlisted`)
    if (!/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(version)) {
      violations.push(`package.json: ${section} dependency ${name} must use an exact version`)
    }
  }
  for (const name of allowedNames) {
    if (!(name in dependencies)) violations.push(`package.json: required ${section} dependency ${name} is missing`)
  }
}

async function dependencyPolicyViolations(root) {
  const violations = []
  const packageText = await optionalText(resolve(root, 'package.json'))
  if (packageText === null) return violations

  let packageJson
  try {
    packageJson = JSON.parse(packageText)
  } catch (error) {
    return [`package.json: invalid JSON (${error.name})`]
  }
  validateDependencySection(packageJson, 'dependencies', allowedRuntimeDependencies, violations)
  validateDependencySection(packageJson, 'devDependencies', allowedDevelopmentDependencies, violations)
  if (packageJson.engines?.node !== '>=20') violations.push('package.json: Node engine boundary must remain >=20')

  const lockText = await optionalText(resolve(root, 'package-lock.json'))
  const notices = await optionalText(resolve(root, 'THIRD_PARTY_NOTICES.md'))
  if (lockText === null) return [...violations, 'package-lock.json: required when package.json is present']
  if (notices === null) violations.push('THIRD_PARTY_NOTICES.md: required when package.json is present')

  let lock
  try {
    lock = JSON.parse(lockText)
  } catch (error) {
    return [...violations, `package-lock.json: invalid JSON (${error.name})`]
  }

  const declared = { ...packageJson.dependencies, ...packageJson.devDependencies }
  for (const [name, version] of Object.entries(declared)) {
    const locked = lock.packages?.[`node_modules/${name}`]
    if (lock.packages?.['']?.dependencies?.[name] !== version && lock.packages?.['']?.devDependencies?.[name] !== version) {
      violations.push(`package-lock.json: root declaration for ${name} does not match package.json`)
    }
    if (locked?.version !== version) violations.push(`package-lock.json: ${name} is not locked to ${version}`)
    if (locked?.license !== 'MIT') violations.push(`package-lock.json: ${name} does not declare the reviewed MIT license`)
    const noticeName = dependencyNoticeNames.get(name)
    if (notices !== null && noticeName && !notices.includes(`${noticeName} ${version}`)) {
      violations.push(`THIRD_PARTY_NOTICES.md: ${name}@${version} notice is missing`)
    }
  }

  const esbuildVersion = lock.packages?.['node_modules/esbuild']?.version
  const allowedScripts = packageJson.allowScripts || {}
  if (
    !esbuildVersion
    || Object.keys(allowedScripts).length !== 1
    || allowedScripts[`esbuild@${esbuildVersion}`] !== true
  ) {
    violations.push('package.json: only the locked esbuild install script may be approved')
  }
  return violations
}

export async function scanPublicTree(
  rootPath,
  { requireDist = false, requireReviewedArtifacts = false } = {},
) {
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
      if (relativePath === 'index.html' || relativePath === 'dist/index.html') {
        violations.push(...securityMetadataViolations(contents, relativePath))
      }
      if (extension === '.vue' || extension === '.html') {
        violations.push(...externalLinkViolations(contents, relativePath))
      }
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
  violations.push(...await dependencyPolicyViolations(root))
  const artifactViolations = validatePublicGeneralizationArtifact(PUBLIC_GENERALIZATION_ARTIFACT)
  violations.push(...artifactViolations.map((violation) => `src/content/generalizationArtifact.js: ${violation}`))
  const passportViolations = validatePublicEvidencePassportArtifact(PUBLIC_EVIDENCE_PASSPORT_ARTIFACT)
  violations.push(...passportViolations.map((violation) => `src/content/evidencePassportArtifact.js: ${violation}`))
  if (requireReviewedArtifacts) {
    const passportManifestPath = resolve(root, 'src/content/evidencePassportManifest.json')
    try {
      const passportManifest = await readFile(passportManifestPath, 'utf8')
      if (canonicalSha256(passportManifest) !== PUBLIC_EVIDENCE_PASSPORT_ARTIFACT.source.sha256) {
        violations.push('src/content/evidencePassportManifest.json: canonical hash does not match the reviewed source lineage')
      }
    } catch (error) {
      if (error?.code !== 'ENOENT') throw error
      violations.push('src/content/evidencePassportManifest.json: reviewed manifest is missing')
    }
  }
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
  const result = await scanPublicTree(process.cwd(), {
    requireDist,
    requireReviewedArtifacts: true,
  })
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
