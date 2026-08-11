import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { chromium } from 'playwright'

const scriptDirectory = dirname(fileURLToPath(import.meta.url))
const [targetUrl, outputPathArgument] = process.argv.slice(2)

if (!targetUrl || !outputPathArgument) {
  throw new Error('Usage: node scripts/run-navigation-performance-cell.mjs <target-url> <output-json>')
}

const target = new URL(targetUrl)
if (!['http:', 'https:'].includes(target.protocol)) {
  throw new Error('The performance target must use HTTP(S)')
}

const outputPath = resolve(outputPathArgument)
const probeSource = await readFile(
  resolve(scriptDirectory, 'navigation-performance-probe.pw.js'),
  'utf8',
)
const probe = Function(`return (${probeSource})`)()
const browser = await chromium.launch({ headless: true })

try {
  const context = await browser.newContext()
  const page = await context.newPage()
  await page.goto(target.href, { waitUntil: 'domcontentloaded' })
  const serializedResult = await probe(page)
  const result = JSON.parse(serializedResult)

  await mkdir(dirname(outputPath), { recursive: true })
  await writeFile(outputPath, `${JSON.stringify(result, null, 2)}\n`, 'utf8')
  console.log(JSON.stringify({
    outputPath,
    schemaVersion: result.schemaVersion,
    samples: result.samples?.length ?? 0,
    errors: result.errors?.length ?? 0,
    browserLogs: result.browserLogs?.length ?? 0,
    failureProfile: result.protocol?.failureProfile ?? 'none',
  }))
} finally {
  await browser.close()
}
