import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'

import { scanPublicTree } from '../scripts/verify-public.mjs'

const tempRoots = []

async function makeTree(files) {
  const root = await mkdtemp(join(tmpdir(), 'nightlight-public-test-'))
  tempRoots.push(root)
  for (const [relativePath, contents] of Object.entries(files)) {
    const absolutePath = join(root, relativePath)
    await mkdir(join(absolutePath, '..'), { recursive: true })
    await writeFile(absolutePath, contents)
  }
  return root
}

afterEach(async () => {
  await Promise.all(tempRoots.splice(0).map((root) => rm(root, { recursive: true, force: true })))
})

describe('fail-closed public bundle scanner', () => {
  it.each([
    ['public/raw.csv', 'x,y', 'public/raw.csv'],
    ['public/raster.tif', 'binary', 'public/raster.tif'],
    ['public/map.geojson', '{}', 'public/map.geojson'],
    ['public/table.parquet', 'binary', 'public/table.parquet'],
    ['public/model.pkl', 'binary', 'public/model.pkl'],
    ['public/workbook.xlsx', 'binary', 'public/workbook.xlsx'],
    ['public/report.docx', 'binary', 'public/report.docx'],
    ['public/notebook.ipynb', '{}', 'public/notebook.ipynb'],
    ['public/app.js.map', '{}', 'public/app.js.map'],
    ['public/private.pem', 'secret', 'public/private.pem'],
    ['public/private.key', 'secret', 'public/private.key'],
    ['public/tool.exe', 'binary', 'public/tool.exe'],
    ['public/private.npy', 'binary', 'public/private.npy'],
    ['public/payload.bin', 'binary', 'public/payload.bin'],
    ['public/payload.dat', 'binary', 'public/payload.dat'],
    ['public/renamed.json', '{}', 'public/renamed.json'],
    ['public/extra.svg', '<svg/>', 'public/extra.svg'],
    ['src/content/extra.js', 'export const hidden = []', 'src/content/extra.js'],
    ['public/facilities_event.json', '{}', 'public/facilities_event.json'],
    ['public/prob_event.json', '{}', 'public/prob_event.json'],
    ['public/ts_event.json', '{}', 'public/ts_event.json'],
    ['public/events_config.js', 'export default {}', 'public/events_config.js'],
    ['public/results_summary.json', '{}', 'public/results_summary.json'],
    ['data/summary.json', '{}', 'data'],
    ['src/data/study.js', 'export const count = 25', 'src/data'],
  ])('rejects prohibited artifact %s', async (relativePath, contents, expectedPath) => {
    const root = await makeTree({ [relativePath]: contents })
    const result = await scanPublicTree(root)

    expect(result.ok).toBe(false)
    expect(result.violations.join('\n')).toContain(expectedPath)
  })

  it('rejects credential-shaped content and runtime network calls', async () => {
    const root = await makeTree({
      'src/secret.js': `const token = '${'ghp_' + '1'.repeat(36)}'`,
      'src/network.js': `${['fet', 'ch'].join('')}('https://example.com/data.json')`,
    })
    const result = await scanPublicTree(root)

    expect(result.ok).toBe(false)
    expect(result.violations.join('\n')).toMatch(/credential|network/i)
  })

  it('allows official attribution hyperlinks but rejects external runtime resources', async () => {
    const attributionRoot = await makeTree({
      'src/views/CreditsView.vue': '<template><a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a></template>',
    })
    const externalResourceRoot = await makeTree({
      'index.html': '<link rel="stylesheet" href="https://example.com/external.css">',
    })

    await expect(scanPublicTree(attributionRoot)).resolves.toEqual({ ok: true, violations: [] })
    const externalResourceResult = await scanPublicTree(externalResourceRoot)
    expect(externalResourceResult.ok).toBe(false)
    expect(externalResourceResult.violations.join('\n')).toMatch(/network/i)
  })

  it('rejects raw-schema markers and local analysis paths in any public text file', async () => {
    const localPath = ['project', 'data', 'raw'].join('/')
    const restrictedField = ['customers', 'out'].join('_')
    const root = await makeTree({
      'README.md': `Internal source: ${localPath} with ${restrictedField}.`,
    })
    const result = await scanPublicTree(root)

    expect(result.ok).toBe(false)
    expect(result.violations.join('\n')).toMatch(/raw schema|local analysis path/i)
  })

  it('does not let nested dependency or Git directory names bypass scanning', async () => {
    const root = await makeTree({
      'public/node_modules/raw.csv': 'x,y',
      'dist/.git/payload.dat': 'binary',
    })
    const result = await scanPublicTree(root)

    expect(result.ok).toBe(false)
    expect(result.violations.join('\n')).toMatch(/nested|raw\.csv|payload\.dat/i)
  })

  it('scans the one approved SVG as text instead of trusting its extension', async () => {
    const restrictedField = ['customers', 'out'].join('_')
    const root = await makeTree({
      'public/observatory-mark.svg': `<svg><text>${restrictedField}</text></svg>`,
    })
    const result = await scanPublicTree(root)

    expect(result.ok).toBe(false)
    expect(result.violations.join('\n')).toMatch(/raw schema/i)
  })

  it('ignores the local Vercel link directory because Git excludes it', async () => {
    const root = await makeTree({
      'README.md': 'Public source tree.',
      '.vercel/project.json': '{"projectId":"prj_example","orgId":"team_example"}',
    })

    await expect(scanPublicTree(root)).resolves.toEqual({ ok: true, violations: [] })
  })

  it('rejects structured restricted derivative fields', async () => {
    const probabilityField = ['prob', 'ability'].join('')
    const zipField = ['zip', 'code'].join('_')
    const root = await makeTree({
      'src/content/unsafe.js': `export default { ${probabilityField}: 0.8, ${zipField}: "00000" }`,
    })
    const result = await scanPublicTree(root)

    expect(result.ok).toBe(false)
    expect(result.violations.join('\n')).toMatch(/restricted field/i)
  })

  it('rejects structured restricted fields in any public text file', async () => {
    const restrictedField = ['prob', 'ability'].join('')
    const root = await makeTree({
      'README.md': `Fixture: { ${restrictedField}: 0.8 }`,
    })
    const result = await scanPublicTree(root)

    expect(result.ok).toBe(false)
    expect(result.violations.join('\n')).toMatch(/restricted field/i)
  })

  it('accepts a minimal allowlisted source tree', async () => {
    const root = await makeTree({
      'src/content/study.js': 'export const count = 25',
      'src/App.vue': '<template><main>Public summary</main></template>',
      'README.md': 'No restricted data is published.',
    })
    const result = await scanPublicTree(root)

    expect(result).toEqual({ ok: true, violations: [] })
  })
})
