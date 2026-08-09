import { readFile } from 'node:fs/promises'
import { describe, expect, it } from 'vitest'

const phaseRoot = new URL('../DOCS/archive/proxy-evidence-phase-20260809/', import.meta.url)

async function readOptional(relativePath) {
  try {
    return await readFile(new URL(relativePath, phaseRoot), 'utf8')
  } catch (error) {
    if (error?.code === 'ENOENT') return null
    throw error
  }
}

describe('research-supported proxy evaluation', () => {
  it('reports evidence-layer judgments without presenting proxy checks as participant validation', async () => {
    const report = await readOptional('proxy-evaluation-report.md')

    expect(report).not.toBeNull()
    expect(report).toMatch(/research-supported proxy evaluation/i)
    expect(report).toMatch(/## Allowed conclusions/i)
    expect(report).toMatch(/## Prohibited conclusions/i)
    expect(report).toMatch(/technical accessibility contract/i)
    expect(report).toMatch(/information clarity proxy/i)
    expect(report).toMatch(/scientific understanding remains unknown/i)
    expect(report).toMatch(/Judgment \| Product evidence \| External basis \| Confidence \| Limitations \| Prohibited product claim/i)
    expect(report).toMatch(/No final CDC (?:CCI|Clear Communication Index) score is reported/i)
    expect(report).not.toMatch(/CDC (?:CCI|Clear Communication Index) score\s*[:=]\s*\d/i)
    expect(report).not.toMatch(/participant(?:s)? (?:completed|understood|validated|confirmed)/i)
  })

  it('links every required official source used by the proxy judgments', async () => {
    const report = await readOptional('proxy-evaluation-report.md') ?? ''
    const officialUrls = [
      'https://www.w3.org/TR/WCAG22/',
      'https://www.w3.org/WAI/WCAG22/Understanding/focus-order.html',
      'https://www.w3.org/WAI/WCAG22/Understanding/headings-and-labels',
      'https://www.w3.org/TR/coga-usable/',
      'https://www.cdc.gov/ccindex/',
      'https://www.cdc.gov/ccindex/tool/how-to-use.html',
      'https://www.gov.uk/service-manual/user-research/using-moderated-usability-testing',
    ]

    for (const url of officialUrls) expect(report).toContain(url)
  })

  it('keeps the owner-run next phase recruitment-free and labels AI review as non-human', async () => {
    const plan = await readOptional('plan.md')

    expect(plan).not.toBeNull()
    expect(plan).toMatch(/automated semantic contract/i)
    expect(plan).toMatch(/content evidence matrix/i)
    expect(plan).toMatch(/two-pass AI\/adversarial review/i)
    expect(plan).toMatch(/AI output is a heuristic review note, not a user quote, participant observation, comprehension result, or authority/i)
    expect(plan).toMatch(/accessibility regression and browser gate/i)
    expect(plan).toMatch(/claim audit and release gate/i)
    expect(plan).toMatch(/optional escalation to human research/i)
    expect(plan).not.toMatch(/recruit(?:ment)? (?:is )?required/i)
  })

  it('marks the unused participant protocol as deferred optional history, not a current gate', async () => {
    const protocol = await readFile(new URL('../USER_STUDY_PROTOCOL.md', import.meta.url), 'utf8')

    expect(protocol).toMatch(/deferred, optional historical instrument/i)
    expect(protocol).toMatch(/not (?:a|the) current release gate/i)
    expect(protocol).toMatch(/No sessions have been run/i)
  })
})
