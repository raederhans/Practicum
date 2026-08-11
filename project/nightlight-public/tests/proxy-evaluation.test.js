import { readFile } from 'node:fs/promises'
import { describe, expect, it } from 'vitest'

const phaseRoot = new URL('../DOCS/archive/proxy-evidence-phase-20260809/', import.meta.url)
const soloEvidenceUrl = new URL(
  '../../../DOCS/archive/p2-p3-solo-evidence-performance-20260810/p2-evidence.md',
  import.meta.url,
)

async function readOptional(relativePath) {
  try {
    return await readFile(new URL(relativePath, phaseRoot), 'utf8')
  } catch (error) {
    if (error?.code === 'ENOENT') return null
    throw error
  }
}

async function readSoloEvidence() {
  return readFile(soloEvidenceUrl, 'utf8')
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

describe('P2 independent solo evidence gate', () => {
  it('pins the exact target and official method boundaries without claiming human validation', async () => {
    const evidence = await readSoloEvidence()

    expect(evidence).toContain('ca8292040a402eae1d2e461708a4cc912867efcb')
    expect(evidence).toMatch(/accessed 2026-08-10/i)
    expect(evidence).toContain('https://www.cdc.gov/ccindex/tool/how-to-use.html')
    expect(evidence).toContain('https://designsystem.digital.gov/documentation/accessibility/')
    expect(evidence).toContain('https://www.w3.org/TR/WCAG22/')
    expect(evidence).toContain('https://www.w3.org/WAI/test-evaluate/conformance/')
    expect(evidence).toMatch(/No final CDC (?:CCI|Clear Communication Index) score/i)
    expect(evidence).toMatch(/scientific understanding remains unknown/i)
    expect(evidence).not.toMatch(/(?:usability|participant|assistive-technology) (?:is |was |has been )?validated/i)
  })

  it('records a searchable claim matrix with evidence, allowed meaning, prohibited meaning, and gate status', async () => {
    const evidence = await readSoloEvidence()
    const claimRows = evidence
      .split(/\r?\n/)
      .filter((line) => /^\| `claim-[^`]+` \|/.test(line))

    expect(evidence).toMatch(/## Material public claim-evidence matrix/i)
    expect(evidence).toMatch(/Claim ID \| Exact visible wording and route \| Evidence source \| Supported interpretation \| Prohibited interpretation \| Gate status/i)
    expect(claimRows.length).toBeGreaterThanOrEqual(6)
    for (const row of claimRows) {
      expect(row.split('|').length).toBeGreaterThanOrEqual(8)
      expect(row).toMatch(/\| (?:supported|withheld|unavailable|needs revision|blocked) \|$/i)
    }
  })

  it('labels both AI review passes as non-human and keeps them outside research evidence', async () => {
    const evidence = await readSoloEvidence()

    expect(evidence).toMatch(/### Pass A — non-human/i)
    expect(evidence).toMatch(/### Pass B — non-human adversarial/i)
    expect(evidence).toMatch(/AI output is a heuristic review note/i)
    expect(evidence).toMatch(/not a user quote, participant observation, comprehension result, or authority/i)
    expect(evidence).not.toMatch(/AI participant|synthetic participant|AI user/i)
  })

  it('keeps the release gate bounded to technical and content-proxy evidence', async () => {
    const evidence = await readSoloEvidence()

    expect(evidence).toMatch(/## Claim audit and proxy release gate/i)
    expect(evidence).toMatch(/must not claim WCAG 2\.2 conformance/i)
    expect(evidence).toMatch(/must not claim usability validation/i)
    expect(evidence).toMatch(/must not claim participant testing/i)
    expect(evidence).toMatch(/must not claim manual assistive-technology validation/i)
    expect(evidence).toMatch(/R².*future-event accuracy/i)
    expect(evidence).toMatch(/AUC.*recovery transport/i)
    expect(evidence).toMatch(/Not assessed.*zero/i)
  })
})
