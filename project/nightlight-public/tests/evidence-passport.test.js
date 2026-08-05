import { readFile } from 'node:fs/promises'

import { describe, expect, it } from 'vitest'

import { EVENTS } from '../src/content/study.js'

const passportUrl = new URL('../src/content/evidencePassportArtifact.js', import.meta.url)
const thirdPartyNoticesUrl = new URL('../THIRD_PARTY_NOTICES.md', import.meta.url)

async function loadPassportModule() {
  try {
    return await import(passportUrl.href)
  } catch (loadError) {
    return { loadError }
  }
}

function collectKeys(value, keys = []) {
  if (!value || typeof value !== 'object') return keys
  for (const [key, nested] of Object.entries(value)) {
    keys.push(key)
    collectKeys(nested, keys)
  }
  return keys
}

describe('Public Evidence Passport Artifact v1', () => {
  it('publishes exactly nine reviewed event passports and leaves sixteen events unassessed', async () => {
    const module = await loadPassportModule()
    expect(module.loadError).toBeUndefined()

    const { PUBLIC_EVIDENCE_PASSPORT_ARTIFACT, validatePublicEvidencePassportArtifact } = module
    expect(validatePublicEvidencePassportArtifact(PUBLIC_EVIDENCE_PASSPORT_ARTIFACT)).toEqual([])
    expect(PUBLIC_EVIDENCE_PASSPORT_ARTIFACT.version).toBe('1.0.0')
    expect(PUBLIC_EVIDENCE_PASSPORT_ARTIFACT.passports.map(({ eventId }) => eventId)).toEqual([
      'eq-pr',
      'ida',
      'ian-charlotte',
      'ian-fortmyers',
      'irma',
      'laura',
      'eq-hatay',
      'maria',
      'michael',
    ])

    const assessed = new Set(PUBLIC_EVIDENCE_PASSPORT_ARTIFACT.passports.map(({ eventId }) => eventId))
    expect(EVENTS.filter(({ id }) => !assessed.has(id))).toHaveLength(16)
    expect(JSON.stringify(PUBLIC_EVIDENCE_PASSPORT_ARTIFACT)).not.toMatch(/dorian|freeport/i)
  })

  it('keeps five components separate and omits overall scores and reconstructable inputs', async () => {
    const module = await loadPassportModule()
    expect(module.loadError).toBeUndefined()
    const artifact = module.PUBLIC_EVIDENCE_PASSPORT_ARTIFACT

    expect(artifact.componentDefinitions.map(({ id, maxPoints }) => [id, maxPoints])).toEqual([
      ['observation-quality', 30],
      ['post-event-coverage', 20],
      ['context-coverage', 20],
      ['covariate-completeness', 20],
      ['data-integrity', 10],
    ])
    for (const passport of artifact.passports) {
      expect(passport.components).toHaveLength(5)
      expect(passport.supportedClaim).toBeTruthy()
      expect(passport.unsupportedClaim).toMatch(/community|recovery|outcome|ranking/i)
      expect(passport.publicationStatus).toBe('reviewed-derived-aggregate')
    }
    expect(collectKeys(artifact).join('\n')).not.toMatch(/eventCount|observedRate|highCensoringShare|poiCount|totalScore|increment|recommendedRole|facility|time.?series|grid|zip/i)
  })

  it('resolves assessed events and returns null instead of inventing missing passports', async () => {
    const module = await loadPassportModule()
    expect(module.loadError).toBeUndefined()

    expect(module.evidencePassportByEventId('ian-fortmyers')?.readinessBand).toBe('mainline_ready')
    expect(module.evidencePassportByEventId('matthew-jax')).toBeNull()
    expect(module.evidencePassportByEventId('dorian-freeport')).toBeNull()
  })

  it.each([
    ['an unknown top-level field', (artifact) => { artifact.privatePath = 'not public' }],
    ['an overall score', (artifact) => { artifact.passports[0].totalScore = 92 }],
    ['a substituted component value', (artifact) => { artifact.passports[0].components[0].points = 0 }],
    ['an outcome ranking label', (artifact) => { artifact.passports[0].readinessLabel = 'Best recovery' }],
    ['an unreviewed source hash', (artifact) => { artifact.source.sha256 = 'a'.repeat(64) }],
    ['a duplicated event passport', (artifact) => { artifact.passports.push(structuredClone(artifact.passports[0])) }],
  ])('rejects %s', async (_, mutate) => {
    const module = await loadPassportModule()
    expect(module.loadError).toBeUndefined()
    const candidate = structuredClone(module.PUBLIC_EVIDENCE_PASSPORT_ARTIFACT)
    mutate(candidate)

    expect(module.validatePublicEvidencePassportArtifact(candidate).join('\n')).toMatch(/unknown|overall|reviewed|component|label|duplicate|source/i)
  })
})

describe('Atlas Evidence Passport surface', () => {
  it('renders semantic assessed and not-assessed states with a scientific boundary', async () => {
    const atlas = await readFile(new URL('../src/views/AtlasView.vue', import.meta.url), 'utf8')

    expect(atlas).toMatch(/evidencePassportByEventId/)
    expect(atlas).toMatch(/Evidence passport/)
    expect(atlas).toMatch(/Not assessed in v1/)
    expect(atlas).toMatch(/<table/)
    expect(atlas).toMatch(/<caption/)
    expect(atlas).toMatch(/supportedClaim/)
    expect(atlas).toMatch(/unsupportedClaim/)
    expect(atlas).toMatch(/analysis admission heuristic/i)
  })

  it('pins the WorldPop notice to the actual Turkey 2020 layer record', async () => {
    const notices = await readFile(thirdPartyNoticesUrl, 'utf8')

    expect(notices).toContain('https://hub.worldpop.org/geodata/summary?id=6443')
    expect(notices).toContain('Global_2000_2020/2020/TUR/tur_ppp_2020.tif')
    expect(notices).toContain('10.5258/SOTON/WP00645')
    expect(notices).not.toMatch(/summary\?id=49896|WP00684/)
  })
})
