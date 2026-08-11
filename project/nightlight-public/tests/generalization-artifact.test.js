import { readFile } from 'node:fs/promises'

import { describe, expect, it } from 'vitest'

import {
  PUBLIC_GENERALIZATION_ARTIFACT,
  validatePublicGeneralizationArtifact,
} from '../src/content/generalizationArtifact.js'

describe('Public Generalization Artifact v1', () => {
  it('admits only traceable aggregate metrics with explicit scientific roles', () => {
    expect(validatePublicGeneralizationArtifact(PUBLIC_GENERALIZATION_ARTIFACT)).toEqual([])
    expect(PUBLIC_GENERALIZATION_ARTIFACT.version).toBe('1.0.0')
    expect(PUBLIC_GENERALIZATION_ARTIFACT.metrics).toHaveLength(3)

    expect(PUBLIC_GENERALIZATION_ARTIFACT.metrics).toMatchObject([
      {
        id: 'in-sample-r-squared',
        validationDesign: 'in-sample, fixed-control descriptive fit',
        modelRole: 'explanatory',
        metricType: 'description',
        value: 0.7603,
        unit: 'coefficient-of-determination [0–1]',
        supportedClaim: 'The specified model describes variation within the analyzed, fixed-control sample.',
        unsupportedClaim: 'Future-event accuracy, causation, or a ranking of community recovery.',
      },
      {
        id: 'cross-event-logit-auc',
        validationDesign: 'leave-one-event-out cross-event damage-ranking',
        modelRole: 'damage-ranking',
        metricType: 'ranking',
        value: 0.4814,
        unit: 'area-under-curve [0–1]',
        supportedClaim: 'In this held-out-event damage-ranking design, the admitted ranking result is below a 0.50 reference.',
        unsupportedClaim: 'A recovery-transport result, calibrated probability, future-event readiness, or a community recovery ranking.',
      },
      {
        id: 'descriptive-sensitivity-ratio',
        validationDesign: 'descriptive sensitivity; not a cross-event validation',
        modelRole: 'secondary-interpretation',
        metricType: 'description',
        value: 0.551,
        unit: 'ratio [unitless]',
        supportedClaim: 'A descriptive sensitivity value under its stated analysis conditions.',
        unsupportedClaim: 'A fairness conclusion, causal mechanism, or transport improvement.',
      },
    ])

    const crossEventDamageRanking = PUBLIC_GENERALIZATION_ARTIFACT.metrics.find((metric) => metric.id === 'cross-event-logit-auc')
    expect(crossEventDamageRanking.modelFamily).toBe('Cross-event logit damage-ranking model')
    expect(crossEventDamageRanking.validationDesign).toBe('leave-one-event-out cross-event damage-ranking')

    for (const metric of PUBLIC_GENERALIZATION_ARTIFACT.metrics) {
      expect(metric.cohort).toBeTruthy()
      expect(metric.sampleLock).toBeTruthy()
      expect(metric.validationDesign).toBeTruthy()
      expect(metric.modelRole).toMatch(/^(explanatory|damage-ranking|recovery-transport|secondary-interpretation)$/)
      expect(metric.metricType).toMatch(/^(description|ranking|calibration)$/)
      expect(metric.sourceArtifactId).toBeTruthy()
      expect(metric.supportedClaim).toBeTruthy()
      expect(metric.unsupportedClaim).toBeTruthy()
    }
  })

  it('allows a withheld metric only when it carries no public value and names the limitation', () => {
    const candidate = structuredClone(PUBLIC_GENERALIZATION_ARTIFACT)
    candidate.metrics[0].publicationStatus = 'withheld'
    candidate.metrics[0].qualityStatus = 'withheld'
    candidate.metrics[0].value = null
    candidate.metrics[0].withheldReason = 'No public value is admitted for this candidate.'

    expect(validatePublicGeneralizationArtifact(candidate)).toEqual([])
  })

  it.each([
    ['a restricted field', (artifact) => { artifact.metrics[0].facility = 'not allowed' }],
    ['an unknown coordinate-shaped field', (artifact) => { artifact.metrics[0].latitude = 18.4 }],
    ['an unknown source-path field', (artifact) => { artifact.sources[0].sourcePath = 'not public' }],
    ['an undeclared unit', (artifact) => { artifact.metrics[0].unit = 'percent' }],
    ['missing source lineage', (artifact) => { artifact.metrics[0].sourceArtifactId = '' }],
    ['an unsupported metric type', (artifact) => { artifact.metrics[0].metricType = 'prediction' }],
    ['a metric-role combination that does not match its reviewed definition', (artifact) => { artifact.metrics[0].metricType = 'calibration' }],
    ['a reviewed metric with a substituted value', (artifact) => { artifact.metrics[1].value = 0.9999 }],
    ['an out-of-range bounded value', (artifact) => { artifact.metrics[1].value = 2 }],
    ['a damage-ranking metric relabeled as recovery transport', (artifact) => { artifact.metrics[1].modelRole = 'recovery-transport' }],
    ['a withheld metric that still carries a public value', (artifact) => {
      artifact.metrics[0].publicationStatus = 'withheld'
      artifact.metrics[0].qualityStatus = 'withheld'
    }],
    ['an untraceable source hash', (artifact) => { artifact.sources[0].sha256 = 'not-a-hash' }],
    ['a syntactically valid but unreviewed source hash', (artifact) => { artifact.sources[0].sha256 = 'a'.repeat(64) }],
  ])('rejects %s', (_, mutate) => {
    const candidate = structuredClone(PUBLIC_GENERALIZATION_ARTIFACT)
    mutate(candidate)

    expect(validatePublicGeneralizationArtifact(candidate).join('\n')).toMatch(/restricted|unknown|unit|lineage|metric type|sha-256|reviewed source hash|reviewed metric definition|range|withheld/i)
  })
})

describe('Generalization Autopsy accessibility shell', () => {
  it('keeps five routes reachable without moving the sequential focus start point', async () => {
    const app = await readFile(new URL('../src/App.vue', import.meta.url), 'utf8')
    const styles = await readFile(new URL('../src/styles/main.css', import.meta.url), 'utf8')
    const navigationReveal = app.match(/function revealActiveNavigation\(\) \{([\s\S]*?)\n\}/)?.[1]

    expect(app).toMatch(/navigation\.scrollLeft\s*=/)
    expect(navigationReveal).toBeDefined()
    expect(navigationReveal).not.toMatch(/scrollIntoView/)
    expect(app).toMatch(/function updateRouteContext\(\)[\s\S]*?revealActiveNavigation\(\)[\s\S]*?onMounted\(updateRouteContext\)/)
    expect(app).toMatch(/aria-current/)
    expect(styles).toMatch(/\.site-nav\s*\{[\s\S]*overflow-x:\s*auto/)
    expect(styles).toMatch(/:focus-visible/)
    expect(styles).toMatch(/\.site-footer\s*\{[\s\S]*color:\s*var\(--muted\)/)
    expect(styles).toMatch(/@media \(prefers-reduced-motion: reduce\)/)
    expect(styles).toMatch(/animation-duration:\s*0\.01ms !important/)
  })

  it('gives findings a text summary and semantic alternatives rather than a color-only chart', async () => {
    const findings = await readFile(new URL('../src/views/FindingsView.vue', import.meta.url), 'utf8')

    expect(findings).toMatch(/This is an analysis of model transport failure, not a ranking of community recovery\./)
    expect(findings).toMatch(/<table/)
    expect(findings).toMatch(/<caption/)
    expect(findings).toMatch(/What improved \/ what failed/)
    expect(findings).toMatch(/Evidence cards/)
    expect(findings).toMatch(/The cross-event damage-ranking check trains without one event/)
    expect(findings).toMatch(/Description, damage ranking, and recovery transport are separate roles/)
    expect(findings).toMatch(/Recovery transport[\s\S]{0,160}<h3>Unavailable<\/h3>/)
    expect(findings).toMatch(/R² \{\{ metrics\['in-sample-r-squared'\]\.value\.toFixed\(4\) \}\}/)
    expect(findings).toMatch(/AUC \{\{ metrics\['cross-event-logit-auc'\]\.value\.toFixed\(4\) \}\}/)
    expect(findings).toMatch(/R² and AUC[\s\S]{0,160}never placed on one shared score scale/)
    expect(findings).toMatch(/<\/div>\s*<figcaption id="fit-caption">/)
    expect(findings.match(/<figcaption/g)).toHaveLength(1)
    expect(findings).not.toMatch(/The transport check trains/)
  })

  it('keeps the findings summary, navigation, and lineage disclosure semantically distinct', async () => {
    const findings = await readFile(new URL('../src/views/FindingsView.vue', import.meta.url), 'utf8')

    expect(findings).toMatch(/class="page-lede page-summary"/)
    expect(findings).toMatch(/<nav class="in-page-nav"/)
    expect(findings).toMatch(/<ol class="content-section-nav">/)
    for (const id of ['attractive-result-title', 'harder-test-title', 'role-matrix-title', 'evidence-cards-title']) {
      expect(findings).toContain(`href="#${id}"`)
      expect(findings).toContain(`id="${id}" tabindex="-1"`)
    }
    expect(findings).toMatch(/<details class="evidence-card__lineage">[\s\S]*Cohort \/ lock[\s\S]*Source/)
  })

  it('documents metric-role and lineage limits at the public contract boundary', async () => {
    const methods = await readFile(new URL('../src/views/MethodsView.vue', import.meta.url), 'utf8')
    const policy = await readFile(new URL('../DATA_POLICY.md', import.meta.url), 'utf8')

    expect(methods).toMatch(/Public Generalization Artifact v1/)
    expect(methods).toMatch(/description, ranking, and calibration/)
    expect(policy).toMatch(/source artifact/i)
    expect(policy).toMatch(/SHA-256/)
  })

  it('keeps source evidence as a public pointer instead of bundling the cross-event artifact', async () => {
    const artifact = JSON.stringify(PUBLIC_GENERALIZATION_ARTIFACT)

    expect(artifact).toContain('cross-event-stop-decision-v3x')
    expect(artifact).not.toMatch(/project[\\/]modeling|\.json/i)
  })
})
