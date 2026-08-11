import { afterEach, describe, expect, it, vi } from 'vitest'

import { ArtifactError, ArtifactSchemaError, UnsupportedArtifactVersionError } from '../errors'
import type { ArtifactSource } from '../ports'
import { HttpArtifactSource } from '../sources/httpArtifactSource'
import { ResultsSummaryRepository } from './resultsSummaryRepository'

const RESULTS_FIXTURE = {
  feature_importance: [{ feature: 'log_dist', rf_imp: 0.54, xgb_imp: 0.24, avg_imp: 0.39 }],
  model_comparison: {
    model_a: { name: 'Model A', mean_auc: 0.967, std: 0.023, f1: 0.83 },
  },
  loeo_by_model: {
    A: [{ held_out: 'Maria_SanJuan', rf_auc: 0.945, xgb_auc: 0.949, logit_auc: 0.914 }],
  },
  prob_stats: {
    maria: { n: 1260, min: 0.05, max: 0.957, mean: 0.456, p25: 0.178, p50: 0.414, p75: 0.659, above_50: 517, above_80: 260 },
  },
}

class FakeSource implements ArtifactSource {
  readonly read = vi.fn<ArtifactSource['read']>()
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('HTTP artifact source errors', () => {
  it('classifies an aborted request', async () => {
    const abortError = Object.assign(new Error('aborted'), { name: 'AbortError' })
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(abortError))

    await expect(new HttpArtifactSource('/base/').read('data/results.json')).rejects.toMatchObject({
      code: 'aborted',
      path: 'data/results.json',
    })
  })

  it('classifies a network failure without returning fallback data', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')))

    await expect(new HttpArtifactSource('/base/').read('data/results.json')).rejects.toMatchObject({
      code: 'network',
    })
  })

  it('distinguishes 404 from other HTTP failures', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 404 }))
    await expect(new HttpArtifactSource('/base/').read('data/missing.json')).rejects.toMatchObject({
      code: 'not-found',
      status: 404,
    })

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 503 }))
    await expect(new HttpArtifactSource('/base/').read('data/down.json')).rejects.toMatchObject({
      code: 'http',
      status: 503,
    })
  })

  it('distinguishes invalid JSON from schema failure', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => '{not json',
    }))

    await expect(new HttpArtifactSource('/base/').read('data/results.json')).rejects.toMatchObject({
      code: 'invalid-json',
    })
  })
})

describe('results summary repository', () => {
  it('caches only validated success', async () => {
    const source = new FakeSource()
    source.read.mockResolvedValue(RESULTS_FIXTURE)
    const repository = new ResultsSummaryRepository(source)

    await expect(repository.get()).resolves.toMatchObject({ wireVersion: 'legacy-v0' })
    await expect(repository.get()).resolves.toMatchObject({ data: RESULTS_FIXTURE })
    expect(source.read).toHaveBeenCalledTimes(1)
  })

  it('does not cache invalid schema', async () => {
    const source = new FakeSource()
    source.read.mockResolvedValue({ ...RESULTS_FIXTURE, prob_stats: null })
    const repository = new ResultsSummaryRepository(source)

    await expect(repository.get()).rejects.toBeInstanceOf(ArtifactSchemaError)
    await expect(repository.get()).rejects.toBeInstanceOf(ArtifactSchemaError)
    expect(source.read).toHaveBeenCalledTimes(2)
  })

  it('fails closed on unsupported versions', async () => {
    const source = new FakeSource()
    source.read.mockResolvedValue({ schemaVersion: 99, data: RESULTS_FIXTURE })

    await expect(new ResultsSummaryRepository(source).get())
      .rejects.toBeInstanceOf(UnsupportedArtifactVersionError)
  })

  it('does not turn unavailable data into zero or an empty result', async () => {
    const source = new FakeSource()
    source.read.mockRejectedValue(new ArtifactError('network', 'offline'))

    await expect(new ResultsSummaryRepository(source).get()).rejects.toMatchObject({ code: 'network' })
  })

  it('rejects an already-aborted call before reading or serving cache', async () => {
    const source = new FakeSource()
    source.read.mockResolvedValue(RESULTS_FIXTURE)
    const repository = new ResultsSummaryRepository(source)
    await repository.get()
    const controller = new AbortController()
    controller.abort()

    await expect(repository.get({ signal: controller.signal })).rejects.toMatchObject({ code: 'aborted' })
    expect(source.read).toHaveBeenCalledTimes(1)
  })
})
