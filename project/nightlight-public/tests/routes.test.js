import { readFile } from 'node:fs/promises'

import { describe, expect, it } from 'vitest'

import { routes } from '../src/router/routes.js'

describe('public navigation contract', () => {
  it('provides the five portfolio routes', () => {
    expect(routes.map(({ path, name }) => ({ path, name }))).toEqual([
      { path: '/', name: 'overview' },
      { path: '/atlas', name: 'atlas' },
      { path: '/findings', name: 'findings' },
      { path: '/methods', name: 'methods' },
      { path: '/credits', name: 'credits' },
    ])
  })

  it('lazy-loads every route view', () => {
    for (const route of routes) {
      expect(typeof route.component).toBe('function')
    }
  })

  it('gives every route a unique document title for navigation context', () => {
    expect(routes.map(({ meta }) => meta?.pageTitle)).toEqual([
      'Overview',
      'Study Atlas',
      'Findings',
      'Methods',
      'Credits / Policy',
    ])
    expect(new Set(routes.map(({ meta }) => meta?.pageTitle)).size).toBe(routes.length)
  })

  it('keeps the overview task, calls to action, facts, and detailed boundary in reading order', async () => {
    const overview = await readFile(new URL('../src/views/OverviewView.vue', import.meta.url), 'utf8')

    expect(overview).toMatch(/class="hero__lead page-lede page-summary"/)
    expect(overview).toMatch(/Explore events, compare evidence states, and inspect where the model does not travel/)
    expect(overview.indexOf('hero__actions')).toBeLessThan(overview.indexOf('metric-strip'))
    expect(overview.indexOf('metric-strip')).toBeLessThan(overview.indexOf('signal-panel'))
    expect(overview.indexOf('signal-panel')).toBeLessThan(overview.indexOf('hero__disclosure'))
    expect(overview).toMatch(/descriptive R², unitless \[0–1\],[\s\S]{0,120}not future-event accuracy/)
    expect(overview).toMatch(/restricted fine-grained records/)
    expect(overview).toMatch(/not human validation and do not prove that readers understand the science/)
  })

  it('gives the long-form routes stable heading-anchor navigation contracts', async () => {
    const routeViews = await Promise.all([
      readFile(new URL('../src/views/FindingsView.vue', import.meta.url), 'utf8'),
      readFile(new URL('../src/views/MethodsView.vue', import.meta.url), 'utf8'),
      readFile(new URL('../src/views/CreditsView.vue', import.meta.url), 'utf8'),
    ])

    for (const view of routeViews) {
      expect(view).toMatch(/class="page-lede page-summary"/)
      expect(view).toMatch(/<nav class="in-page-nav"/)
      expect(view).toMatch(/<ol class="content-section-nav">/)
      expect(view).toMatch(/@click\.prevent="navigateToSection"/)
    }
  })

  it('keeps the Methods workflow scannable without weakening private and public boundaries', async () => {
    const methods = await readFile(new URL('../src/views/MethodsView.vue', import.meta.url), 'utf8')

    expect(methods).toMatch(/Private inputs → processed signals → place-level model → admission → public artifact/)
    expect(methods.match(/<details class="method-timeline__details">/g)).toHaveLength(5)
    expect(methods.match(/class="method-timeline__number">0[1-5]/g)).toHaveLength(5)
    expect(methods.match(/<h3>[^<]+<\/h3>/g)).toHaveLength(5)
    expect(methods).toMatch(/Private boundary:[\s\S]*raw and fine-grained inputs/)
    expect(methods).toMatch(/Output retained privately:[\s\S]*temporal extracts and intermediate tables/)
    expect(methods).toMatch(/Public result:[\s\S]*aggregated model diagnostics/)
    expect(methods).toMatch(/Public result:[\s\S]*component states and admission band/)
    expect(methods).toMatch(/Published boundary:[\s\S]*aggregate-only static artifacts/)
    expect(methods).toMatch(/Missing assessment remains Not assessed; unavailable components remain unavailable rather than becoming zero/)
  })

  it('puts Credits trust facts before detailed authorship, rights, runtime, limits, and license sections', async () => {
    const credits = await readFile(new URL('../src/views/CreditsView.vue', import.meta.url), 'utf8')

    for (const fact of ['AGGREGATE-ONLY', 'LOCAL ASSETS', 'OPTIONAL LOCAL LOG', 'USER-ACTIVATED LINKS']) {
      expect(credits).toContain(fact)
    }
    expect(credits.indexOf('trust-facts-title')).toBeLessThan(credits.indexOf('credits-authorship-title'))
    for (const id of [
      'credits-authorship-title',
      'credits-sources-title',
      'credits-runtime-title',
      'credits-limits-title',
      'credits-license-title',
      'credits-public-limit-title',
    ]) {
      expect(credits).toContain(`href="#${id}"`)
      expect(credits).toContain(`id="${id}" tabindex="-1"`)
    }
    expect(credits).toMatch(/Qiushi Yu[\s\S]*Zhiyuan Zhao[\s\S]*ORNL EAGLE-I/)
    expect(credits).toMatch(/credits-authorship-title[\s\S]*<h3>Personal portfolio edition<\/h3>/)
    expect(credits).toMatch(/credits-sources-title[\s\S]*<h3>ORNL EAGLE-I outages<\/h3>/)
    expect(credits).toMatch(/do not establish real-user understanding/)
    expect(credits).toMatch(/LICENSE[\s\S]*CREDITS\.md[\s\S]*DATA_POLICY\.md[\s\S]*THIRD_PARTY_NOTICES\.md/)
  })
})
