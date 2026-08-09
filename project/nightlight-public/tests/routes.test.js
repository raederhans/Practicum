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
})
