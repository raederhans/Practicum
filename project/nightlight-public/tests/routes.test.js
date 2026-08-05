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
})
