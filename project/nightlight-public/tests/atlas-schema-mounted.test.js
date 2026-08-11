// @vitest-environment happy-dom

import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'
import { createMemoryHistory, createRouter } from 'vue-router'

vi.mock('../src/content/evidencePassportArtifact.js', async (importOriginal) => {
  const actual = await importOriginal()
  const originalLookup = actual.evidencePassportByEventId
  return {
    ...actual,
    evidencePassportByEventId(eventId) {
      const passport = originalLookup(eventId)
      if (eventId !== 'ian-charlotte' || !passport) return passport
      return { ...passport, components: passport.components.slice(1) }
    },
  }
})

const mountedWrappers = []

beforeEach(() => {
  Object.defineProperty(window, 'matchMedia', {
    configurable: true,
    value: vi.fn(() => ({ matches: false, addEventListener: vi.fn(), removeEventListener: vi.fn() })),
  })
})

afterEach(() => {
  for (const wrapper of mountedWrappers.splice(0)) wrapper.unmount()
  document.body.replaceChildren()
  vi.restoreAllMocks()
})

describe('mounted Atlas schema failure surface', () => {
  it('withholds component pairing when a mounted Passport does not match the reviewed schema', async () => {
    const { default: AtlasView } = await import('../src/views/AtlasView.vue')
    const router = createRouter({
      history: createMemoryHistory(),
      routes: [{ path: '/atlas', name: 'atlas', component: AtlasView }],
    })
    await router.push('/atlas?mode=compare&a=ian-charlotte&b=ian-fortmyers')
    await router.isReady()
    const wrapper = mount(AtlasView, { attachTo: document.body, global: { plugins: [router] } })
    mountedWrappers.push(wrapper)
    await flushPromises()
    await nextTick()

    const failure = wrapper.get('#comparison-schema-title').element.closest('section')
    expect(failure.textContent).toMatch(/Schema not comparable/i)
    expect(failure.textContent).toMatch(/pairing was withheld/i)
    expect(failure.textContent).toMatch(/not counted as differences/i)
  })
})
