// @vitest-environment happy-dom

import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick, onUpdated } from 'vue'
import { createMemoryHistory, createRouter } from 'vue-router'

import App from '../src/App.vue'
import { EVENTS } from '../src/content/study.js'
import { evidencePassportByEventId } from '../src/content/evidencePassportArtifact.js'
import { routes } from '../src/router/routes.js'
import AtlasView from '../src/views/AtlasView.vue'

const mountedWrappers = []
const TransitionHarness = {
  name: 'TransitionHarness',
  inheritAttrs: false,
  setup(_, { attrs, slots }) {
    onUpdated(() => attrs.onAfterEnter?.())
    return () => slots.default?.()
  },
}

async function settle() {
  await flushPromises()
  await nextTick()
  await new Promise((resolve) => window.setTimeout(resolve, 0))
  await flushPromises()
}

async function mountAtlas(initialPath = '/atlas') {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [{ path: '/atlas', name: 'atlas', component: AtlasView }],
  })
  await router.push(initialPath)
  await router.isReady()
  const wrapper = mount(AtlasView, { attachTo: document.body, global: { plugins: [router] } })
  mountedWrappers.push(wrapper)
  await settle()
  return { router, wrapper }
}

beforeEach(() => {
  window.sessionStorage.clear()
  Object.defineProperty(window, 'matchMedia', {
    configurable: true,
    value: vi.fn((query) => ({
      matches: false,
      media: query,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    })),
  })
  Object.defineProperty(window, 'requestAnimationFrame', {
    configurable: true,
    value: (callback) => {
      callback(0)
      return 1
    },
  })
  Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
    configurable: true,
    value: vi.fn(),
  })
})

afterEach(() => {
  for (const wrapper of mountedWrappers.splice(0)) wrapper.unmount()
  document.body.replaceChildren()
  vi.restoreAllMocks()
})

describe('mounted Atlas route and focus behavior', () => {
  it('hydrates invalid URL input and replaces it with the exact canonical query', async () => {
    const { router, wrapper } = await mountAtlas('/atlas?mode=unknown&hazard=unknown&event=unknown&view=table&extra=1')

    expect(router.currentRoute.value.query).toEqual({ mode: 'explore', event: EVENTS[0].id })
    expect(wrapper.get('#atlas-mode-explore').element.checked).toBe(true)
    expect(wrapper.get('#event-selection-title').text()).toBe(EVENTS[0].name)
  })

  it('uses replace for debounced search and push for discrete mode changes', async () => {
    const { router, wrapper } = await mountAtlas()
    const replaceSpy = vi.spyOn(router, 'replace')
    const pushSpy = vi.spyOn(router, 'push')

    await wrapper.get('input[name="atlas-query"]').setValue('Maria')
    await new Promise((resolve) => window.setTimeout(resolve, 225))
    await settle()

    expect(replaceSpy).toHaveBeenCalledWith(expect.objectContaining({
      name: 'atlas',
      query: expect.objectContaining({ mode: 'explore', q: 'Maria' }),
    }))

    await wrapper.get('#atlas-mode-compare').trigger('change')
    await settle()
    expect(pushSpy).toHaveBeenCalledWith(expect.objectContaining({
      name: 'atlas',
      query: expect.objectContaining({ mode: 'compare' }),
    }))
  })

  it('restores selection through back and forward navigation', async () => {
    const { router, wrapper } = await mountAtlas()
    const eventButtons = wrapper.findAll('.atlas-index__list button')
    await eventButtons[1].trigger('click')
    await settle()
    const secondId = router.currentRoute.value.query.event

    await wrapper.findAll('.atlas-index__list button')[2].trigger('click')
    await settle()
    const thirdId = router.currentRoute.value.query.event
    expect(thirdId).not.toBe(secondId)

    router.back()
    await settle()
    expect(router.currentRoute.value.query.event).toBe(secondId)
    expect(wrapper.get('#event-selection-title').text()).toBe(EVENTS.find(({ id }) => id === secondId).name)

    router.forward()
    await settle()
    expect(router.currentRoute.value.query.event).toBe(thirdId)
  })

  it('cross-seeds the selected Explore event into Compare without a second URL owner', async () => {
    const { router, wrapper } = await mountAtlas()
    await wrapper.findAll('.atlas-index__list button')[3].trigger('click')
    await settle()
    const selectedId = router.currentRoute.value.query.event

    await wrapper.get('#atlas-mode-compare').trigger('change')
    await settle()

    expect(router.currentRoute.value.query).toMatchObject({ mode: 'compare', a: selectedId })
    expect(router.currentRoute.value.query.b).not.toBe(selectedId)
    expect(wrapper.get('select[name="comparison-event-a"]').element.value).toBe(selectedId)
  })

  it('keeps in-page Evidence Passport focus inside Atlas without changing the route', async () => {
    const { router, wrapper } = await mountAtlas()
    const fullPath = router.currentRoute.value.fullPath

    await wrapper.get('a[href="#evidence-passport-title"]').trigger('click')
    await nextTick()

    expect(document.activeElement).toBe(wrapper.get('#evidence-passport-title').element)
    expect(router.currentRoute.value.fullPath).toBe(fullPath)
  })

  it('renders unassessed evidence as unavailable, never as zero or a worse outcome', async () => {
    const assessed = EVENTS.find(({ id }) => evidencePassportByEventId(id))
    const unassessed = EVENTS.find(({ id }) => !evidencePassportByEventId(id))
    const { wrapper } = await mountAtlas(`/atlas?mode=compare&a=${unassessed.id}&b=${assessed.id}`)
    const unavailable = wrapper.get('.comparison-unassessed')

    expect(unavailable.text()).toMatch(/Not assessed|unavailable/i)
    expect(unavailable.text()).toMatch(/not zero|missing reviewed public evidence/i)
    expect(unavailable.text()).not.toMatch(/worse recovery|score|rank/i)
  })

  it('leaves route-level H1 focus ownership in App after navigation', async () => {
    const router = createRouter({ history: createMemoryHistory(), routes })
    await router.push('/')
    await router.isReady()
    const wrapper = mount(App, {
      attachTo: document.body,
      global: { plugins: [router], stubs: { transition: TransitionHarness } },
    })
    mountedWrappers.push(wrapper)
    await settle()

    await router.push('/atlas')
    await settle()
    await settle()

    const heading = wrapper.get('h1[data-route-focus]').element
    expect(heading.textContent).toMatch(/Broad context.*bounded detail/is)
    expect(document.activeElement).toBe(heading)
  })
})
