<script setup>
import { nextTick, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'

const navigation = [
  { to: '/', label: 'Overview', index: '01' },
  { to: '/atlas', label: 'Study Atlas', index: '02' },
  { to: '/findings', label: 'Findings', index: '03' },
  { to: '/methods', label: 'Methods', index: '04' },
  { to: '/credits', label: 'Credits / Policy', index: '05' },
]

const route = useRoute()
const navigationElement = ref(null)
const navigationToggleElement = ref(null)
const mainElement = ref(null)
const isMobileNavigationOpen = ref(false)
const siteTitle = 'Nightlight Disaster Observatory'
let enteredRoutePath = route.path

function closeMobileNavigation() {
  isMobileNavigationOpen.value = false
}

function toggleMobileNavigation() {
  isMobileNavigationOpen.value = !isMobileNavigationOpen.value
}

function handleNavigationEscape(event) {
  if (!isMobileNavigationOpen.value) return

  event.stopPropagation()
  closeMobileNavigation()
  nextTick(() => navigationToggleElement.value?.focus({ preventScroll: true }))
}

function focusMainContent() {
  const main = mainElement.value
  if (!main) return

  main.focus({ preventScroll: true })
  main.scrollIntoView({ block: 'start' })
}

function revealActiveNavigation() {
  nextTick(() => {
    requestAnimationFrame(() => {
      const navigation = navigationElement.value
      const activeItem = navigation?.querySelector('[aria-current="page"]')
      if (!navigation || !activeItem) return

      const centeredLeft = activeItem.offsetLeft - ((navigation.clientWidth - activeItem.offsetWidth) / 2)
      const maxScrollLeft = Math.max(0, navigation.scrollWidth - navigation.clientWidth)
      navigation.scrollLeft = Math.max(0, Math.min(centeredLeft, maxScrollLeft))
    })
  })
}

function updateRouteContext() {
  document.title = route.meta.pageTitle ? `${route.meta.pageTitle} | ${siteTitle}` : siteTitle
  revealActiveNavigation()
}

function focusRouteHeading() {
  const heading = mainElement.value?.querySelector('[data-route-focus]')
  if (!heading) return

  heading.classList.add('focus-target--route')
  heading.focus({ preventScroll: true })
}

function handleRouteEnter() {
  updateRouteContext()
  if (route.path !== enteredRoutePath) {
    enteredRoutePath = route.path
    focusRouteHeading()
  }
}

watch(() => route.path, updateRouteContext, { flush: 'post' })
watch(() => route.fullPath, () => closeMobileNavigation(), { flush: 'post' })
onMounted(updateRouteContext)
</script>

<template>
  <a class="skip-link" href="#main-content" @click.prevent="focusMainContent">Skip to research content</a>
  <div class="site-shell">
    <header class="site-header" @keydown.esc="handleNavigationEscape">
      <RouterLink class="identity" to="/" aria-label="Nightlight Disaster Observatory overview">
        <svg class="identity__mark" viewBox="0 0 48 48" aria-hidden="true">
          <circle cx="24" cy="24" r="17" />
          <path d="M7 28h10l4-12 6 22 4-13h10" />
          <circle class="identity__spark" cx="24" cy="24" r="2.5" />
        </svg>
        <span>
          <strong>Nightlight</strong>
          <small>Disaster Observatory</small>
        </span>
      </RouterLink>

      <button
        ref="navigationToggleElement"
        class="mobile-nav-toggle"
        type="button"
        :aria-expanded="isMobileNavigationOpen ? 'true' : 'false'"
        aria-controls="primary-navigation"
        @click="toggleMobileNavigation"
      >
        {{ isMobileNavigationOpen ? 'Close menu' : 'Menu' }}
      </button>

      <nav
        id="primary-navigation"
        ref="navigationElement"
        class="site-nav"
        :class="{ 'site-nav--open': isMobileNavigationOpen }"
        aria-label="Primary navigation"
      >
        <RouterLink v-for="item in navigation" :key="item.to" :to="item.to" :aria-current="route.path === item.to ? 'page' : undefined">
          <span aria-hidden="true">{{ item.index }}</span>{{ item.label }}
        </RouterLink>
      </nav>

      <div class="header-status" aria-label="Public edition status">
        <span class="status-light" aria-hidden="true"></span>
        Public / aggregate-only
      </div>
    </header>

    <main id="main-content" ref="mainElement" tabindex="-1">
      <RouterView v-slot="{ Component }">
        <Transition name="route" mode="out-in" @after-enter="handleRouteEnter">
          <component :is="Component" />
        </Transition>
      </RouterView>
    </main>

    <footer class="site-footer">
      <p>Independent student research portfolio · 2026</p>
      <p>Local assets only · No analytics · No external requests</p>
    </footer>
  </div>
</template>
