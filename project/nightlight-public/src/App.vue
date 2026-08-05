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

function revealActiveNavigation() {
  nextTick(() => {
    requestAnimationFrame(() => {
      navigationElement.value
        ?.querySelector('[aria-current="page"]')
        ?.scrollIntoView({ block: 'nearest', inline: 'center' })
    })
  })
}

watch(() => route.path, revealActiveNavigation, { flush: 'post' })
onMounted(revealActiveNavigation)
</script>

<template>
  <a class="skip-link" href="#main-content">Skip to research content</a>
  <div class="site-shell">
    <header class="site-header">
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

      <nav ref="navigationElement" class="site-nav" aria-label="Primary navigation">
        <RouterLink v-for="item in navigation" :key="item.to" :to="item.to" :aria-current="route.path === item.to ? 'page' : undefined">
          <span aria-hidden="true">{{ item.index }}</span>{{ item.label }}
        </RouterLink>
      </nav>

      <div class="header-status" aria-label="Public edition status">
        <span class="status-light" aria-hidden="true"></span>
        Public / aggregate-only
      </div>
    </header>

    <main id="main-content" tabindex="-1">
      <RouterView v-slot="{ Component }">
        <Transition name="route" mode="out-in">
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
