<template>
  <div id="layout" :class="{ 'no-earth-bg': isHomePage }">
    <NavBar />
    <main class="main-content">
      <RouterView v-slot="{ Component, route }">
        <Transition name="page">
          <component :is="Component" :key="route.path" />
        </Transition>
      </RouterView>
    </main>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import NavBar from '@/components/NavBar.vue'

const route = useRoute()
const isHomePage = computed(() => route.path === '/')
</script>

<style>
#layout {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background:
    radial-gradient(circle at 72% 22%, rgba(0,212,255,0.12), transparent 34%),
    radial-gradient(circle at 18% 78%, rgba(0,229,160,0.08), transparent 30%),
    linear-gradient(180deg, #061424 0%, #030d1a 100%);
}
#layout.no-earth-bg {
  background: var(--bg);
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding-top: var(--nav-h);
  position: relative;
}

/* Page transition — synchronized fade for bg + content */
.page-enter-active {
  transition: opacity 0.4s ease;
}
.page-leave-active {
  transition: opacity 0.4s ease;
}
.page-enter-from {
  opacity: 0;
}
.page-leave-to {
  opacity: 0;
}
</style>
