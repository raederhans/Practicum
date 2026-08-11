<script setup>
import { onUnmounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'

import { localResearchLog } from '../lib/localResearchAnalytics.js'

const route = useRoute()
const state = ref(localResearchLog.snapshot())

const unsubscribe = localResearchLog.subscribe((nextState) => {
  state.value = nextState
})

watch(
  () => route.name,
  (routeName) => {
    if (typeof routeName === 'string') localResearchLog.recordSurfaceViewed(routeName)
  },
  { immediate: true },
)

onUnmounted(unsubscribe)

function optIn() {
  localResearchLog.grantConsent()
  if (typeof route.name === 'string') localResearchLog.recordSurfaceViewed(route.name)
}

function clearEvents() {
  localResearchLog.clearEvents()
}

function stopAndClear() {
  localResearchLog.withdrawConsent()
}

function exportEvents() {
  const snapshot = localResearchLog.exportSnapshot()
  if (!snapshot) return

  const contents = `${JSON.stringify(snapshot, null, 2)}\n`
  const objectUrl = URL.createObjectURL(new Blob([contents], { type: 'application/json' }))
  const anchor = document.createElement('a')
  anchor.href = objectUrl
  anchor.download = 'nightlight-local-research-log.json'
  anchor.hidden = true
  document.body.append(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(objectUrl)
}
</script>

<template>
  <section class="local-research-log" aria-labelledby="local-research-log-title">
    <details>
      <summary>
        <span id="local-research-log-title">Optional local research log</span>
        <strong>{{ state.consent === 'granted' ? `On · ${state.count} events` : 'Off by default' }}</strong>
      </summary>
      <div class="local-research-log__panel">
        <p>
          This tab-only log helps inspect which fixed evidence pages and Atlas mode are used in a voluntary research session.
          It never sends data, sets cookies, creates a persistent identifier, or records free text, model inputs, device details, or location.
        </p>

        <template v-if="state.consent !== 'granted'">
          <p>Nothing is recorded until you opt in. Closing the tab also ends the session.</p>
          <button type="button" @click="optIn">Opt in for this tab</button>
        </template>

        <template v-else>
          <p role="status" aria-live="polite">
            Local logging is on for consent version {{ state.consentVersion }}. {{ state.count }} allowlisted events are stored in this tab.
          </p>
          <ol v-if="state.events.length" class="local-research-log__events">
            <li v-for="event in state.events" :key="event.ordinal">
              <span>#{{ event.ordinal }}</span>
              <strong>{{ event.name }}</strong>
              <small>{{ Object.values(event.properties).join(' · ') }}</small>
            </li>
          </ol>
          <p v-else>No local research events recorded yet.</p>
          <div class="local-research-log__actions">
            <button type="button" @click="exportEvents">Export JSON</button>
            <button type="button" @click="clearEvents">Clear events</button>
            <button type="button" @click="stopAndClear">Stop and clear</button>
          </div>
        </template>
      </div>
    </details>
  </section>
</template>
