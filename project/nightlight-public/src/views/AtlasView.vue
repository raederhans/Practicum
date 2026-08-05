<script setup>
import { computed, ref, watch } from 'vue'

import { DATA_BOUNDARY } from '../content/copy.js'
import { EVENTS, EVENT_TYPES } from '../content/study.js'
import { filterEvents } from '../domain/filterEvents.js'
import { projectPoint } from '../domain/projectPoint.js'
import { resolveSelectedId } from '../domain/resolveSelectedId.js'

const query = ref('')
const selectedType = ref('All')
const selectedId = ref(EVENTS[0].id)

const visibleEvents = computed(() => filterEvents(EVENTS, { type: selectedType.value, query: query.value }))
const selectedEvent = computed(() => visibleEvents.value.find((event) => event.id === selectedId.value) ?? null)

watch(visibleEvents, (events) => {
  selectedId.value = resolveSelectedId(events, selectedId.value)
}, { flush: 'sync' })
</script>

<template>
  <article class="page atlas-page">
    <header class="page-heading page-heading--split">
      <div>
        <p class="eyebrow"><span>Field note 02</span> Study atlas</p>
        <h1>Broad context,<br><em>bounded detail.</em></h1>
      </div>
      <p>
        Explore a hand-cleaned event index using names, years, broad locations, and centers rounded to one decimal.
        This is orientation—not a release of the analytical layers.
      </p>
    </header>

    <section class="atlas-controls" aria-label="Filter event index">
      <label>
        <span>Search the index</span>
        <input v-model="query" type="search" placeholder="Event, place, or year" autocomplete="off">
      </label>
      <label>
        <span>Hazard family</span>
        <select v-model="selectedType">
          <option v-for="type in EVENT_TYPES" :key="type" :value="type">{{ type }}</option>
        </select>
      </label>
      <p><strong>{{ visibleEvents.length }}</strong> / {{ EVENTS.length }} broad references</p>
    </section>

    <section class="atlas-workbench">
      <div class="atlas-map">
        <div class="atlas-map__legend">
          <span><i></i> broad event center</span>
          <span>Local SVG · no tiles</span>
        </div>
        <svg viewBox="0 0 960 540" role="img" aria-labelledby="atlas-title atlas-desc">
          <title id="atlas-title">Broad event context atlas</title>
          <desc id="atlas-desc">A schematic coordinate field with selectable event-level points. Fine-grained layers are not present.</desc>
          <defs>
            <pattern id="atlas-grid" width="72" height="72" patternUnits="userSpaceOnUse">
              <path d="M72 0H0V72" />
            </pattern>
          </defs>
          <rect class="atlas-field" x="20" y="20" width="920" height="500" rx="5" />
          <rect class="atlas-grid" x="20" y="20" width="920" height="500" rx="5" />
          <path class="atlas-land" d="M174 178l44-44 86-31 94 14 76-17 95 24 58 44 107 34 64 67-27 43-65 10-47 73-69 32-82-11-65 31-84-42-93-14-56-68-79-31 13-65z" />
          <path class="atlas-coastline" d="M174 178l44-44 86-31 94 14 76-17 95 24 58 44 107 34 64 67-27 43-65 10-47 73-69 32-82-11-65 31-84-42-93-14-56-68-79-31 13-65z" />
          <g
            v-for="event in visibleEvents"
            :key="event.id"
            class="atlas-point"
            :class="{ 'atlas-point--selected': selectedEvent?.id === event.id }"
            :transform="`translate(${projectPoint(event.center)[0]} ${projectPoint(event.center)[1]})`"
            aria-hidden="true"
          >
            <circle class="atlas-point__halo" r="14" />
            <circle class="atlas-point__dot" r="5" />
          </g>
          <g class="atlas-axis" aria-hidden="true">
            <text x="48" y="514">170°W</text>
            <text x="856" y="514">45°E</text>
            <text x="27" y="54">72°N</text>
            <text x="27" y="489">16°N</text>
          </g>
        </svg>

        <div v-if="selectedEvent" class="atlas-readout" aria-live="polite">
          <span>{{ selectedEvent.type }} / {{ selectedEvent.year }}</span>
          <strong>{{ selectedEvent.name }}</strong>
          <p>{{ selectedEvent.location }} · {{ Math.abs(selectedEvent.center[1]).toFixed(1) }}°{{ selectedEvent.center[1] >= 0 ? 'N' : 'S' }}, {{ Math.abs(selectedEvent.center[0]).toFixed(1) }}°{{ selectedEvent.center[0] >= 0 ? 'E' : 'W' }}</p>
        </div>
      </div>

      <div class="atlas-index" aria-label="Event index">
        <div class="atlas-index__header">
          <span>Index / public</span>
          <span>{{ visibleEvents.length.toString().padStart(2, '0') }}</span>
        </div>
        <div v-if="visibleEvents.length" class="atlas-index__list">
          <button
            v-for="(event, index) in visibleEvents"
            :key="event.id"
            type="button"
            :class="{ 'is-selected': selectedEvent?.id === event.id }"
            :aria-pressed="selectedEvent?.id === event.id"
            @click="selectedId = event.id"
          >
            <span>{{ (index + 1).toString().padStart(2, '0') }}</span>
            <span><strong>{{ event.name }}</strong><small>{{ event.location }}</small></span>
            <time :datetime="String(event.year)">{{ event.year }}</time>
          </button>
        </div>
        <p v-else class="atlas-index__empty">No broad event references match these filters.</p>
      </div>
    </section>

    <aside class="withheld-panel">
      <span class="withheld-panel__stamp">WITHHELD</span>
      <div>
        <h2>Fine-grained layers are intentionally absent.</h2>
        <p>{{ DATA_BOUNDARY.status }}</p>
      </div>
      <ul>
        <li v-for="item in DATA_BOUNDARY.excluded" :key="item">{{ item }}</li>
      </ul>
    </aside>
  </article>
</template>
