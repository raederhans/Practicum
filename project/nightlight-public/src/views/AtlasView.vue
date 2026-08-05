<script setup>
import { computed, ref, watch } from 'vue'

import { DATA_BOUNDARY } from '../content/copy.js'
import {
  PUBLIC_EVIDENCE_PASSPORT_ARTIFACT,
  evidencePassportByEventId,
} from '../content/evidencePassportArtifact.js'
import { EVENTS, EVENT_TYPES } from '../content/study.js'
import { filterEvents } from '../domain/filterEvents.js'
import { projectPoint } from '../domain/projectPoint.js'
import { resolveSelectedId } from '../domain/resolveSelectedId.js'

const query = ref('')
const selectedType = ref('All')
const selectedId = ref(EVENTS[0].id)

const visibleEvents = computed(() => filterEvents(EVENTS, { type: selectedType.value, query: query.value }))
const selectedEvent = computed(() => visibleEvents.value.find((event) => event.id === selectedId.value) ?? null)
const selectedPassport = computed(() => evidencePassportByEventId(selectedEvent.value?.id))
const componentDefinitions = new Map(
  PUBLIC_EVIDENCE_PASSPORT_ARTIFACT.componentDefinitions.map((definition) => [definition.id, definition]),
)
const componentStatusLabels = Object.freeze({
  available: 'Available',
  limited: 'Limited',
  unavailable: 'Unavailable',
})

function componentDefinition(componentId) {
  return componentDefinitions.get(componentId)
}

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

    <section
      v-if="selectedEvent"
      class="evidence-passport"
      aria-labelledby="evidence-passport-title"
      aria-live="polite"
    >
      <header class="evidence-passport__header">
        <div>
          <p class="eyebrow"><span>Admission note</span> Evidence passport</p>
          <h2 id="evidence-passport-title">{{ selectedEvent.name }} · {{ selectedEvent.location }}</h2>
        </div>
        <p>
          This is an analysis admission heuristic. It describes whether the project can inspect this event with its
          current evidence—not how well a community recovered.
        </p>
      </header>

      <template v-if="selectedPassport">
        <div class="evidence-passport__status">
          <span :class="`passport-band passport-band--${selectedPassport.readinessBand}`">
            {{ selectedPassport.readinessLabel }}
          </span>
          <p>{{ PUBLIC_EVIDENCE_PASSPORT_ARTIFACT.bandDefinitions.find(({ id }) => id === selectedPassport.readinessBand)?.meaning }}</p>
        </div>

        <div class="evidence-passport__table-wrap">
          <table class="evidence-table passport-components">
            <caption>Reviewed component-level evidence; the project does not display or interpret an overall score</caption>
            <thead>
              <tr>
                <th scope="col">Component</th>
                <th scope="col">Points</th>
                <th scope="col">State</th>
                <th scope="col">What it checks</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="component in selectedPassport.components" :key="component.id">
                <th scope="row">{{ componentDefinition(component.id)?.label }}</th>
                <td><strong>{{ component.points }}</strong> / {{ component.maxPoints }}</td>
                <td>
                  <span :class="`component-state component-state--${component.status}`">
                    {{ componentStatusLabels[component.status] }}
                  </span>
                </td>
                <td>{{ componentDefinition(component.id)?.meaning }}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="evidence-passport__claims">
          <article>
            <h3>Supported claim</h3>
            <p>{{ selectedPassport.supportedClaim }}</p>
          </article>
          <article>
            <h3>Unsupported claim</h3>
            <p>{{ selectedPassport.unsupportedClaim }}</p>
          </article>
        </div>

        <footer class="evidence-passport__source">
          <span>Artifact {{ PUBLIC_EVIDENCE_PASSPORT_ARTIFACT.version }} · {{ selectedPassport.publicationStatus }}</span>
          <span>Source SHA-256 <code>{{ PUBLIC_EVIDENCE_PASSPORT_ARTIFACT.source.sha256 }}</code></span>
          <span>{{ PUBLIC_EVIDENCE_PASSPORT_ARTIFACT.source.attribution }}</span>
        </footer>
      </template>

      <div v-else class="evidence-passport__empty">
        <span class="passport-band passport-band--unassessed">Not assessed in v1</span>
        <div>
          <h3>No reviewed Evidence Passport exists for this event.</h3>
          <p>
            Missing assessment is not evidence of poor data or worse recovery. This event remains in the broad public
            index, while its component evidence stays unclaimed until a reviewed source is admitted.
          </p>
        </div>
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
