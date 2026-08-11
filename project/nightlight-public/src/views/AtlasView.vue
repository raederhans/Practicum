<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { DATA_BOUNDARY } from '../content/copy.js'
import {
  PUBLIC_EVIDENCE_PASSPORT_ARTIFACT,
  evidencePassportByEventId,
} from '../content/evidencePassportArtifact.js'
import { EVENTS, HAZARD_FAMILIES } from '../content/study.js'
import {
  PRESET_COMPARISONS,
  PRESET_DISCLAIMER,
  buildEventComparison,
  resolveComparisonPeerId,
} from '../domain/compareEvents.js'
import { filterEvents } from '../domain/filterEvents.js'
import { projectPoint } from '../domain/projectPoint.js'
import { resolveSelectedId } from '../domain/resolveSelectedId.js'

const route = useRoute()
const router = useRouter()

const VIEW_MODES = Object.freeze(['explore', 'compare'])
const MOBILE_VIEWS = Object.freeze(['list', 'map'])
const EVENT_IDS = new Set(EVENTS.map(({ id }) => id))
const PRESET_BY_ID = new Map(PRESET_COMPARISONS.map((preset) => [preset.id, preset]))
const DEFAULT_LEFT_ID = EVENTS[0].id
const DEFAULT_RIGHT_ID = resolveComparisonPeerId(EVENTS, DEFAULT_LEFT_ID, DEFAULT_LEFT_ID)
const SEARCH_DEBOUNCE_MS = 200

const viewMode = ref('explore')
const query = ref('')
const selectedHazardFamily = ref('All')
const selectedId = ref(DEFAULT_LEFT_ID)
const comparisonLeftId = ref(DEFAULT_LEFT_ID)
const comparisonRightId = ref(DEFAULT_RIGHT_ID)
const selectedPresetId = ref(null)
const atlasMobileView = ref('list')
const isMobileViewport = ref(false)

let searchTimer = null
let mobileMediaQuery = null

const visibleEvents = computed(() => filterEvents(EVENTS, { hazardFamily: selectedHazardFamily.value, query: query.value }))
const selectedEvent = computed(() => visibleEvents.value.find((event) => event.id === selectedId.value) ?? null)
const selectedPassport = computed(() => evidencePassportByEventId(selectedEvent.value?.id))
const comparisonLeftEvent = computed(() => EVENTS.find(({ id }) => id === comparisonLeftId.value) ?? null)
const comparisonRightEvent = computed(() => EVENTS.find(({ id }) => id === comparisonRightId.value) ?? null)
const comparisonLeftPassport = computed(() => evidencePassportByEventId(comparisonLeftEvent.value?.id))
const comparisonRightPassport = computed(() => evidencePassportByEventId(comparisonRightEvent.value?.id))
const comparison = computed(() => {
  if (!comparisonLeftEvent.value || !comparisonRightEvent.value) return null
  if (comparisonLeftEvent.value.id === comparisonRightEvent.value.id) return null
  return buildEventComparison(
    comparisonLeftEvent.value,
    comparisonRightEvent.value,
    comparisonLeftPassport.value,
    comparisonRightPassport.value,
    PUBLIC_EVIDENCE_PASSPORT_ARTIFACT,
  )
})
const eventGroups = Object.freeze(
  HAZARD_FAMILIES.filter((hazardFamily) => hazardFamily !== 'All').map((hazardFamily) => Object.freeze({
    hazardFamily,
    events: Object.freeze(EVENTS.filter((event) => event.hazardFamily === hazardFamily)),
  })),
)
const activePresetId = computed(() => selectedPresetId.value)
const activePreset = computed(() => PRESET_BY_ID.get(activePresetId.value) ?? null)
const firstComponentPair = computed(() => comparison.value?.componentPairs[0] ?? null)
const showEventList = computed(() => !isMobileViewport.value || atlasMobileView.value === 'list')
const showAtlasMap = computed(() => !isMobileViewport.value || atlasMobileView.value === 'map')
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

function passportLabel(eventId) {
  return evidencePassportByEventId(eventId)?.readinessLabel ?? 'Not assessed in v1'
}

function eventOptionLabel(event) {
  return `${event.name} — ${event.location} (${event.year}) — ${passportLabel(event.id)}`
}

function formatSummaryValue(summary) {
  if (summary.value === null) return '—'
  return `${summary.prefix}${summary.value}${summary.maximum ? ` / ${summary.maximum}` : ''}`
}

function navigateToSection(event) {
  const target = document.getElementById(event.currentTarget.hash.slice(1))
  if (!(target instanceof HTMLElement)) return
  target.scrollIntoView({ block: 'start' })
  target.focus({ preventScroll: true })
}

function scalarQueryValue(value) {
  return typeof value === 'string' ? value : ''
}

function validOption(value, options, fallback) {
  return options.includes(value) ? value : fallback
}

function validEventId(value) {
  return EVENT_IDS.has(value) ? value : null
}

function normalizedSearch(value) {
  return scalarQueryValue(value).slice(0, 80).trim()
}

function clearSearchTimer() {
  if (searchTimer === null) return
  window.clearTimeout(searchTimer)
  searchTimer = null
}

function atlasQuery() {
  const nextQuery = { mode: viewMode.value }

  if (viewMode.value === 'explore') {
    const search = normalizedSearch(query.value)
    if (search) nextQuery.q = search
    if (selectedHazardFamily.value !== 'All') nextQuery.hazard = selectedHazardFamily.value
    if (selectedId.value && EVENT_IDS.has(selectedId.value)) nextQuery.event = selectedId.value
    if (atlasMobileView.value !== 'list') nextQuery.view = atlasMobileView.value
    return nextQuery
  }

  nextQuery.a = comparisonLeftId.value
  nextQuery.b = comparisonRightId.value
  const preset = PRESET_BY_ID.get(selectedPresetId.value)
  if (
    preset
    && preset.eventIds[0] === comparisonLeftId.value
    && preset.eventIds[1] === comparisonRightId.value
  ) {
    nextQuery.preset = preset.id
  }
  return nextQuery
}

function queryMatchesRoute(expectedQuery, routeQuery) {
  const routeKeys = Object.keys(routeQuery).sort()
  const expectedKeys = Object.keys(expectedQuery).sort()
  if (routeKeys.length !== expectedKeys.length) return false
  return expectedKeys.every((key, index) => (
    key === routeKeys[index] && scalarQueryValue(routeQuery[key]) === expectedQuery[key]
  ))
}

function navigateWithState(method = 'push') {
  clearSearchTimer()
  const location = { name: 'atlas', query: atlasQuery() }
  if (method === 'replace') {
    void router.replace(location)
    return
  }
  void router.push(location)
}

function hydrateFromRoute(routeQuery) {
  clearSearchTimer()
  const nextMode = validOption(scalarQueryValue(routeQuery.mode), VIEW_MODES, 'explore')
  const nextSearch = normalizedSearch(routeQuery.q)
  const nextHazard = validOption(scalarQueryValue(routeQuery.hazard), HAZARD_FAMILIES, 'All')
  const nextMobileView = validOption(scalarQueryValue(routeQuery.view), MOBILE_VIEWS, 'list')
  const nextVisibleEvents = filterEvents(EVENTS, { hazardFamily: nextHazard, query: nextSearch })
  const requestedSelectedId = validEventId(scalarQueryValue(routeQuery.event))
  const nextSelectedId = resolveSelectedId(nextVisibleEvents, requestedSelectedId)

  const requestedPreset = PRESET_BY_ID.get(scalarQueryValue(routeQuery.preset)) ?? null
  let nextLeftId = validEventId(scalarQueryValue(routeQuery.a))
  let nextRightId = validEventId(scalarQueryValue(routeQuery.b))
  if (requestedPreset && (!nextLeftId || !nextRightId)) {
    [nextLeftId, nextRightId] = requestedPreset.eventIds
  }
  nextLeftId ??= DEFAULT_LEFT_ID
  nextRightId = resolveComparisonPeerId(EVENTS, nextLeftId, nextRightId)
  const nextPresetId = requestedPreset
    && requestedPreset.eventIds[0] === nextLeftId
    && requestedPreset.eventIds[1] === nextRightId
    ? requestedPreset.id
    : null

  viewMode.value = nextMode
  if (nextMode === 'explore') {
    query.value = nextSearch
    selectedHazardFamily.value = nextHazard
    selectedId.value = nextSelectedId
    atlasMobileView.value = nextMobileView
  } else {
    comparisonLeftId.value = nextLeftId
    comparisonRightId.value = nextRightId
    selectedPresetId.value = nextPresetId
  }

  const canonicalQuery = atlasQuery()
  if (!queryMatchesRoute(canonicalQuery, routeQuery)) {
    void router.replace({ name: 'atlas', query: canonicalQuery })
  }
}

function updateViewMode(nextMode) {
  if (!VIEW_MODES.includes(nextMode) || nextMode === viewMode.value) return
  viewMode.value = nextMode
  navigateWithState()
}

function updateSearch(event) {
  query.value = event.currentTarget.value.slice(0, 80)
  selectedId.value = resolveSelectedId(visibleEvents.value, selectedId.value)
  clearSearchTimer()
  searchTimer = window.setTimeout(() => {
    searchTimer = null
    navigateWithState('replace')
  }, SEARCH_DEBOUNCE_MS)
}

function updateHazardFamily(event) {
  selectedHazardFamily.value = validOption(event.currentTarget.value, HAZARD_FAMILIES, 'All')
  selectedId.value = resolveSelectedId(visibleEvents.value, selectedId.value)
  navigateWithState()
}

function selectEvent(eventId) {
  if (!visibleEvents.value.some(({ id }) => id === eventId)) return
  selectedId.value = eventId
  comparisonLeftId.value = eventId
  comparisonRightId.value = resolveComparisonPeerId(EVENTS, eventId, comparisonRightId.value)
  selectedPresetId.value = null
  navigateWithState()
}

function clearFilters() {
  query.value = ''
  selectedHazardFamily.value = 'All'
  selectedId.value = resolveSelectedId(EVENTS, selectedId.value)
  navigateWithState()
}

function updateMobileView(nextView) {
  if (!MOBILE_VIEWS.includes(nextView) || nextView === atlasMobileView.value) return
  atlasMobileView.value = nextView
  navigateWithState()
}

function applyPreset(preset) {
  viewMode.value = 'compare'
  comparisonLeftId.value = preset.eventIds[0]
  comparisonRightId.value = preset.eventIds[1]
  selectedPresetId.value = preset.id
  navigateWithState()
}

function swapComparisonEvents() {
  const previousLeftId = comparisonLeftId.value
  comparisonLeftId.value = comparisonRightId.value
  comparisonRightId.value = previousLeftId
  selectedPresetId.value = null
  navigateWithState()
}

function updateComparisonLeft(event) {
  const nextLeftId = validEventId(event.currentTarget.value) ?? DEFAULT_LEFT_ID
  comparisonLeftId.value = nextLeftId
  comparisonRightId.value = resolveComparisonPeerId(EVENTS, nextLeftId, comparisonRightId.value)
  selectedPresetId.value = null
  navigateWithState()
}

function updateComparisonRight(event) {
  const nextRightId = validEventId(event.currentTarget.value)
  comparisonRightId.value = resolveComparisonPeerId(EVENTS, comparisonLeftId.value, nextRightId)
  selectedPresetId.value = null
  navigateWithState()
}

const comparisonLiveSummary = computed(() => {
  if (!comparison.value) return 'Choose two different events to compare.'
  if (comparison.value.schemaStatus !== 'paired-v1') {
    return `Comparison updated. ${comparison.value.compatibility.label}. ${comparison.value.passportCoverage} of 2 reviewed Passports; component pairing is unavailable.`
  }
  return `Comparison updated. ${comparison.value.compatibility.label}. Review the interpretation limits and each v1 component row separately. No similarity score is computed.`
})

function syncMobileViewport(event) {
  isMobileViewport.value = event.matches
}

watch(() => route.query, hydrateFromRoute, { deep: true, immediate: true })

onMounted(() => {
  mobileMediaQuery = window.matchMedia('(max-width: 899px)')
  syncMobileViewport(mobileMediaQuery)
  mobileMediaQuery.addEventListener('change', syncMobileViewport)
})

onUnmounted(() => {
  clearSearchTimer()
  mobileMediaQuery?.removeEventListener('change', syncMobileViewport)
})
</script>

<template>
  <article class="page atlas-page">
    <header class="page-heading page-heading--split">
      <div>
        <p class="eyebrow"><span>Field note 02</span> Study atlas</p>
        <h1 class="focus-target" data-route-focus tabindex="-1">Broad context,<br><em>bounded detail.</em></h1>
      </div>
      <p>
        Explore a hand-cleaned event index using names, years, broad locations, and centers rounded to one decimal.
        This is orientation—not a release of the analytical layers.
      </p>
    </header>

    <fieldset class="atlas-view-mode" aria-describedby="atlas-view-help">
      <legend>Evidence view</legend>
      <label :class="{ 'is-active': viewMode === 'explore' }">
        <input id="atlas-mode-explore" :checked="viewMode === 'explore'" type="radio" name="atlas-evidence-view" value="explore" aria-controls="atlas-explore-panel" @change="updateViewMode('explore')">
        <span><strong>Explore one event</strong><small>Map, index, and one Evidence Passport</small></span>
      </label>
      <label :class="{ 'is-active': viewMode === 'compare' }">
        <input id="atlas-mode-compare" :checked="viewMode === 'compare'" type="radio" name="atlas-evidence-view" value="compare" aria-controls="atlas-compare-panel" @change="updateViewMode('compare')">
        <span><strong>Compare events</strong><small>Category-first evidence inspection</small></span>
      </label>
    </fieldset>

    <div v-if="viewMode === 'explore'" id="atlas-explore-panel" class="atlas-mode-panel">
      <section class="atlas-controls atlas-task-summary" aria-label="Filter event index">
        <label>
          <span>Search the index</span>
          <input :value="query" type="search" name="atlas-query" placeholder="Event, place, or year…" autocomplete="off" @input="updateSearch">
        </label>
        <label>
          <span>Hazard family</span>
          <select :value="selectedHazardFamily" name="atlas-hazard" @change="updateHazardFamily">
            <option v-for="hazardFamily in HAZARD_FAMILIES" :key="hazardFamily" :value="hazardFamily">{{ hazardFamily }}</option>
          </select>
        </label>
        <p><strong>{{ visibleEvents.length }}</strong> / {{ EVENTS.length }} broad references</p>
      </section>

      <section
        v-if="selectedEvent"
        class="event-selection-summary"
        aria-labelledby="event-selection-title"
        aria-live="polite"
        aria-atomic="true"
      >
        <div>
          <span>Selected public reference</span>
          <h2 id="event-selection-title">{{ selectedEvent.name }}</h2>
          <p>{{ selectedEvent.location }} · {{ selectedEvent.year }} · {{ passportLabel(selectedEvent.id) }}</p>
        </div>
        <a href="#evidence-passport-title" @click.prevent="navigateToSection">View Evidence Passport</a>
      </section>

      <section v-else class="event-selection-summary" aria-labelledby="event-selection-empty-title">
        <div>
          <span>Selected public reference</span>
          <h2 id="event-selection-empty-title">No matching event</h2>
          <p>Clear the current filters to restore the broad public index.</p>
        </div>
        <button type="button" class="atlas-clear-filters" @click="clearFilters">Clear filters</button>
      </section>

      <fieldset class="atlas-mobile-view" aria-label="Choose the mobile atlas view">
        <legend>List / Map</legend>
        <button type="button" :aria-pressed="atlasMobileView === 'list'" aria-controls="atlas-event-list" @click="updateMobileView('list')">List</button>
        <button type="button" :aria-pressed="atlasMobileView === 'map'" aria-controls="atlas-event-map" @click="updateMobileView('map')">Map</button>
      </fieldset>

      <section class="atlas-workbench">
      <div
        id="atlas-event-list"
        v-show="showEventList"
        class="atlas-index"
        :aria-hidden="isMobileViewport && !showEventList ? 'true' : undefined"
        aria-label="Event index"
      >
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
            @click="selectEvent(event.id)"
          >
            <span>{{ (index + 1).toString().padStart(2, '0') }}</span>
            <span><strong>{{ event.name }}</strong><small>{{ event.location }}</small></span>
            <time :datetime="String(event.year)">{{ event.year }}</time>
          </button>
        </div>
        <div v-else class="atlas-index__empty">
          <p>No broad event references match these filters.</p>
        </div>
      </div>

      <div
        id="atlas-event-map"
        v-show="showAtlasMap"
        class="atlas-map"
        :aria-hidden="isMobileViewport && !showAtlasMap ? 'true' : undefined"
      >
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

        <div v-if="selectedEvent" class="atlas-readout">
          <span>{{ selectedEvent.type }} / {{ selectedEvent.year }}</span>
          <strong>{{ selectedEvent.name }}</strong>
          <p>{{ selectedEvent.location }} · {{ Math.abs(selectedEvent.center[1]).toFixed(1) }}°{{ selectedEvent.center[1] >= 0 ? 'N' : 'S' }}, {{ Math.abs(selectedEvent.center[0]).toFixed(1) }}°{{ selectedEvent.center[0] >= 0 ? 'E' : 'W' }}</p>
        </div>
      </div>
      </section>

      <section
        v-if="selectedEvent"
        class="evidence-passport"
        aria-labelledby="evidence-passport-title"
      >
      <header class="evidence-passport__header">
        <div>
          <p class="eyebrow"><span>Admission note</span> Evidence passport</p>
          <h2 id="evidence-passport-title" tabindex="-1">{{ selectedEvent.name }} · {{ selectedEvent.location }}</h2>
        </div>
        <p>
          This is an analysis admission heuristic. It describes whether the project can inspect this event with its
          current evidence—not how well a community recovered. The displayed band was assigned upstream from a
          weighted sum of five workflow rule outputs; it is not an event-quality measure.
        </p>
      </header>

      <template v-if="selectedPassport">
        <div class="evidence-passport__status">
          <span :class="`status-badge passport-band passport-band--${selectedPassport.readinessBand}`">
            {{ selectedPassport.readinessLabel }}
          </span>
          <p>{{ PUBLIC_EVIDENCE_PASSPORT_ARTIFACT.bandDefinitions.find(({ id }) => id === selectedPassport.readinessBand)?.meaning }}</p>
        </div>

        <div class="evidence-passport__table-wrap data-table-wrap">
          <table class="evidence-table passport-components">
            <caption>Reviewed component-level evidence; the displayed admission band was assigned upstream from a weighted sum, and Compare Mode computes no new overall score</caption>
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
                  <span :class="`status-badge component-state component-state--${component.status}`">
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
          <span>Private-source consistency checks were verified in the restricted environment; private inputs are not present in this public build.</span>
          <span>{{ PUBLIC_EVIDENCE_PASSPORT_ARTIFACT.source.attribution }}</span>
        </footer>
      </template>

      <div v-else class="evidence-passport__empty state-panel">
        <span class="status-badge passport-band passport-band--unassessed">Not assessed in v1</span>
        <div>
          <h3>No reviewed Evidence Passport exists for this event.</h3>
          <p>
            Missing assessment is not evidence of poor data or worse recovery. This event remains in the broad public
            index, while its component evidence stays unclaimed until a reviewed source is admitted.
          </p>
        </div>
      </div>
      </section>
    </div>

    <section v-else id="atlas-compare-panel" class="atlas-comparison" aria-labelledby="atlas-comparison-title">
      <header class="comparison-header">
        <div>
          <p class="eyebrow"><span>Comparison note</span> Evidence before outcome</p>
          <h2 id="atlas-comparison-title">Compare reviewed evidence.</h2>
        </div>
        <p>
          Choose any two public events. The documented hazard family leads the reading; reviewed component values appear only when
          both events have an Evidence Passport. This view does not compare recovery outcomes.
        </p>
      </header>

      <div class="comparison-selectors" aria-describedby="comparison-selector-note">
        <label for="comparison-left-event">
          <span>Event A</span>
          <select id="comparison-left-event" :value="comparisonLeftId" name="comparison-event-a" @change="updateComparisonLeft">
            <optgroup v-for="group in eventGroups" :key="group.hazardFamily" :label="group.hazardFamily">
              <option
                v-for="event in group.events"
                :key="event.id"
                :value="event.id"
                :disabled="event.id === comparisonRightId"
              >
                {{ eventOptionLabel(event) }}
              </option>
            </optgroup>
          </select>
        </label>

        <button
          type="button"
          class="comparison-swap"
          aria-label="Swap Event A and Event B"
          @click="swapComparisonEvents"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M7 7h11m0 0-3-3m3 3-3 3M17 17H6m0 0 3 3m-3-3 3-3" />
          </svg>
          <span>Swap A / B</span>
        </button>

        <label for="comparison-right-event">
          <span>Event B</span>
          <select id="comparison-right-event" :value="comparisonRightId" name="comparison-event-b" @change="updateComparisonRight">
            <optgroup v-for="group in eventGroups" :key="group.hazardFamily" :label="group.hazardFamily">
              <option
                v-for="event in group.events"
                :key="event.id"
                :value="event.id"
                :disabled="event.id === comparisonLeftId"
              >
                {{ eventOptionLabel(event) }}
              </option>
            </optgroup>
          </select>
        </label>
      </div>
      <p id="comparison-selector-note" class="comparison-selector-note">
        All 25 public references remain selectable here, independent of the map filters. The same event cannot occupy
        both sides.
      </p>

      <aside
        v-if="comparison"
        class="comparison-compatibility"
        :class="`comparison-compatibility--${comparison.compatibility.tone}`"
      >
        <div>
          <span>Interpretation status</span>
          <h3>{{ comparison.compatibility.label }}</h3>
          <p>{{ comparison.compatibility.summary }}</p>
        </div>
      </aside>

      <section class="comparison-presets" aria-labelledby="comparison-presets-title">
        <div>
          <h3 id="comparison-presets-title">Guided comparisons</h3>
          <p>{{ PRESET_DISCLAIMER }}</p>
        </div>
        <div class="comparison-presets__grid">
          <button
            v-for="preset in PRESET_COMPARISONS"
            :key="preset.id"
            type="button"
            :aria-pressed="activePresetId === preset.id"
            :class="{ 'is-active': activePresetId === preset.id }"
            @click="applyPreset(preset)"
          >
            <strong>{{ preset.label }}</strong>
            <span>{{ preset.context }}</span>
          </button>
        </div>
      </section>

      <aside v-if="activePreset" class="comparison-preset-note">
        <span>Preset reading note</span>
        <p>{{ activePreset.note }}</p>
      </aside>

      <p class="comparison-live-summary" aria-live="polite" aria-atomic="true">
        {{ comparisonLiveSummary }}
      </p>

      <section v-if="comparison" class="comparison-key-result" aria-labelledby="comparison-key-result-title">
        <header>
          <span>Key component result</span>
          <h3 id="comparison-key-result-title">Start with one reviewed category.</h3>
          <p>The first v1 schema row is shown as an orientation point, not as a score, rank, or most important signal.</p>
        </header>

        <div v-if="comparisonLeftEvent && comparisonRightEvent" class="comparison-events">
          <article>
            <span>Event A · {{ comparisonLeftEvent.type }} · {{ comparisonLeftEvent.hazardFamily }}</span>
            <h3>{{ comparisonLeftEvent.name }}</h3>
            <p>{{ comparisonLeftEvent.location }} · {{ comparisonLeftEvent.region }} · {{ comparisonLeftEvent.year }}</p>
            <strong>Analysis-admission status: {{ passportLabel(comparisonLeftEvent.id) }}</strong>
          </article>
          <article>
            <span>Event B · {{ comparisonRightEvent.type }} · {{ comparisonRightEvent.hazardFamily }}</span>
            <h3>{{ comparisonRightEvent.name }}</h3>
            <p>{{ comparisonRightEvent.location }} · {{ comparisonRightEvent.region }} · {{ comparisonRightEvent.year }}</p>
            <strong>Analysis-admission status: {{ passportLabel(comparisonRightEvent.id) }}</strong>
          </article>
        </div>

        <article v-if="firstComponentPair" class="comparison-component comparison-component--key">
          <div class="comparison-component__definition">
            <span>First evidence category</span>
            <h4>{{ componentDefinition(firstComponentPair.id)?.label }}</h4>
            <p>{{ componentDefinition(firstComponentPair.id)?.meaning }}</p>
          </div>
          <div class="comparison-component__event">
            <span>Event A</span>
            <strong>{{ firstComponentPair.left.points }} / {{ firstComponentPair.left.maxPoints }}</strong>
            <em :class="`status-badge component-state component-state--${firstComponentPair.left.status}`">
              {{ componentStatusLabels[firstComponentPair.left.status] }}
            </em>
          </div>
          <div class="comparison-component__event">
            <span>Event B</span>
            <strong>{{ firstComponentPair.right.points }} / {{ firstComponentPair.right.maxPoints }}</strong>
            <em :class="`status-badge component-state component-state--${firstComponentPair.right.status}`">
              {{ componentStatusLabels[firstComponentPair.right.status] }}
            </em>
          </div>
          <div class="comparison-component__relation">
            <span>Relationship</span>
            <strong>{{ firstComponentPair.samePublishedValue ? 'Same v1 rule-bin value' : 'Different v1 rule-bin values' }}</strong>
            <p>Measurement equivalence is not established.</p>
          </div>
        </article>

        <div v-else class="comparison-unassessed state-panel">
          <span class="status-badge passport-band passport-band--unassessed">Key component unavailable</span>
          <p>Component pairing is withheld when reviewed v1 evidence is missing or invalid; missing evidence is not zero.</p>
        </div>

        <dl class="comparison-summary" aria-label="Dynamic evidence summary">
          <div v-for="summary in comparison.summaries" :key="summary.id">
            <dt>{{ summary.label }}</dt>
            <dd>{{ formatSummaryValue(summary) }}</dd>
            <span v-if="summary.value !== null">{{ summary.suffix }}</span>
            <span v-else>Not available</span>
            <p>{{ summary.note }}</p>
          </div>
        </dl>
      </section>

      <section
        v-if="comparison?.componentPairs.length"
        class="comparison-components"
        aria-labelledby="comparison-components-title"
      >
        <header>
          <div>
            <p class="eyebrow"><span>Five categories</span> Points stay separate</p>
            <h3 id="comparison-components-title">Reviewed component ledger</h3>
          </div>
          <p>
            Each number is shown against its own component maximum. Compare Mode computes no new total, average,
            ordering, or event result. The displayed readiness band was assigned upstream from a weighted sum.
          </p>
        </header>

        <article v-for="pair in comparison.componentPairs" :key="pair.id" class="comparison-component">
          <div class="comparison-component__definition">
            <span>Evidence category</span>
            <h4>{{ componentDefinition(pair.id)?.label }}</h4>
            <p>{{ componentDefinition(pair.id)?.meaning }}</p>
          </div>
          <div class="comparison-component__event">
            <span>Event A</span>
            <strong>{{ pair.left.points }} / {{ pair.left.maxPoints }}</strong>
            <em :class="`status-badge component-state component-state--${pair.left.status}`">
              {{ componentStatusLabels[pair.left.status] }}
            </em>
          </div>
          <div class="comparison-component__event">
            <span>Event B</span>
            <strong>{{ pair.right.points }} / {{ pair.right.maxPoints }}</strong>
            <em :class="`status-badge component-state component-state--${pair.right.status}`">
              {{ componentStatusLabels[pair.right.status] }}
            </em>
          </div>
          <div class="comparison-component__relation">
            <span>Relationship</span>
            <strong>{{ pair.samePublishedValue ? 'Same v1 rule-bin value' : 'Different v1 rule-bin values' }}</strong>
            <p>
              {{ pair.sameState ? 'The published states match.' : 'The published states differ.' }}
              Measurement equivalence is not established.
            </p>
          </div>
        </article>
      </section>

      <section
        v-else-if="comparison?.schemaStatus === 'not-comparable'"
        class="comparison-unassessed state-panel"
        aria-labelledby="comparison-schema-title"
      >
        <span class="status-badge passport-band passport-band--unassessed">Schema not comparable</span>
        <div>
          <h3 id="comparison-schema-title">Component pairing was withheld.</h3>
          <p>The Passport version, component IDs, order, maxima, points, or states did not match the reviewed v1 schema.</p>
          <p>Malformed or missing rows are not counted as differences.</p>
        </div>
      </section>

      <section v-else class="comparison-unassessed state-panel" aria-labelledby="comparison-unassessed-title">
        <span class="status-badge passport-band passport-band--unassessed">Component comparison unavailable</span>
        <div>
          <h3 id="comparison-unassessed-title">One or both events are Not assessed in v1.</h3>
          <p v-if="comparisonLeftEvent && !comparisonLeftPassport">
            <strong>{{ comparisonLeftEvent.name }}</strong> has no reviewed public Evidence Passport.
          </p>
          <p v-if="comparisonRightEvent && !comparisonRightPassport">
            <strong>{{ comparisonRightEvent.name }}</strong> has no reviewed public Evidence Passport.
          </p>
          <p>Not assessed means missing reviewed public evidence—not zero, poor data, or worse recovery.</p>
        </div>
      </section>

      <aside v-if="comparison" class="comparison-measurement-boundary">
        <div>
          <span>Measurement frame / descriptive only</span>
          <h3>Cross-event measurement comparability is not established.</h3>
          <p>{{ comparison.measurementBoundary.statement }}</p>
        </div>
        <div>
          <ul>
            <li v-for="condition in comparison.measurementBoundary.unknownConditions || []" :key="condition">
              {{ condition }}
            </li>
          </ul>
          <ul>
            <li v-for="warning in comparison.warnings" :key="warning">{{ warning }}</li>
          </ul>
        </div>
        <p>Private-source consistency was verified in the restricted environment; this public build cannot reproduce the withheld upstream inputs.</p>
      </aside>

      <footer class="comparison-boundary">
        <span>Boundary / always applies</span>
        <h3>Compare evidence, not outcomes.</h3>
        <p>
          Component points are discrete analysis-admission rule outputs. No similarity score is computed. Compare Mode
          computes no new total, average, event rank, or overall observability measure; the existing readiness band was
          assigned upstream from a weighted sum. Same v1 rule-bin values do not make events equivalent.
        </p>
      </footer>
    </section>

    <details id="atlas-view-help" class="definition-disclosure">
      <summary>Define the evidence states and full Atlas boundary</summary>
      <div>
        <p><strong>Evidence Passport</strong> means five separate, reviewed analysis-admission rule outputs. Its band is not a recovery measure, event grade, or ranking.</p>
        <p><strong>Not assessed</strong> means no reviewed public Passport exists. <strong>Unavailable</strong> means a specific component cannot support the current workflow. Neither state means zero or worse recovery.</p>
      </div>
    </details>

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
