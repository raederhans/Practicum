<script setup>
import { computed, ref, watch } from 'vue'

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

const viewMode = ref('explore')
const query = ref('')
const selectedHazardFamily = ref('All')
const selectedId = ref(EVENTS[0].id)
const comparisonLeftId = ref(EVENTS[0].id)
const comparisonRightId = ref(resolveComparisonPeerId(EVENTS, comparisonLeftId.value, comparisonLeftId.value))

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
const activePresetId = computed(() => PRESET_COMPARISONS.find(({ eventIds }) => (
  eventIds[0] === comparisonLeftId.value && eventIds[1] === comparisonRightId.value
))?.id ?? null)
const activePreset = computed(() => PRESET_COMPARISONS.find(({ id }) => id === activePresetId.value) ?? null)
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

function applyPreset(preset) {
  viewMode.value = 'compare'
  comparisonLeftId.value = preset.eventIds[0]
  comparisonRightId.value = preset.eventIds[1]
}

function swapComparisonEvents() {
  const previousLeftId = comparisonLeftId.value
  comparisonLeftId.value = comparisonRightId.value
  comparisonRightId.value = previousLeftId
}

const comparisonLiveSummary = computed(() => {
  if (!comparison.value) return 'Choose two different events to compare.'
  if (comparison.value.schemaStatus !== 'paired-v1') {
    return `Comparison updated. ${comparison.value.compatibility.label}. ${comparison.value.passportCoverage} of 2 reviewed Passports; component pairing is unavailable.`
  }
  return `Comparison updated. ${comparison.value.compatibility.label}. Review the interpretation limits and each v1 component row separately. No similarity score is computed.`
})

watch(visibleEvents, (events) => {
  selectedId.value = resolveSelectedId(events, selectedId.value)
}, { flush: 'sync' })

watch(selectedId, (eventId) => {
  if (viewMode.value !== 'explore' || !eventId) return
  comparisonLeftId.value = eventId
}, { flush: 'sync' })

watch(comparisonLeftId, (eventId) => {
  comparisonRightId.value = resolveComparisonPeerId(EVENTS, eventId, comparisonRightId.value)
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

    <fieldset class="atlas-view-mode">
      <legend>Evidence view</legend>
      <label :class="{ 'is-active': viewMode === 'explore' }">
        <input v-model="viewMode" type="radio" name="atlas-evidence-view" value="explore">
        <span><strong>Explore one event</strong><small>Map, index, and one Evidence Passport</small></span>
      </label>
      <label :class="{ 'is-active': viewMode === 'compare' }">
        <input v-model="viewMode" type="radio" name="atlas-evidence-view" value="compare">
        <span><strong>Compare events</strong><small>Category-first evidence inspection</small></span>
      </label>
    </fieldset>

    <template v-if="viewMode === 'explore'">
      <section class="atlas-controls" aria-label="Filter event index">
      <label>
        <span>Search the index</span>
        <input v-model="query" type="search" placeholder="Event, place, or year" autocomplete="off">
      </label>
      <label>
        <span>Hazard family</span>
        <select v-model="selectedHazardFamily">
          <option v-for="hazardFamily in HAZARD_FAMILIES" :key="hazardFamily" :value="hazardFamily">{{ hazardFamily }}</option>
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
          current evidence—not how well a community recovered. The displayed band was assigned upstream from a
          weighted sum of five workflow rule outputs; it is not an event-quality measure.
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
          <span>Private-source recomputation requires restricted-environment verification and is not performed by this public build.</span>
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
    </template>

    <section v-else class="atlas-comparison" aria-labelledby="atlas-comparison-title">
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

      <div class="comparison-selectors" aria-describedby="comparison-selector-note">
        <label for="comparison-left-event">
          <span>Event A</span>
          <select id="comparison-left-event" v-model="comparisonLeftId">
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
          <select id="comparison-right-event" v-model="comparisonRightId">
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

      <div v-if="comparisonLeftEvent && comparisonRightEvent" class="comparison-events">
        <article>
          <span>Event A · {{ comparisonLeftEvent.type }} · {{ comparisonLeftEvent.hazardFamily }}</span>
          <h3>{{ comparisonLeftEvent.name }}</h3>
          <p>{{ comparisonLeftEvent.location }} · {{ comparisonLeftEvent.region }} · {{ comparisonLeftEvent.year }}</p>
          <strong>{{ passportLabel(comparisonLeftEvent.id) }}</strong>
        </article>
        <article>
          <span>Event B · {{ comparisonRightEvent.type }} · {{ comparisonRightEvent.hazardFamily }}</span>
          <h3>{{ comparisonRightEvent.name }}</h3>
          <p>{{ comparisonRightEvent.location }} · {{ comparisonRightEvent.region }} · {{ comparisonRightEvent.year }}</p>
          <strong>{{ passportLabel(comparisonRightEvent.id) }}</strong>
        </article>
      </div>

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
        <ul>
          <li v-for="warning in comparison.warnings" :key="warning">{{ warning }}</li>
        </ul>
      </aside>

      <aside v-if="activePreset" class="comparison-preset-note">
        <span>Preset reading note</span>
        <p>{{ activePreset.note }}</p>
      </aside>

      <aside v-if="comparison" class="comparison-measurement-boundary">
        <div>
          <span>Measurement frame / descriptive only</span>
          <h3>Cross-event measurement comparability is not established.</h3>
          <p>{{ comparison.measurementBoundary.statement }}</p>
        </div>
        <ul>
          <li v-for="condition in comparison.measurementBoundary.unknownConditions || []" :key="condition">
            {{ condition }}
          </li>
        </ul>
        <p>Private-source recomputation remains a restricted-environment release gate.</p>
      </aside>

      <p class="comparison-live-summary" aria-live="polite" aria-atomic="true">
        {{ comparisonLiveSummary }}
      </p>

      <dl v-if="comparison" class="comparison-summary" aria-label="Dynamic evidence summary">
        <div v-for="summary in comparison.summaries" :key="summary.id">
          <dt>{{ summary.label }}</dt>
          <dd>{{ formatSummaryValue(summary) }}</dd>
          <span v-if="summary.value !== null">{{ summary.suffix }}</span>
          <span v-else>Not available</span>
          <p>{{ summary.note }}</p>
        </div>
      </dl>

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
            <em :class="`component-state component-state--${pair.left.status}`">
              {{ componentStatusLabels[pair.left.status] }}
            </em>
          </div>
          <div class="comparison-component__event">
            <span>Event B</span>
            <strong>{{ pair.right.points }} / {{ pair.right.maxPoints }}</strong>
            <em :class="`component-state component-state--${pair.right.status}`">
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
        class="comparison-unassessed"
        aria-labelledby="comparison-schema-title"
      >
        <span class="passport-band passport-band--unassessed">Schema not comparable</span>
        <div>
          <h3 id="comparison-schema-title">Component pairing was withheld.</h3>
          <p>The Passport version, component IDs, order, maxima, points, or states did not match the reviewed v1 schema.</p>
          <p>Malformed or missing rows are not counted as differences.</p>
        </div>
      </section>

      <section v-else class="comparison-unassessed" aria-labelledby="comparison-unassessed-title">
        <span class="passport-band passport-band--unassessed">Component comparison unavailable</span>
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
