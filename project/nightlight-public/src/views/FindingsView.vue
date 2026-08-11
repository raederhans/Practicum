<script setup>
import { PUBLIC_GENERALIZATION_ARTIFACT } from '../content/generalizationArtifact.js'

const metrics = Object.freeze(Object.fromEntries(
  PUBLIC_GENERALIZATION_ARTIFACT.metrics.map((metric) => [metric.id, metric]),
))
const sources = Object.freeze(Object.fromEntries(
  PUBLIC_GENERALIZATION_ARTIFACT.sources.map((source) => [source.id, source]),
))
const evidenceCards = Object.freeze([
  metrics['in-sample-r-squared'],
  metrics['cross-event-logit-auc'],
  metrics['descriptive-sensitivity-ratio'],
].filter((metric) => metric.publicationStatus === 'admitted'))
const roleRows = Object.freeze([
  ['Explanatory', 'Description', 'In-sample fixed-control fit', 'R² belongs here; it is not future-event accuracy.'],
  ['Damage ranking', 'Ranking', 'Leave-one-event-out damage ranking', 'The admitted AUC is below the 0.50 reference.'],
  ['Recovery transport', 'Unavailable', 'No admitted recovery-transport metric in v1', 'Do not relabel a damage-ranking AUC as recovery transport.'],
  ['Secondary interpretation', 'Description', 'Descriptive sensitivity', 'Interpretive only; not a causal or fairness result.'],
])

function navigateToSection(event) {
  const target = document.getElementById(event.currentTarget.hash.slice(1))
  if (!(target instanceof HTMLElement)) return
  target.scrollIntoView({ block: 'start' })
  target.focus({ preventScroll: true })
}
</script>

<template>
  <article class="page findings-page">
    <header class="page-heading page-heading--split">
      <div>
        <p class="eyebrow"><span>Field note 03</span> Generalization Autopsy</p>
        <h1 class="focus-target" data-route-focus tabindex="-1">Useful here.<br><em>Unproven there.</em></h1>
      </div>
      <p class="page-lede page-summary">
        Description, damage ranking, and recovery transport are separate roles. This public artifact
        admits evidence for the first two and no recovery-transport metric.
      </p>
    </header>

    <p class="claim-boundary">This is an analysis of model transport failure, not a ranking of community recovery.</p>

    <section class="findings-role-summary" aria-labelledby="findings-role-summary-title">
      <h2 id="findings-role-summary-title" tabindex="-1">Three roles, three distinct readings.</h2>
      <div class="evidence-cards">
        <article>
          <p class="eyebrow"><span>Description</span> Known sample</p>
          <h3>R² {{ metrics['in-sample-r-squared'].value.toFixed(4) }}</h3>
          <p>{{ metrics['in-sample-r-squared'].unit }}. Descriptive, in-sample, and not future-event accuracy.</p>
        </article>
        <article>
          <p class="eyebrow"><span>Damage ranking</span> Held-out event</p>
          <h3>AUC {{ metrics['cross-event-logit-auc'].value.toFixed(4) }}</h3>
          <p>{{ metrics['cross-event-logit-auc'].unit }}. A ranking statistic below the 0.50 reference, not a calibrated probability.</p>
        </article>
        <article>
          <p class="eyebrow"><span>Recovery transport</span> No admitted metric</p>
          <h3>Unavailable</h3>
          <p>No admitted recovery-transport metric exists in v1. The damage-ranking AUC cannot fill this role.</p>
        </article>
      </div>
    </section>

    <nav class="in-page-nav" aria-labelledby="findings-page-nav-title">
      <p id="findings-page-nav-title">On this page</p>
      <ol class="content-section-nav">
        <li><a href="#attractive-result-title" @click.prevent="navigateToSection">Descriptive fit</a></li>
        <li><a href="#harder-test-title" @click.prevent="navigateToSection">Damage-ranking test</a></li>
        <li><a href="#role-matrix-title" @click.prevent="navigateToSection">Role matrix</a></li>
        <li><a href="#evidence-cards-title" @click.prevent="navigateToSection">Evidence cards</a></li>
      </ol>
    </nav>

    <details class="definition-disclosure">
      <summary>How to read R², AUC, admission, and recovery outcome</summary>
      <div>
        <p>A model can describe variation in a known sample and still fail the harder test of travelling to a held-out event. These roles are kept separate instead of turning one favorable number into a forecast.</p>
        <p><strong>R²</strong> is the unitless [0–1] descriptive coefficient admitted for this fixed sample. It is not future-event accuracy.</p>
        <p><strong>AUC</strong> is a unitless [0–1] ranking statistic; 0.50 is the no-ranking reference used here. It is not a calibrated probability or recovery-transport measure.</p>
        <p><strong>Analysis admission/readiness</strong> describes whether declared evidence checks are available. A <strong>recovery outcome</strong> would describe what happened to a community; this page does not rank those outcomes.</p>
      </div>
    </details>

    <section class="autopsy-section" aria-labelledby="attractive-result-title">
      <div class="autopsy-section__heading">
        <p class="eyebrow"><span>01</span> The attractive result</p>
        <h2 id="attractive-result-title" tabindex="-1">A strong descriptive fit can be real—and still be local.</h2>
        <p>
          The R² below is explanatory, in-sample, and fixed-control. It summarizes how the specified
          model fits the analyzed data; it is not a probability of success at the next event.
        </p>
      </div>

      <figure class="finding-hero" aria-labelledby="fit-title fit-caption">
        <div class="finding-hero__number">
          <span>{{ metrics['in-sample-r-squared'].modelFamily }}</span>
          <strong>{{ metrics['in-sample-r-squared'].value.toFixed(4) }}</strong>
          <small>R² / explanatory description · {{ metrics['in-sample-r-squared'].unit }}</small>
        </div>
        <div class="finding-hero__chart">
          <svg viewBox="0 0 760 270" role="img" aria-labelledby="fit-title fit-desc">
            <title id="fit-title">In-sample descriptive fit summary</title>
            <desc id="fit-desc">R squared is 0.7603. This is a descriptive in-sample fit statistic, not future-event accuracy.</desc>
            <g class="fit-grid"><path d="M120 30V230M262 30V230M404 30V230M546 30V230M688 30V230" /></g>
            <text x="24" y="136">R²</text>
            <rect class="fit-bar fit-bar--primary" x="120" y="104" :width="metrics['in-sample-r-squared'].value * 568" height="48" />
            <text class="fit-value" :x="128 + metrics['in-sample-r-squared'].value * 568" y="136">{{ metrics['in-sample-r-squared'].value.toFixed(4) }}</text>
            <text x="120" y="251">0</text><text x="671" y="251">1.0</text>
          </svg>
        </div>
        <figcaption id="fit-caption">The displayed bar uses the artifact's unitless [0–1] range for this descriptive R². It says nothing by itself about a new event.</figcaption>
      </figure>

      <div class="data-table-wrap">
        <table class="evidence-table">
          <caption>Text alternative for the descriptive fit chart</caption>
          <thead><tr><th scope="col">Metric</th><th scope="col">Value and unit</th><th scope="col">Role</th><th scope="col">Meaning</th></tr></thead>
          <tbody><tr><td>R²</td><td>0.7603 · unitless [0–1]</td><td>Explanatory / description</td><td>Within-sample fixed-control fit, not future-event accuracy.</td></tr></tbody>
        </table>
      </div>
    </section>

    <section class="autopsy-section autopsy-section--contrast" aria-labelledby="harder-test-title">
      <div class="autopsy-section__heading">
        <p class="eyebrow"><span>02</span> The harder test</p>
        <h2 id="harder-test-title" tabindex="-1">Leave one event out, then ask whether the role travels.</h2>
        <p>
          The cross-event damage-ranking check trains without one event and evaluates that held-out event. R² and AUC
          are both bounded statistics, but they measure different tasks and are never placed on one shared score scale here.
        </p>
      </div>
      <div class="transport-callout">
        <p class="eyebrow"><span>Held-out damage ranking</span> {{ metrics['cross-event-logit-auc'].validationDesign }}</p>
        <strong>{{ metrics['cross-event-logit-auc'].value.toFixed(4) }}</strong>
        <p>AUC / damage ranking · {{ metrics['cross-event-logit-auc'].unit }}. The reviewed aggregate is below the 0.50 no-ranking reference. It does not answer a recovery-transport task or provide a calibrated prediction.</p>
      </div>
    </section>

    <section class="autopsy-section" aria-labelledby="role-matrix-title">
      <div class="autopsy-section__heading">
        <p class="eyebrow"><span>03</span> What improved / what failed</p>
        <h2 id="role-matrix-title" tabindex="-1">The role matrix prevents a good answer from answering the wrong question.</h2>
      </div>
      <div class="data-table-wrap">
        <table class="evidence-table role-matrix">
          <caption>Model roles and what this public artifact can support</caption>
          <thead><tr><th scope="col">Role</th><th scope="col">Metric type</th><th scope="col">Evaluation</th><th scope="col">Reading rule</th></tr></thead>
          <tbody><tr v-for="row in roleRows" :key="row[0]"><th scope="row">{{ row[0] }}</th><td>{{ row[1] }}</td><td>{{ row[2] }}</td><td>{{ row[3] }}</td></tr></tbody>
        </table>
      </div>
    </section>

    <section class="autopsy-section" aria-labelledby="evidence-cards-title">
      <div class="autopsy-section__heading">
        <p class="eyebrow"><span>04</span> Evidence cards</p>
        <h2 id="evidence-cards-title" tabindex="-1">Small, inspectable claims—not a hidden score.</h2>
      </div>
      <div class="evidence-cards">
        <article v-for="metric in evidenceCards" :key="metric.id" class="evidence-card">
          <p class="eyebrow"><span>{{ metric.modelRole }}</span> {{ metric.metricType }}</p>
          <h3>{{ metric.metricName }} <strong>{{ metric.value.toFixed(4) }}</strong></h3>
          <dl>
            <div><dt>Task</dt><dd>{{ metric.validationDesign }}</dd></div>
            <div><dt>Value / unit</dt><dd>{{ metric.value.toFixed(4) }} · {{ metric.unit }}</dd></div>
            <div><dt>Quality</dt><dd><span class="status-badge status-label">{{ metric.qualityStatus }}</span> · {{ metric.publicationStatus }}</dd></div>
            <div><dt>Supports</dt><dd>{{ metric.supportedClaim }}</dd></div>
            <div><dt>Does not support</dt><dd>{{ metric.unsupportedClaim }}</dd></div>
          </dl>
          <details class="evidence-card__lineage">
            <summary>Inspect cohort and source lineage</summary>
            <dl>
              <div><dt>Cohort / lock</dt><dd>{{ metric.cohort }} · {{ metric.sampleLock }}</dd></div>
              <div><dt>Source</dt><dd>{{ sources[metric.sourceArtifactId].id }} · {{ sources[metric.sourceArtifactId].version }} · SHA-256 {{ sources[metric.sourceArtifactId].sha256.slice(0, 12) }}…</dd></div>
            </dl>
          </details>
        </article>
      </div>
    </section>

  </article>
</template>
