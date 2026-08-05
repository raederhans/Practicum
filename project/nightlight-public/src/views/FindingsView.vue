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
</script>

<template>
  <article class="page findings-page">
    <header class="page-heading page-heading--split">
      <div>
        <p class="eyebrow"><span>Field note 03</span> Generalization Autopsy</p>
        <h1>Useful here.<br><em>Unproven there.</em></h1>
      </div>
      <p>
        A model can describe variation in a known sample and still fail the harder test of travelling
        to a held-out event. This page separates those roles instead of turning one favorable number
        into a forecast.
      </p>
    </header>

    <p class="claim-boundary">This is an analysis of model transport failure, not a ranking of community recovery.</p>

    <section class="autopsy-section" aria-labelledby="attractive-result-title">
      <div class="autopsy-section__heading">
        <p class="eyebrow"><span>01</span> The attractive result</p>
        <h2 id="attractive-result-title">A strong descriptive fit can be real—and still be local.</h2>
        <p>
          The R² below is explanatory, in-sample, and fixed-control. It summarizes how the specified
          model fits the analyzed data; it is not a probability of success at the next event.
        </p>
      </div>

      <figure class="finding-hero" aria-labelledby="fit-title fit-caption">
        <div class="finding-hero__number">
          <span>{{ metrics['in-sample-r-squared'].modelFamily }}</span>
          <strong>{{ metrics['in-sample-r-squared'].value.toFixed(4) }}</strong>
          <small>R² / explanatory description</small>
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
          <figcaption id="fit-caption">The displayed bar is one descriptive R². It says nothing by itself about a new event.</figcaption>
        </div>
      </figure>

      <table class="evidence-table">
        <caption>Text alternative for the descriptive fit chart</caption>
        <thead><tr><th scope="col">Metric</th><th scope="col">Value</th><th scope="col">Role</th><th scope="col">Meaning</th></tr></thead>
        <tbody><tr><td>R²</td><td>0.7603</td><td>Explanatory / description</td><td>Within-sample fixed-control fit, not future-event accuracy.</td></tr></tbody>
      </table>
    </section>

    <section class="autopsy-section autopsy-section--contrast" aria-labelledby="harder-test-title">
      <div class="autopsy-section__heading">
        <p class="eyebrow"><span>02</span> The harder test</p>
        <h2 id="harder-test-title">Leave one event out, then ask whether the role travels.</h2>
        <p>
          The cross-event damage-ranking check trains without one event and evaluates that held-out event. R² and AUC
          are both bounded statistics, but they measure different tasks and are never placed on one shared score scale here.
        </p>
      </div>
      <div class="transport-callout">
        <p class="eyebrow"><span>Held-out damage ranking</span> {{ metrics['cross-event-logit-auc'].validationDesign }}</p>
        <strong>{{ metrics['cross-event-logit-auc'].value.toFixed(4) }}</strong>
        <p>AUC / damage ranking. The reviewed aggregate is below the 0.50 reference. It does not answer a recovery-transport task or provide a calibrated prediction.</p>
      </div>
    </section>

    <section class="autopsy-section" aria-labelledby="role-matrix-title">
      <div class="autopsy-section__heading">
        <p class="eyebrow"><span>03</span> What improved / what failed</p>
        <h2 id="role-matrix-title">The role matrix prevents a good answer from answering the wrong question.</h2>
      </div>
      <table class="evidence-table role-matrix">
        <caption>Model roles and what this public artifact can support</caption>
        <thead><tr><th scope="col">Role</th><th scope="col">Metric type</th><th scope="col">Evaluation</th><th scope="col">Reading rule</th></tr></thead>
        <tbody><tr v-for="row in roleRows" :key="row[0]"><th scope="row">{{ row[0] }}</th><td>{{ row[1] }}</td><td>{{ row[2] }}</td><td>{{ row[3] }}</td></tr></tbody>
      </table>
    </section>

    <section class="autopsy-section" aria-labelledby="evidence-cards-title">
      <div class="autopsy-section__heading">
        <p class="eyebrow"><span>04</span> Evidence cards</p>
        <h2 id="evidence-cards-title">Small, inspectable claims—not a hidden score.</h2>
      </div>
      <div class="evidence-cards">
        <article v-for="metric in evidenceCards" :key="metric.id" class="evidence-card">
          <p class="eyebrow"><span>{{ metric.modelRole }}</span> {{ metric.metricType }}</p>
          <h3>{{ metric.metricName }} <strong>{{ metric.value.toFixed(4) }}</strong></h3>
          <dl>
            <div><dt>Task</dt><dd>{{ metric.validationDesign }}</dd></div>
            <div><dt>Quality</dt><dd><span class="status-label">{{ metric.qualityStatus }}</span> · {{ metric.publicationStatus }}</dd></div>
            <div><dt>Supports</dt><dd>{{ metric.supportedClaim }}</dd></div>
            <div><dt>Does not support</dt><dd>{{ metric.unsupportedClaim }}</dd></div>
            <div><dt>Cohort / lock</dt><dd>{{ metric.cohort }} · {{ metric.sampleLock }}</dd></div>
            <div><dt>Source</dt><dd>{{ sources[metric.sourceArtifactId].id }} · {{ sources[metric.sourceArtifactId].version }} · SHA-256 {{ sources[metric.sourceArtifactId].sha256.slice(0, 12) }}…</dd></div>
          </dl>
        </article>
      </div>
    </section>

  </article>
</template>
