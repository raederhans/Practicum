<script setup>
import { FINDINGS_COPY } from '../content/copy.js'
import { STUDY_SUMMARY } from '../content/study.js'
</script>

<template>
  <article class="page findings-page">
    <header class="page-heading page-heading--split">
      <div>
        <p class="eyebrow"><span>Field note 03</span> Findings</p>
        <h1>Strong description.<br><em>Careful claims.</em></h1>
      </div>
      <p>
        These are compact summaries of the analyzed sample. They describe model fit and sensitivity;
        they do not establish causes, predict future disasters, or rank communities.
      </p>
    </header>

    <section class="finding-hero">
      <div class="finding-hero__number">
        <span>{{ FINDINGS_COPY.model.label }}</span>
        <strong>{{ STUDY_SUMMARY.descriptiveModel.rSquared.toFixed(4) }}</strong>
        <small>R² / descriptive</small>
      </div>
      <div class="finding-hero__chart">
        <svg viewBox="0 0 760 270" role="img" aria-labelledby="fit-title fit-desc">
          <title id="fit-title">Descriptive model fit summary</title>
          <desc id="fit-desc">Bars show R squared 0.7603 and adjusted R squared 0.7543.</desc>
          <g class="fit-grid">
            <path d="M120 30V230M262 30V230M404 30V230M546 30V230M688 30V230" />
          </g>
          <text x="24" y="91">R²</text>
          <rect class="fit-bar fit-bar--primary" x="120" y="58" :width="STUDY_SUMMARY.descriptiveModel.rSquared * 568" height="48" />
          <text class="fit-value" :x="128 + STUDY_SUMMARY.descriptiveModel.rSquared * 568" y="89">{{ STUDY_SUMMARY.descriptiveModel.rSquared.toFixed(4) }}</text>
          <text x="24" y="183">ADJ. R²</text>
          <rect class="fit-bar fit-bar--secondary" x="120" y="150" :width="STUDY_SUMMARY.descriptiveModel.adjustedRSquared * 568" height="48" />
          <text class="fit-value" :x="128 + STUDY_SUMMARY.descriptiveModel.adjustedRSquared * 568" y="181">{{ STUDY_SUMMARY.descriptiveModel.adjustedRSquared.toFixed(4) }}</text>
          <text x="120" y="251">0</text><text x="671" y="251">1.0</text>
        </svg>
        <p>{{ FINDINGS_COPY.model.interpretation }}</p>
      </div>
    </section>

    <section class="finding-grid">
      <article>
        <p class="eyebrow"><span>Sample</span> M1+ complete cases</p>
        <strong class="finding-grid__number">{{ STUDY_SUMMARY.descriptiveModel.n }}</strong>
        <p>observations in the reported descriptive model.</p>
      </article>
      <article>
        <p class="eyebrow"><span>Coverage</span> Stage 3</p>
        <strong class="finding-grid__number">{{ STUDY_SUMMARY.stage3.events }} / {{ STUDY_SUMMARY.stage3.states }}</strong>
        <p>events across U.S. states, represented by {{ STUDY_SUMMARY.stage3.observations.toLocaleString() }} ZIP-event observations.</p>
      </article>
      <article class="finding-grid__caution">
        <p class="eyebrow"><span>Diagnostic</span> {{ FINDINGS_COPY.sensitivity.label }}</p>
        <strong class="finding-grid__number">{{ FINDINGS_COPY.sensitivity.value.toFixed(3) }}</strong>
        <p>{{ FINDINGS_COPY.sensitivity.caution }}</p>
      </article>
    </section>

    <section class="limits-ledger">
      <div>
        <p class="eyebrow"><span>Interpretation ledger</span> What the page can support</p>
        <h2>Read the coefficient; keep the boundary.</h2>
      </div>
      <dl>
        <div><dt>Supported</dt><dd>Within-sample descriptive association and model fit.</dd></div>
        <div><dt>Not supported</dt><dd>Causal effect, fairness judgment, household outcome, or future-event forecast.</dd></div>
        <div><dt>Public evidence</dt><dd>Aggregated sample counts, model summaries, methods, and data attribution.</dd></div>
      </dl>
    </section>
  </article>
</template>
