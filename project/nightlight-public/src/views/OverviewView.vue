<script setup>
import { DATA_BOUNDARY } from '../content/copy.js'
import { STUDY_SUMMARY } from '../content/study.js'
</script>

<template>
  <article class="page overview-page">
    <section class="hero">
      <div class="hero__copy">
        <p class="eyebrow"><span>Field note 00</span> Public research edition</p>
        <h1>Reading recovery<br><em>in the dark.</em></h1>
        <p class="hero__lead">
          A two-stage study of disaster impacts, electricity outages, and changes in nighttime light—
          presented here as an inspectable portfolio without publishing restricted fine-grained records.
        </p>
        <div class="hero__actions">
          <RouterLink class="button button--primary" to="/atlas">Open the study atlas</RouterLink>
          <RouterLink class="button button--quiet" to="/methods">Trace the method</RouterLink>
        </div>
      </div>

      <div class="signal-panel" aria-label="Abstract nighttime recovery signal illustration">
        <div class="signal-panel__meta">
          <span>OBS / NIGHT-LIGHT</span>
          <span>AGGREGATE TRACE</span>
        </div>
        <svg viewBox="0 0 640 430" role="img" aria-labelledby="signal-title signal-desc">
          <title id="signal-title">Abstract recovery trace</title>
          <desc id="signal-desc">A local decorative line drops after an event marker and gradually returns toward baseline.</desc>
          <g class="signal-grid">
            <path d="M60 52V378M164 52V378M268 52V378M372 52V378M476 52V378M580 52V378" />
            <path d="M60 76H580M60 152H580M60 228H580M60 304H580M60 378H580" />
          </g>
          <path class="signal-baseline" d="M60 126H580" />
          <path class="signal-trace" d="M60 118C112 121 146 112 192 120S251 128 284 121L302 322C330 342 357 300 382 276S430 218 458 190 514 148 580 132" />
          <path class="signal-afterglow" d="M302 322C330 342 357 300 382 276S430 218 458 190 514 148 580 132" />
          <line class="event-line" x1="302" y1="62" x2="302" y2="360" />
          <circle class="event-pulse" cx="302" cy="322" r="8" />
          <text x="314" y="78">T₀ / EVENT</text>
          <text x="72" y="360">PRE</text>
          <text x="528" y="360">RECOVERY</text>
        </svg>
        <p><span>Signal note</span> Illustration only; no observed series is shipped with this site.</p>
      </div>
    </section>

    <section class="metric-strip" aria-label="Study scale summary">
      <div>
        <span class="metric-strip__index">S2</span>
        <strong>{{ STUDY_SUMMARY.stage2.events }}</strong>
        <p>disaster events across {{ STUDY_SUMMARY.stage2.jurisdictions }} jurisdictions</p>
      </div>
      <div>
        <span class="metric-strip__index">S3</span>
        <strong>{{ STUDY_SUMMARY.stage3.observations.toLocaleString() }}</strong>
        <p>ZIP-event observations across {{ STUDY_SUMMARY.stage3.events }} events</p>
      </div>
      <div>
        <span class="metric-strip__index">M1+</span>
        <strong>{{ STUDY_SUMMARY.descriptiveModel.rSquared.toFixed(4) }}</strong>
        <p>descriptive R², n = {{ STUDY_SUMMARY.descriptiveModel.n }}</p>
      </div>
    </section>

    <section class="overview-grid">
      <div class="section-intro">
        <p class="eyebrow"><span>Research question</span> From signal to explanation</p>
        <h2>What shapes the visible pace of recovery?</h2>
      </div>
      <div class="research-note">
        <p>
          Nighttime light can register disruption and return, but the signal is neither a household-level
          measure nor a verdict on community resilience. The analysis asks how recovery patterns align with
          outage exposure and place-level characteristics inside the observed sample.
        </p>
        <RouterLink class="text-link" to="/findings">Read the findings and their limits <span aria-hidden="true">→</span></RouterLink>
      </div>
    </section>

    <aside class="boundary-banner">
      <div class="boundary-banner__mark" aria-hidden="true">//</div>
      <div>
        <strong>Deliberate publication boundary</strong>
        <p>{{ DATA_BOUNDARY.status }} The interface shows study-scale facts, broad event references, methods, and attribution only.</p>
      </div>
      <RouterLink to="/credits">Inspect the policy</RouterLink>
    </aside>
  </article>
</template>
