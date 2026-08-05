<template>
  <article class="page methods-page">
    <header class="page-heading page-heading--split">
      <div>
        <p class="eyebrow"><span>Field note 04</span> Methods</p>
        <h1>A traceable path<br><em>from light to model.</em></h1>
      </div>
      <p>
        The public edition describes the workflow but does not ship inputs, fine-grained derivatives,
        intermediate tables, or fitted model artifacts.
      </p>
    </header>

    <section class="method-timeline" aria-label="Research workflow">
      <article>
        <span class="method-timeline__number">01</span>
        <div>
          <p class="eyebrow">Signal preparation</p>
          <h2>Establish comparable nighttime-light traces.</h2>
          <p>Align event windows, apply quality screens, and separate pre-event reference periods from post-event change.</p>
        </div>
        <p class="method-timeline__output">Output retained privately:<br><strong>quality-controlled temporal extracts</strong></p>
      </article>
      <article>
        <span class="method-timeline__number">02</span>
        <div>
          <p class="eyebrow">Event-level analysis</p>
          <h2>Describe disruption and return across 25 events.</h2>
          <p>Summarize the scale and timing of visible change across 17 jurisdictions without treating light as a direct welfare measure.</p>
        </div>
        <p class="method-timeline__output">Public result:<br><strong>sample scale and broad context</strong></p>
      </article>
      <article>
        <span class="method-timeline__number">03</span>
        <div>
          <p class="eyebrow">Place-level modeling</p>
          <h2>Relate recovery variation to outage exposure and context.</h2>
          <p>Construct ZIP-event observations, screen complete cases, and report descriptive fit with explicit interpretation limits.</p>
        </div>
        <p class="method-timeline__output">Public result:<br><strong>aggregated model diagnostics</strong></p>
      </article>
    </section>

    <section class="method-principles" aria-labelledby="public-artifact-title">
      <div>
        <p class="eyebrow"><span>Public contract</span> Inspectable limits</p>
        <h2 id="public-artifact-title">Public Generalization Artifact v1</h2>
      </div>
      <ol>
        <li><span>01</span><strong>Every public metric has a role.</strong><p>Metrics are labeled as description, ranking, and calibration so one task is not mistaken for another.</p></li>
        <li><span>02</span><strong>Lineage is part of the value.</strong><p>Each admitted aggregate identifies a source artifact, version, SHA-256, cohort, sample lock, and validation design.</p></li>
        <li><span>03</span><strong>Cross-event validation has its own test.</strong><p>Held-out-event evaluation is reported separately from in-sample explanatory fit; unlike metrics are not merged into one score.</p></li>
        <li><span>04</span><strong>Withholding is an outcome.</strong><p>When a value lacks an approved aggregate lineage or a safe public grain, the contract keeps it unavailable rather than substituting a demonstration.</p></li>
      </ol>
    </section>

    <section class="method-principles">
      <div>
        <p class="eyebrow"><span>Operating rules</span> Research hygiene</p>
        <h2>Four guardrails shape the portfolio.</h2>
      </div>
      <ol>
        <li><span>01</span><strong>Observation is not causation.</strong><p>Model summaries remain descriptive unless a design can identify effects.</p></li>
        <li><span>02</span><strong>Light is a proxy.</strong><p>Nighttime radiance can reveal disruption, but it does not represent every lived consequence.</p></li>
        <li><span>03</span><strong>Aggregation is a boundary.</strong><p>Public outputs are deliberately coarser than the internal analytical material.</p></li>
        <li><span>04</span><strong>Absence stays visible.</strong><p>The site names withheld layers instead of substituting demonstrations or synthetic stand-ins.</p></li>
      </ol>
    </section>
  </article>
</template>
