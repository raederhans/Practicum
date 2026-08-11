<script setup>
function navigateToSection(event) {
  const target = document.getElementById(event.currentTarget.hash.slice(1))
  if (!(target instanceof HTMLElement)) return
  target.scrollIntoView({ block: 'start' })
  target.focus({ preventScroll: true })
}
</script>

<template>
  <article class="page methods-page">
    <header class="page-heading page-heading--split">
      <div>
        <p class="eyebrow"><span>Field note 04</span> Methods</p>
        <h1 class="focus-target" data-route-focus tabindex="-1">A traceable path<br><em>from light to model.</em></h1>
      </div>
      <p class="page-lede page-summary">
        Private inputs → processed signals → place-level model → admission → public artifact.
      </p>
    </header>

    <nav class="in-page-nav" aria-labelledby="methods-page-nav-title">
      <p id="methods-page-nav-title">On this page</p>
      <ol class="content-section-nav">
        <li><a href="#research-workflow-title" @click.prevent="navigateToSection">Five-step workflow</a></li>
        <li><a href="#public-artifact-title" @click.prevent="navigateToSection">Public contract</a></li>
        <li><a href="#methods-guardrails-title" @click.prevent="navigateToSection">Operating guardrails</a></li>
      </ol>
    </nav>

    <details class="definition-disclosure">
      <summary>What the public edition does not ship</summary>
      <div>
        <p>The public edition describes the workflow but does not ship inputs, fine-grained derivatives, intermediate tables, or fitted model artifacts.</p>
      </div>
    </details>

    <section class="method-timeline" aria-labelledby="research-workflow-title">
      <h2 id="research-workflow-title" tabindex="-1">Five steps from private evidence to a bounded public artifact.</h2>
      <article id="method-step-inputs">
        <span class="method-timeline__number">01</span>
        <div>
          <p class="eyebrow">Input boundary</p>
          <h3>Review source rights, event windows, and usable evidence.</h3>
          <details class="method-timeline__details">
            <summary>Read step details</summary>
            <p>Nighttime-light, outage, and place-context inputs remain inside the analytical workspace at their approved grain. Missing or restricted inputs are not replaced with demonstrations.</p>
          </details>
        </div>
        <p class="method-timeline__output">Private boundary:<br><strong>raw and fine-grained inputs</strong></p>
      </article>
      <article id="method-step-signals">
        <span class="method-timeline__number">02</span>
        <div>
          <p class="eyebrow">Signal processing</p>
          <h3>Establish quality-controlled nighttime-light traces.</h3>
          <details class="method-timeline__details">
            <summary>Read step details</summary>
            <p>Align event windows, apply quality screens, and separate pre-event reference periods from post-event change without treating light as a direct welfare measure.</p>
          </details>
        </div>
        <p class="method-timeline__output">Output retained privately:<br><strong>temporal extracts and intermediate tables</strong></p>
      </article>
      <article id="method-step-model">
        <span class="method-timeline__number">03</span>
        <div>
          <p class="eyebrow">Place-level modeling</p>
          <h3>Relate recovery variation to outage exposure and context.</h3>
          <details class="method-timeline__details">
            <summary>Read step details</summary>
            <p>Construct ZIP-event observations, screen complete cases, and report descriptive fit with explicit interpretation limits.</p>
          </details>
        </div>
        <p class="method-timeline__output">Public result:<br><strong>aggregated model diagnostics</strong></p>
      </article>
      <article id="method-step-admission">
        <span class="method-timeline__number">04</span>
        <div>
          <p class="eyebrow">Analysis admission</p>
          <h3>Keep five Evidence Passport checks separate.</h3>
          <details class="method-timeline__details">
            <summary>Read step details</summary>
            <p>Reviewed rule outputs describe whether the declared workflow can inspect an event. Missing assessment remains Not assessed; unavailable components remain unavailable rather than becoming zero.</p>
          </details>
        </div>
        <p class="method-timeline__output">Public result:<br><strong>component states and admission band</strong></p>
      </article>
      <article id="method-step-publication">
        <span class="method-timeline__number">05</span>
        <div>
          <p class="eyebrow">Public artifact</p>
          <h3>Admit only reviewed aggregates with traceable lineage.</h3>
          <details class="method-timeline__details">
            <summary>Read step details</summary>
            <p>Each public value carries its role, cohort, sample lock, validation design, source version, and SHA-256. The static site bundles local artifacts only and makes no runtime data request.</p>
          </details>
        </div>
        <p class="method-timeline__output">Published boundary:<br><strong>aggregate-only static artifacts</strong></p>
      </article>
    </section>

    <section class="method-principles" aria-labelledby="public-artifact-title">
      <div>
        <p class="eyebrow"><span>Public contract</span> Inspectable limits</p>
        <h2 id="public-artifact-title" tabindex="-1">Public Generalization Artifact v1</h2>
      </div>
      <ol>
        <li><span>01</span><strong>Every public metric has a role.</strong><p>Metrics are labeled as description, ranking, and calibration so one task is not mistaken for another.</p></li>
        <li><span>02</span><strong>Lineage is part of the value.</strong><p>Each admitted aggregate identifies a source artifact, version, SHA-256, cohort, sample lock, and validation design.</p></li>
        <li><span>03</span><strong>Cross-event validation has its own test.</strong><p>Held-out-event evaluation is reported separately from in-sample explanatory fit; unlike metrics are not merged into one score.</p></li>
        <li><span>04</span><strong>Withholding is an outcome.</strong><p>When a value lacks an approved aggregate lineage or a safe public grain, the contract keeps it unavailable rather than substituting a demonstration.</p></li>
      </ol>
    </section>

    <section class="method-principles" aria-labelledby="methods-guardrails-title">
      <div>
        <p class="eyebrow"><span>Operating rules</span> Research hygiene</p>
        <h2 id="methods-guardrails-title" tabindex="-1">Four guardrails shape the portfolio.</h2>
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
