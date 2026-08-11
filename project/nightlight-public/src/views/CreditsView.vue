<script setup>
import { DATA_BOUNDARY } from '../content/copy.js'

function navigateToSection(event) {
  const target = document.getElementById(event.currentTarget.hash.slice(1))
  if (!(target instanceof HTMLElement)) return
  target.scrollIntoView({ block: 'start' })
  target.focus({ preventScroll: true })
}
</script>

<template>
  <article class="page credits-page">
    <header class="page-heading page-heading--split">
      <div>
        <p class="eyebrow"><span>Field note 05</span> Credits / data policy</p>
        <h1 class="focus-target" data-route-focus tabindex="-1">Credit the work.<br><em>Keep the line clear.</em></h1>
      </div>
      <p class="page-lede page-summary">
        Authorship, source rights, public grain, runtime privacy, known limits, and license remain separate.
      </p>
    </header>

    <section class="policy-facts page-summary" aria-labelledby="trust-facts-title">
      <h2 id="trust-facts-title" tabindex="-1">Public trust facts</h2>
      <article>
        <span class="policy-grid__code">AGGREGATE-ONLY</span>
        <p>Reviewed study-scale facts and aggregate diagnostics only.</p>
      </article>
      <article>
        <span class="policy-grid__code">LOCAL ASSETS</span>
        <p>No external fonts, maps, or runtime data are loaded.</p>
      </article>
      <article>
        <span class="policy-grid__code">OPTIONAL LOCAL LOG</span>
        <p>A tab-only research log is off by default and records only after explicit opt-in. It never sends a request.</p>
      </article>
      <article>
        <span class="policy-grid__code">USER-ACTIVATED LINKS</span>
        <p>External attribution pages open only when a reader follows a link.</p>
      </article>
    </section>

    <nav class="in-page-nav" aria-labelledby="credits-page-nav-title">
      <p id="credits-page-nav-title">On this page</p>
      <ol class="content-section-nav">
        <li><a href="#credits-authorship-title" @click.prevent="navigateToSection">Authorship</a></li>
        <li><a href="#credits-sources-title" @click.prevent="navigateToSection">Sources &amp; rights</a></li>
        <li><a href="#credits-runtime-title" @click.prevent="navigateToSection">Runtime &amp; privacy</a></li>
        <li><a href="#credits-limits-title" @click.prevent="navigateToSection">Known limits</a></li>
        <li><a href="#credits-license-title" @click.prevent="navigateToSection">License</a></li>
        <li><a href="#credits-public-limit-title" @click.prevent="navigateToSection">Public limit</a></li>
      </ol>
    </nav>

    <section class="policy-grid" aria-labelledby="credits-authorship-title">
      <h2 id="credits-authorship-title" tabindex="-1">Authorship and collaboration</h2>
      <article>
        <span class="policy-grid__code">A / AUTHORSHIP</span>
        <h3>Personal portfolio edition</h3>
        <p>Qiushi Yu prepared this public, aggregate-only edition and its independent presentation.</p>
      </article>
      <article>
        <span class="policy-grid__code">B / COLLABORATION</span>
        <h3>Original practicum contribution</h3>
        <p>Zhiyuan Zhao is credited for collaborative visualization work and Stage 2–3 contributions in the original practicum project.</p>
      </article>
    </section>

    <section class="policy-grid" aria-labelledby="credits-sources-title">
      <h2 id="credits-sources-title" tabindex="-1">Sources, rights, and independent interpretation</h2>
      <article>
        <span class="policy-grid__code">C / SOURCE</span>
        <h3>ORNL EAGLE-I outages</h3>
        <p>
          Aggregated results independently processed from the
          <a href="https://figshare.com/articles/dataset/The_Environment_for_Analysis_of_Geo-Located_Energy_Information_s_Recorded_Electricity_Outages_2014-2022/24237376" target="_blank" rel="noopener noreferrer">ORNL EAGLE-I Recorded Electricity Outages dataset</a>,
          DOI 10.6084/m9.figshare.24237376, licensed
          <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noopener noreferrer">CC BY 4.0</a>.
        </p>
      </article>
      <article>
        <span class="policy-grid__code">D / NO ENDORSEMENT</span>
        <h3>Independent interpretation</h3>
        <p>Neither Oak Ridge National Laboratory nor the U.S. Department of Energy endorses this project or its conclusions.</p>
      </article>
    </section>

    <section class="policy-facts" aria-labelledby="credits-runtime-title">
      <h2 id="credits-runtime-title" tabindex="-1">Runtime and privacy</h2>
      <article>
        <span class="policy-grid__code">E / PUBLIC GRAIN</span>
        <h3>Aggregate-only public content</h3>
        <p>Public values are reviewed study-scale facts, broad event references, component states, and aggregate diagnostics. Fine-grained or reversible records stay outside the release.</p>
      </article>
      <article>
        <span class="policy-grid__code">F / RUNTIME</span>
        <h3>Local assets, optional tab-only research log</h3>
        <p>The application loads no external fonts, maps, analytics service, or runtime data. The optional research log uses only this tab, has no cookie or persistent identifier, and can be exported or cleared locally. Attribution links above are user-activated exits; the app does not request them in the background.</p>
      </article>
    </section>

    <section class="policy-facts" aria-labelledby="credits-limits-title">
      <h2 id="credits-limits-title" tabindex="-1">Known limits</h2>
      <article>
        <span class="policy-grid__code">G / KNOWN LIMITS</span>
        <h3>Technical checks are bounded</h3>
        <p>The public artifact does not reproduce every withheld upstream input. Technical browser checks do not establish real-user understanding or complete screen-reader, speech-input, switch-access, and multi-browser support.</p>
      </article>
    </section>

    <section class="license-ledger">
      <div>
        <p class="eyebrow"><span>License ledger</span> Separate grants</p>
        <h2 id="credits-license-title" tabindex="-1">MIT covers this edition’s original code and documentation.</h2>
      </div>
      <div>
        <p>It does not relicense third-party datasets, private analytical inputs, or withheld derivative material.</p>
        <p>The public app directory contains the full notices in <strong>LICENSE</strong>, <strong>CREDITS.md</strong>, <strong>DATA_POLICY.md</strong>, and <strong>THIRD_PARTY_NOTICES.md</strong>.</p>
      </div>
    </section>

    <aside class="withheld-panel">
      <span class="withheld-panel__stamp">PUBLIC LIMIT</span>
      <div>
        <h2 id="credits-public-limit-title" tabindex="-1">{{ DATA_BOUNDARY.status }}</h2>
        <p>The repository and production build are scanned against this boundary before release.</p>
      </div>
      <ul>
        <li v-for="item in DATA_BOUNDARY.excluded" :key="item">{{ item }}</li>
      </ul>
    </aside>
  </article>
</template>
