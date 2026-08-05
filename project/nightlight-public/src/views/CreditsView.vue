<script setup>
import { DATA_BOUNDARY } from '../content/copy.js'
</script>

<template>
  <article class="page credits-page">
    <header class="page-heading page-heading--split">
      <div>
        <p class="eyebrow"><span>Field note 05</span> Credits / data policy</p>
        <h1>Credit the work.<br><em>Keep the line clear.</em></h1>
      </div>
      <p>
        Code licensing, contributor credit, source attribution, and data publication rights are separate.
        This page states each one plainly.
      </p>
    </header>

    <section class="policy-grid">
      <article>
        <span class="policy-grid__code">A / AUTHORSHIP</span>
        <h2>Personal portfolio edition</h2>
        <p>Qiushi Yu prepared this public, aggregate-only edition and its independent presentation.</p>
      </article>
      <article>
        <span class="policy-grid__code">B / COLLABORATION</span>
        <h2>Original practicum contribution</h2>
        <p>Zhiyuan Zhao is credited for collaborative visualization work and Stage 2–3 contributions in the original practicum project.</p>
      </article>
      <article>
        <span class="policy-grid__code">C / SOURCE</span>
        <h2>ORNL EAGLE-I outages</h2>
        <p>
          Aggregated results independently processed from the
          <a href="https://figshare.com/articles/dataset/The_Environment_for_Analysis_of_Geo-Located_Energy_Information_s_Recorded_Electricity_Outages_2014-2022/24237376" target="_blank" rel="noopener noreferrer">ORNL EAGLE-I Recorded Electricity Outages dataset</a>,
          DOI 10.6084/m9.figshare.24237376, licensed
          <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noopener noreferrer">CC BY 4.0</a>.
        </p>
      </article>
      <article>
        <span class="policy-grid__code">D / NO ENDORSEMENT</span>
        <h2>Independent interpretation</h2>
        <p>Neither Oak Ridge National Laboratory nor the U.S. Department of Energy endorses this project or its conclusions.</p>
      </article>
    </section>

    <section class="license-ledger">
      <div>
        <p class="eyebrow"><span>License ledger</span> Separate grants</p>
        <h2>MIT covers this edition’s original code and documentation.</h2>
      </div>
      <div>
        <p>It does not relicense third-party datasets, private analytical inputs, or withheld derivative material.</p>
        <p>Full notices are included in <strong>LICENSE</strong>, <strong>CREDITS.md</strong>, <strong>DATA_POLICY.md</strong>, and <strong>THIRD_PARTY_NOTICES.md</strong>.</p>
      </div>
    </section>

    <aside class="withheld-panel">
      <span class="withheld-panel__stamp">PUBLIC LIMIT</span>
      <div>
        <h2>{{ DATA_BOUNDARY.status }}</h2>
        <p>The repository and production build are scanned against this boundary before release.</p>
      </div>
      <ul>
        <li v-for="item in DATA_BOUNDARY.excluded" :key="item">{{ item }}</li>
      </ul>
    </aside>
  </article>
</template>
