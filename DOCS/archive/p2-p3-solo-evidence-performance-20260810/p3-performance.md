# P3 Headed Throttled Performance Evidence

## Decision

Runtime evidence supports one retained product change: defer creation of the below-fold `map_preview.png` image until its existing image container intersects the viewport. The existing map-card text and real `RouterLink` remain in the document from first render. Native `loading="lazy"` by itself was measured and rejected as a no-op because Chromium still fetched the image immediately.

This is a local controlled-browser result, not field performance. It does not establish performance on other browsers, devices, networks, geographic locations, Pages/CDN delivery, or user hardware.

## Fixed target and protocol

| Field | Value |
| --- | --- |
| Repository base | `ca8292040a402eae1d2e461708a4cc912867efcb` plus the uncommitted P3 instrumentation/candidate listed in the handoff |
| Date | 2026-08-10, Asia/Singapore |
| Browser | One headed Chromium `151.0.7922.76`; Playwright session `practicum-p3-throttled-20260810` |
| Viewport | 1365 × 768 CSS pixels, device pixel ratio 1 |
| Target | Local Vite production preview, `http://127.0.0.1:43242/Practicum/` |
| Network | CDP `Network.emulateNetworkConditions`: 150 ms latency, 200,000 B/s download, 93,750 B/s upload, `cellular4g` |
| Samples | 7 retained samples per scenario; no post-hoc sample deletion |
| Cold | `Network.clearBrowserCache` before the measured navigation |
| Warm | Cache clear, one unmeasured warm-up of the same route/phase, then the measured navigation |
| Summary | Median plus nearest-rank p95; IQR/min/max retained for noise interpretation |
| Home success | `.hero__title` attached; preview load is a separately observed image load, not the home success condition |
| Map canvas signal | `.maplibregl-canvas` DOM attachment; construction only, explicitly not ready |
| Map ready signal | `data-map-ready=true`, set synchronously inside the active MapLibre `load` handler |
| External settle | No tracked HTTP(S) request in flight for 750 ms, with a 15,000 ms ceiling; failed URLs remain counted |

The probe initially omitted attribute observation, so its first map run waited successfully for `[data-map-ready=true]` but recorded a null timestamp. That defective timestamp set is not used. The observer was regression-tested with `attributeFilter: ['data-map-ready']`, SPA signals were reset, pre-phase timestamps were rejected, and all four map scenarios were rerun.

## Home preview result

`public/map_preview.png` is 885,131 bytes on disk. The original implementation created the image immediately even though the map card is below the first viewport.

| Variant / cache | n | Hero median / p95 (ms) | Preview loaded median / p95 (ms) | Network settle median / p95 (ms) | Median measured resource transfer | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Original eager, cold | 7 | 975.2 / 990.7 | 5,825.2 / 5,860.4 | 6,843 / 6,861 | 955,820 B | Baseline: preview starts at approximately hero attachment and dominates cold transfer |
| Original eager, warm | 7 | 547.9 / 561.4 | 722.8 / 746.0 | 1,580 / 1,596 | 1,800 B | Browser-cache case; preview still requested |
| Native `loading=lazy`, cold | 7 | 966.8 / 996.3 | 5,828.2 / 5,874.4 | 6,859 / 6,871 | 955,836 B | Measured no-op; 7/7 still transferred 885,431 B for the image |
| Native `loading=lazy`, warm | 7 | 535.0 / 557.1 | 707.2 / 744.4 | 1,579 / 1,589 | 1,800 B | Measured no-op |
| Viewport-intersection candidate, cold | 7 | 969.4 / 2,990.3 | not requested | 2,390 / 6,733 | 70,529 B | Retained: all 7 samples avoided preview transfer before scroll |
| Viewport-intersection candidate, warm | 7 | 537.2 / 539.2 | not requested | 1,564 / 1,572 | 1,500 B | Retained |

Against the original cold median, the retained candidate reduces pre-scroll measured transfer by about 885,291 B (92.6%) and network-settle median by 4,453 ms (65.1%). It does not support a p95 improvement claim: one candidate cold sample settled at 6,733 ms and made nearest-rank p95 6,733 ms. The other six cold samples settled between 2,357 and 2,406 ms; cold IQR was 27 ms. The outlier transferred the same 70,529 B and made no preview request, so it is retained as environmental/runtime noise rather than removed or attributed to the image.

### Behavior preservation smoke

The final candidate was checked in the same headed browser under the throttle:

- after the home hero had been present for one second, there was no preview image node and no `map_preview.png` resource entry;
- the map card was still an `A` element with `href="#/map"` and its full explanatory text;
- scrolling the card into view created the image and loaded it successfully (`naturalWidth = 958`);
- the real link accepted focus, Enter navigated to `#/map`, and the map reached `data-map-ready=true`.

This supports only fetch deferral and preservation of the exercised link/keyboard behavior. It is not a usability or accessibility-conformance result.

## MapLibre ready and external settle

The table reports the corrected rerun only.

| Scenario | n | Canvas median / p95 (ms) | MapLibre ready median / p95 (ms) | External settle median / p95 (ms) | MapLibre transfer median | Total measured resource transfer median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct cold | 7 | 2,375.7 / 2,421.4 | 5,637.3 / 6,138.4 | 13,520 / 14,022 | 218,171 B | 296,928 B |
| Direct warm | 7 | 593.7 / 604.7 | 728.7 / 747.1 | 4,379 / 4,414 | 300 B | 2,400 B |
| Home → Map SPA cold | 7 | 1,382.6 / 1,388.0 | 4,606.3 / 4,685.1 | 12,526 / 12,537 | 218,171 B | 232,500 B |
| Home → Map SPA warm | 7 | 17.7 / 21.5 | 121.3 / 131.0 | 3,792 / 3,805 | already cached; no phase resource entry | 300 B |

All scenarios produced 7/7 samples, no probe exception, no browser warning/error during the corrected rerun, and no 15-second network-settle timeout. Canvas attachment was 104–3,803 ms earlier than MapLibre ready depending on cache/navigation, confirming that canvas construction cannot substitute for ready.

External settle is deliberately separate. Requests included `basemaps.cartocdn.com`, `tiles.basemaps.cartocdn.com`, `fonts.googleapis.com`, and `unpkg.com`. Most map samples recorded 20 failed external URLs while still reaching the MapLibre load signal; two direct-warm samples had zero failed URLs and settled near 1.5 seconds. This variability is why the 4.38-second warm-direct median and 13.52-second cold-direct settle must be read as a CARTO/external-network proxy, not application-ready time or guaranteed production behavior.

## Bundle context, not runtime conclusion

`npm run analyze:bundle` reported MapLibre at 803,051 B raw / 217,871 B gzip and isolated from the home initial route. The final candidate's home-initial route was 276,000 B raw / 67,669 B gzip; direct map was 1,107,420 B raw / 293,170 B gzip; incremental map-after-home was 847,342 B raw / 230,698 B gzip. These sizes explain likely transfer pressure but were not used as substitutes for the headed runtime decision.

## Sample ledger

The auditable per-sample subset used for the retained comparisons is in `measurements.csv`. Raw Playwright output was task-owned temporary evidence; it is summarized here and removed during cleanup. The CSV retains every corrected map sample and every home no-op/final-candidate sample.

## Claim boundary

Allowed: under this exact local Chromium/CDP protocol, the original/native-lazy home fetched the 885 KB preview before scroll, explicit viewport deferral avoided that transfer in 7/7 samples, reduced median pre-scroll transfer/settle, and preserved the exercised real-link behavior.

Not allowed: field-performance improvement, p95 improvement, universal browser behavior, production CDN/Pages performance, user-perceived speed, accessibility validation, MapLibre/CARTO reliability, or proof that external failures are caused by repository code.
