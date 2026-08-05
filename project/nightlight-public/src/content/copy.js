export const FINDINGS_COPY = Object.freeze({
  model: Object.freeze({
    label: 'M1+ descriptive fit',
    interpretation: 'A compact summary of association within the analyzed sample—not a forecast and not a causal estimate.',
  }),
  sensitivity: Object.freeze({
    value: 0.551,
    label: 'Descriptive sensitivity',
    caution: 'This sensitivity is descriptive only. It is not causal evidence and not a fairness conclusion.',
  }),
})

export const DATA_BOUNDARY = Object.freeze({
  status: 'Fine-grained analytical layers are not published in this public edition.',
  excluded: Object.freeze([
    'raw outage records',
    'time-series extracts',
    'facility locations',
    'pixel-level surfaces',
    'reversible fine-grained tables',
  ]),
})
