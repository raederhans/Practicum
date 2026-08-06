export const STUDY_SUMMARY = Object.freeze({
  stage2: Object.freeze({ events: 25, jurisdictions: 17 }),
  stage3: Object.freeze({ observations: 1002, events: 22, states: 15 }),
  descriptiveModel: Object.freeze({ rSquared: 0.7603, adjustedRSquared: 0.7543, n: 977 }),
  descriptiveSensitivity: 0.551,
})

export const EVENTS = Object.freeze([
  { id: 'maria', name: 'Hurricane Maria', year: 2017, location: 'San Juan, Puerto Rico', region: 'Caribbean', type: 'Hurricane', hazardFamily: 'Tropical cyclone', center: [-66.1, 18.4] },
  { id: 'irma', name: 'Hurricane Irma', year: 2017, location: 'Miami, Florida', region: 'Southeast', type: 'Hurricane', hazardFamily: 'Tropical cyclone', center: [-80.2, 25.8] },
  { id: 'ida', name: 'Hurricane Ida', year: 2021, location: 'New Orleans, Louisiana', region: 'South', type: 'Hurricane', hazardFamily: 'Tropical cyclone', center: [-90.1, 30.0] },
  { id: 'laura', name: 'Hurricane Laura', year: 2020, location: 'Lake Charles, Louisiana', region: 'South', type: 'Hurricane', hazardFamily: 'Tropical cyclone', center: [-93.2, 30.2] },
  { id: 'michael', name: 'Hurricane Michael', year: 2018, location: 'Panama City, Florida', region: 'Southeast', type: 'Hurricane', hazardFamily: 'Tropical cyclone', center: [-85.6, 30.2] },
  { id: 'eq-pr', name: 'Puerto Rico Earthquake', year: 2020, location: 'San Juan, Puerto Rico', region: 'Caribbean', type: 'Earthquake', hazardFamily: 'Earthquake', center: [-66.1, 18.4] },
  { id: 'ian-charlotte', name: 'Hurricane Ian', year: 2022, location: 'Charlotte Harbor, Florida', region: 'Southeast', type: 'Hurricane', hazardFamily: 'Tropical cyclone', center: [-82.0, 26.9] },
  { id: 'ian-fortmyers', name: 'Hurricane Ian', year: 2022, location: 'Fort Myers, Florida', region: 'Southeast', type: 'Hurricane', hazardFamily: 'Tropical cyclone', center: [-81.9, 26.6] },
  { id: 'eq-hatay', name: 'Turkey Earthquake', year: 2023, location: 'Hatay, Turkey', region: 'International', type: 'Earthquake', hazardFamily: 'Earthquake', center: [36.2, 36.2] },
  { id: 'matthew-jax', name: 'Hurricane Matthew', year: 2016, location: 'Jacksonville, Florida', region: 'Southeast', type: 'Hurricane', hazardFamily: 'Tropical cyclone', center: [-81.7, 30.3] },
  { id: 'florence-wilm', name: 'Hurricane Florence', year: 2018, location: 'Wilmington, North Carolina', region: 'Southeast', type: 'Hurricane', hazardFamily: 'Tropical cyclone', center: [-78.0, 34.2] },
  { id: 'zeta-atlanta', name: 'Hurricane Zeta', year: 2020, location: 'Atlanta, Georgia', region: 'Southeast', type: 'Hurricane', hazardFamily: 'Tropical cyclone', center: [-84.4, 33.8] },
  { id: 'zeta-birmingham', name: 'Hurricane Zeta', year: 2020, location: 'Birmingham, Alabama', region: 'Southeast', type: 'Hurricane', hazardFamily: 'Tropical cyclone', center: [-86.8, 33.5] },
  { id: 'isaias-nj', name: 'Tropical Storm Isaias', year: 2020, location: 'Newark, New Jersey', region: 'Northeast', type: 'Tropical storm', hazardFamily: 'Tropical cyclone', center: [-74.2, 40.7] },
  { id: 'irma-savannah', name: 'Hurricane Irma', year: 2017, location: 'Savannah, Georgia', region: 'Southeast', type: 'Hurricane', hazardFamily: 'Tropical cyclone', center: [-81.1, 32.1] },
  { id: 'matthew-nc', name: 'Hurricane Matthew', year: 2016, location: 'Fayetteville, North Carolina', region: 'Southeast', type: 'Hurricane', hazardFamily: 'Tropical cyclone', center: [-78.9, 35.1] },
  { id: 'florence-sc', name: 'Hurricane Florence', year: 2018, location: 'Myrtle Beach, South Carolina', region: 'Southeast', type: 'Hurricane', hazardFamily: 'Tropical cyclone', center: [-78.9, 33.7] },
  { id: 'isaias-ny', name: 'Tropical Storm Isaias', year: 2020, location: 'Westchester, New York', region: 'Northeast', type: 'Tropical storm', hazardFamily: 'Tropical cyclone', center: [-73.8, 41.0] },
  { id: 'uri-houston', name: 'Winter Storm Uri', year: 2021, location: 'Houston, Texas', region: 'South', type: 'Winter storm', hazardFamily: 'Winter storm', center: [-95.4, 29.8] },
  { id: 'derecho-chicago', name: 'Derecho', year: 2020, location: 'Chicago, Illinois', region: 'Midwest', type: 'Derecho', hazardFamily: 'Severe convective storm', center: [-87.6, 41.9] },
  { id: 'severe-detroit', name: 'Severe Storms', year: 2019, location: 'Detroit, Michigan', region: 'Midwest', type: 'Severe storm', hazardFamily: 'Severe convective storm', center: [-83.0, 42.3] },
  { id: 'noreaster-boston', name: 'Nor\'easter', year: 2021, location: 'Boston, Massachusetts', region: 'Northeast', type: 'Winter storm', hazardFamily: 'Winter storm', center: [-71.1, 42.4] },
  { id: 'icestorm-okc', name: 'Ice Storm', year: 2020, location: 'Oklahoma City, Oklahoma', region: 'South', type: 'Ice storm', hazardFamily: 'Winter storm', center: [-97.5, 35.5] },
  { id: 'severe-nashville', name: 'Severe Storms', year: 2023, location: 'Nashville, Tennessee', region: 'South', type: 'Severe storm', hazardFamily: 'Severe convective storm', center: [-86.8, 36.2] },
  { id: 'atmos-seattle', name: 'Atmospheric River', year: 2022, location: 'Seattle, Washington', region: 'Pacific', type: 'Atmospheric river', hazardFamily: 'Atmospheric river', center: [-122.3, 47.6] },
])

export const HAZARD_FAMILIES = Object.freeze(['All', ...new Set(EVENTS.map((event) => event.hazardFamily))])
