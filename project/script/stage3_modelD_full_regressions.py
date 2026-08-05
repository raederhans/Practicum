"""
stage3_modelD_full_regressions.py
=================================
Run the full Stage 3 regression suite (M1, M1+, M3 SEM, M8, M8+) using
Model D's probability maps. Mirrors the structure of regression_results.json
so we can place Model A and Model D side-by-side.

Outputs:
    data/result/stage3/regression_results_modelD_full.json
"""

import argparse
import os, sys, json, glob


def _configure_console_output():
    """Keep CLI output writable when Windows pipes use a legacy code page."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(errors='backslashreplace')


_configure_console_output()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
RAW_DIR      = os.path.join(DATA_DIR, 'raw')
STAGE2_DIR   = os.path.join(DATA_DIR, 'result', 'stage2')
STAGE3_DIR   = os.path.join(DATA_DIR, 'result', 'stage3')

parser = argparse.ArgumentParser(description='Run the full Model D Stage 3 regressions.')
parser.add_argument("--raw-dir", default=RAW_DIR, help='Directory containing raw covariates.')
parser.add_argument("--input-dir", default=STAGE3_DIR, help='Directory containing the ZIP panel.')
parser.add_argument("--output-dir", default=STAGE3_DIR, help='Directory for regression results.')
args = parser.parse_args()
RAW_DIR = os.path.abspath(args.raw_dir)
INPUT_DIR = os.path.abspath(args.input_dir)
STAGE3_DIR = os.path.abspath(args.output_dir)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# ─── Load panels ──────────────────────────────────────────────────────
panel = pd.read_parquet(os.path.join(INPUT_DIR, 'zipcode_panel_modelD.parquet'))
panel['ZCTA5CE20'] = panel['ZCTA5CE20'].astype(str).str.zfill(5)
print(f"Panel: {len(panel)} ZIP-event obs, {panel['event_id'].nunique()} events")

# ─── Merge Census ACS controls ────────────────────────────────────────
acs = pd.read_csv(os.path.join(RAW_DIR, 'acs_zcta_2022.csv'))
acs['ZCTA5CE20'] = acs['ZCTA5CE20'].astype(str).str.zfill(5)
# pop density requires area; we have area in panel
panel = panel.merge(acs[['ZCTA5CE20', 'total_pop', 'median_income']], on='ZCTA5CE20', how='left')
panel['pop_density'] = panel['total_pop'] / panel['area_km2'].clip(lower=0.01)
panel['log_pop_density'] = np.log(panel['pop_density'].clip(lower=1))
panel['log_income']      = np.log(panel['median_income'].clip(lower=1000))

n_with_acs = panel[['log_pop_density', 'log_income']].notna().all(axis=1).sum()
print(f"After ACS merge: {n_with_acs}/{len(panel)} ZIPs have full Census controls")

# ─── Helpers ──────────────────────────────────────────────────────────
def event_dummies(df):
    return pd.get_dummies(pd.Categorical(df['event_id']).codes, prefix='ev', drop_first=True).astype(float)

def fit_ols(df, y_col, x_cols, name):
    df = df.copy()
    mask = df[[y_col] + x_cols].notna().all(axis=1)
    df = df[mask]
    fe = event_dummies(df)
    X = pd.concat([df[x_cols].astype(float).reset_index(drop=True), fe.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    y = df[y_col].astype(float).values
    groups = df['event_id'].astype(str).values
    m = sm.OLS(y, X).fit(
        cov_type='cluster',
        cov_kwds={'groups': groups, 'use_correction': True},
        use_t=True,
    )
    out = {
        'name': name,
        'unit_of_analysis': 'ZIP-event observation',
        'covariance': 'cluster-robust',
        'inference_distribution': 'Student t',
        'cluster_variable': 'event_id',
        'n_clusters': int(df['event_id'].nunique()),
        'n': int(len(df)),
        'r_squared': round(float(m.rsquared), 4),
        'adj_r_squared': round(float(m.rsquared_adj), 4),
        'coefs': {c: [round(float(m.params[c]), 4),
                      float(m.pvalues[c])] for c in x_cols},
    }
    print(f"\n[{name}]  n={out['n']}  R²={out['r_squared']}")
    for c in x_cols:
        print(f"   {c:<22s}  β={out['coefs'][c][0]:+.4f}  p={out['coefs'][c][1]:.4g}")
    return out


def build_event_block_knn(df, k=5):
    """Build kNN weights independently within each event block."""
    from libpysal.weights import KNN, W

    frame = df.reset_index(drop=True)
    neighbors = {i: [] for i in range(len(frame))}
    event_by_row = frame['event_id'].astype(str).to_dict()

    for event_id, event_rows in frame.groupby('event_id', sort=False):
        global_ids = event_rows.index.to_list()
        if len(global_ids) <= k:
            raise ValueError(f"Event {event_id!r} has {len(global_ids)} rows; k={k} requires at least {k + 1}")
        local_w = KNN.from_array(event_rows[['cx', 'cy']].to_numpy(), k=k)
        for local_source, local_targets in local_w.neighbors.items():
            source = global_ids[int(local_source)]
            neighbors[source] = [global_ids[int(target)] for target in local_targets]

    weights = W(neighbors, id_order=list(range(len(frame))), silence_warnings=True)
    weights.transform = 'r'
    cross_event_edges = sum(
        event_by_row[source] != event_by_row[target]
        for source, targets in weights.neighbors.items()
        for target in targets
    )
    metadata = {
        'event_blocks': int(frame['event_id'].nunique()),
        'components': int(weights.n_components),
        'cross_event_edges': int(cross_event_edges),
    }
    return weights, metadata

results = {}

# ─── M1 · OLS baseline (no Census, repeat for sanity) ─────────────────
results['m1'] = fit_ols(panel, 'mean_prob', ['fac_density'], 'M1 · OLS  mean_prob ~ fac_density + FE')

# ─── M1+ · OLS + Census controls ──────────────────────────────────────
results['m1_plus'] = fit_ols(panel, 'mean_prob',
                             ['fac_density', 'log_pop_density', 'log_income'],
                             'M1+ · OLS  + Census controls')

# ─── M5 · Hurricane subsample (no wind data unless we recompute; skip wind here) ──
# Use what we have: hurricane subset
hur = panel[panel['disaster_type'] == 'hurricane'].copy()
results['m5_hurricane'] = fit_ols(hur, 'mean_prob',
                                  ['fac_density'],
                                  'M5 · Hurricane subset  mean_prob ~ fac_density + FE')
results['m5_plus_hurricane'] = fit_ols(hur, 'mean_prob',
                                       ['fac_density', 'log_pop_density', 'log_income'],
                                       'M5+ · Hurricane subset + Census')

# ─── M3 · Spatial Error Model (KNN k=5) ───────────────────────────────
print('\n[M3 · SEM] attempting GM_Error_Het ...')
try:
    from libpysal.weights import KNN
    from spreg import GM_Error_Het
    # Need lat/lon for ZCTA centroids — pull from raw shapefile
    import geopandas as gpd
    zcta_dir = os.path.join(RAW_DIR, 'zcta520')
    shp = glob.glob(os.path.join(zcta_dir, '*.shp'))[0]
    zcta_gdf = gpd.read_file(shp).to_crs('EPSG:5070')   # CONUS Albers (m)
    zcta_gdf['ZCTA5CE20'] = zcta_gdf['ZCTA5CE20'].astype(str).str.zfill(5)
    zcta_gdf['cx'] = zcta_gdf.geometry.centroid.x
    zcta_gdf['cy'] = zcta_gdf.geometry.centroid.y
    panel_xy = panel.merge(zcta_gdf[['ZCTA5CE20', 'cx', 'cy']], on='ZCTA5CE20', how='left')
    sub = panel_xy.dropna(subset=['mean_prob', 'fac_density', 'cx', 'cy']).reset_index(drop=True)
    w, spatial_diagnostics = build_event_block_knn(sub, k=5)
    y = sub[['mean_prob']].values
    fe = event_dummies(sub).reset_index(drop=True)
    feature_names = ['fac_density'] + fe.columns.to_list()
    X = pd.concat([sub[['fac_density']].astype(float), fe], axis=1).to_numpy()
    sem = GM_Error_Het(y, X, w=w, name_y='mean_prob', name_x=feature_names)
    # GM_Error_Het.betas = [intercept, fac_density, lambda]  (lambda is last)
    results['m3_sem'] = {
        'name': 'M3 · Spatial Error Model (GM_Error_Het, KNN k=5)',
        'status': 'diagnostic-only',
        'type': 'Spatial Error Model',
        'unit_of_analysis': 'ZIP-event observation',
        'spatial_weights': 'event-block kNN',
        'k': 5,
        'n': int(len(sub)),
        'event_blocks': spatial_diagnostics['event_blocks'],
        'components': spatial_diagnostics['components'],
        'cross_event_edges': spatial_diagnostics['cross_event_edges'],
        'features': feature_names,
        'covariates': ['fac_density'],
        'lambda': round(float(sem.betas[-1, 0]), 4),
        'intercept': round(float(sem.betas[0, 0]), 4),
        'fac_density_coef': round(float(sem.betas[1, 0]), 4),
        'fac_density_p': float(sem.z_stat[1][1]),
        'pseudo_r2': round(float(sem.pr2), 4),
    }
    print(f"   λ={results['m3_sem']['lambda']}  fac={results['m3_sem']['fac_density_coef']}"
          f"  p={results['m3_sem']['fac_density_p']}  pseudoR²={results['m3_sem']['pseudo_r2']}")
except ImportError as e:
    print(f"   [SKIP] {e}")
    results['m3_sem'] = {'error': str(e)}

# ─── M8 · Closed-loop validation (severity ~ mean_prob) ───────────────
# Requires loading EAGLE-I severity per zip — reuse logic from original script
print('\n[M8 · severity] computing EAGLE-I severity per ZIP ...')
EVENTS_META = {
    'Irma_Miami':            {'state': 'Florida',          'landfall': '2017-09-10'},
    'Ida_NewOrleans':        {'state': 'Louisiana',        'landfall': '2021-08-29'},
    'Laura_LakeCharles':     {'state': 'Louisiana',        'landfall': '2020-08-27'},
    'Michael_PanamaCity':    {'state': 'Florida',          'landfall': '2018-10-10'},
    'Ian_CharlotteHarbor':   {'state': 'Florida',          'landfall': '2022-09-28'},
    'Ian_FortMyers':         {'state': 'Florida',          'landfall': '2022-09-28'},
    'Florence_Wilmington':   {'state': 'North Carolina',   'landfall': '2018-09-14'},
    'Irma_Savannah':         {'state': 'Georgia',          'landfall': '2017-09-11'},
    'Isaias_Newark':         {'state': 'New Jersey',       'landfall': '2020-08-04'},
    'Matthew_Jacksonville':  {'state': 'Florida',          'landfall': '2016-10-07'},
    'Zeta_Atlanta':          {'state': 'Georgia',          'landfall': '2020-10-29'},
    'Zeta_Birmingham':       {'state': 'Alabama',          'landfall': '2020-10-29'},
    'Matthew_Fayetteville':  {'state': 'North Carolina',   'landfall': '2016-10-08'},
    'Florence_MyrtleBeach':  {'state': 'South Carolina',   'landfall': '2018-09-14'},
    'Isaias_Westchester':    {'state': 'New York',         'landfall': '2020-08-04'},
    'Uri_Houston':           {'state': 'Texas',            'landfall': '2021-02-15'},
    'Derecho_Chicago':       {'state': 'Illinois',         'landfall': '2020-08-10'},
    'Severe_Detroit':        {'state': 'Michigan',         'landfall': '2019-07-20'},
    'Noreaster_Boston':      {'state': 'Massachusetts',    'landfall': '2021-10-27'},
    'IceStorm_OKC':          {'state': 'Oklahoma',         'landfall': '2020-10-27'},
    'Severe_Nashville':      {'state': 'Tennessee',        'landfall': '2023-07-18'},
    'Atmos_Seattle':         {'state': 'Washington',       'landfall': '2022-11-04'},
}

eagle_dir = os.path.join(RAW_DIR, 'Outage_Dataset_R1')
eagle_dfs = []
for yr in range(2014, 2024):
    p = os.path.join(eagle_dir, f'eaglei_outages_with_events_{yr}.csv')
    if os.path.exists(p):
        eagle_dfs.append(pd.read_csv(p, low_memory=False))
eagle = pd.concat(eagle_dfs, ignore_index=True)
eagle['event_began'] = pd.to_datetime(eagle['Datetime Event Began'], errors='coerce')
print(f"   EAGLE-I loaded: {len(eagle):,} records")

severities = []
for eid, meta in EVENTS_META.items():
    target = pd.Timestamp(meta['landfall'])
    mask = (
        eagle['state_event'].str.contains(meta['state'], case=False, na=False) &
        ((eagle['event_began'] - target).abs().dt.days <= 14)
    )
    sub = eagle[mask]
    if sub.empty: continue
    cs = sub.groupby('fips').agg(total_customers=('max_customers', 'sum'),
                                  mean_duration=('duration', 'mean')).reset_index()
    cs['severity_county'] = np.log1p(cs['total_customers']) * cs['mean_duration']
    cs['event_id'] = eid
    # Simple county-level mean — assign all ZIPs in a county the same severity
    severities.append(cs[['event_id', 'fips', 'severity_county']])

sev = pd.concat(severities, ignore_index=True) if severities else pd.DataFrame()
print(f"   severity rows: {len(sev)}")

# Need to map ZIP → county. Use ZCTA centroid in counties shapefile.
import geopandas as gpd
counties_dir = os.path.join(RAW_DIR, 'counties')
shp = glob.glob(os.path.join(counties_dir, '*.shp'))
if shp:
    cnty = gpd.read_file(shp[0]).to_crs('EPSG:4326')
    cnty['fips'] = (cnty['STATEFP'].astype(str) + cnty['COUNTYFP'].astype(str)).astype(int)
    zcta_dir = os.path.join(RAW_DIR, 'zcta520')
    zcta_shp = glob.glob(os.path.join(zcta_dir, '*.shp'))[0]
    zcta_gdf = gpd.read_file(zcta_shp).to_crs('EPSG:4326')
    zcta_gdf['ZCTA5CE20'] = zcta_gdf['ZCTA5CE20'].astype(str).str.zfill(5)
    zcta_projected = zcta_gdf.to_crs('EPSG:5070')
    zcta_gdf['centroid'] = zcta_projected.geometry.centroid.to_crs('EPSG:4326')
    centroids = zcta_gdf.set_geometry('centroid')[['ZCTA5CE20', 'centroid']].rename(columns={'centroid': 'geometry'}).set_geometry('geometry')
    z2c = gpd.sjoin(centroids, cnty[['fips', 'geometry']], how='left', predicate='within')
    z2c = z2c[['ZCTA5CE20', 'fips']].drop_duplicates(subset='ZCTA5CE20')
    panel_sev = panel.merge(z2c, on='ZCTA5CE20', how='left').merge(sev, on=['event_id', 'fips'], how='left')
    panel_sev = panel_sev.dropna(subset=['severity_county'])
    print(f"   ZIPs with severity: {len(panel_sev)} / {len(panel)}")

    # M8 · severity ~ mean_prob + FE
    results['m8'] = fit_ols(panel_sev, 'severity_county',
                            ['mean_prob'],
                            'M8 · severity ~ mean_prob + FE')
    # M8+ · + Census
    panel_sev = panel_sev.dropna(subset=['log_pop_density', 'log_income'])
    results['m8_plus'] = fit_ols(panel_sev, 'severity_county',
                                 ['mean_prob', 'log_pop_density', 'log_income'],
                                 'M8+ · severity ~ mean_prob + Census + FE')
else:
    print('   [SKIP] county shapefile not found')

# ─── Save ─────────────────────────────────────────────────────────────
out = os.path.join(STAGE3_DIR, 'regression_results_modelD_full.json')
os.makedirs(STAGE3_DIR, exist_ok=True)
with open(out, 'w') as f:
    json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
print(f"\nSaved: {out}")
