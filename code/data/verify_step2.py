# Cross-check: does the Step 2 batch reproduce the Step 1 single-ROI curve?
# Step 1 used plane-3, ARRAY INDEX 16 of session 409828_2018-11-06_14-02-59.
import warnings
import numpy as np
import pandas as pd
import pynwb
from pathlib import Path

warnings.filterwarnings('ignore')

SESSION = '409828_2018-11-06_14-02-59_filtered_2026-04-09_04-59-00'
NWB = f'/data/409828_V1DD_Filtered/{SESSION}/409828_2018-11-06_14-02-59.nwb.zarr'

# --- Re-run "Simplified code for Step 1" verbatim in spirit ---------------
nwb = pynwb.read_nwb(NWB)
stim_table = nwb.intervals['stimulus_table'].to_dataframe()
dg_stim = stim_table[stim_table['stim_name'] == 'drifting_gratings_full'].copy()

roi_idx = 16
dff_obj = nwb.processing['plane-3']['dff']
roi_dff = np.array(dff_obj.data[:, roi_idx])
timestamps = dff_obj.timestamps[:]
roi_id = int(dff_obj.rois.to_dataframe().reset_index(drop=True).iloc[roi_idx]['roi'])
print(f"Step 1's `roi = 16` is array index 16 -> roi identifier {roi_id}")

resp_start, resp_end = 0.7, 2.5
trial_responses = []
for _, row in dg_stim.iterrows():
    t_rel = timestamps - row['start_time']
    base_idx = np.where((t_rel >= -0.5) & (t_rel < 0.0))[0]
    baseline = float(np.nanmean(roi_dff[base_idx])) if len(base_idx) > 0 else 0.0
    resp_idx = np.where((t_rel >= resp_start) & (t_rel <= resp_end))[0]
    trial_responses.append(float(np.nanmean(roi_dff[resp_idx])) - baseline
                           if len(resp_idx) else np.nan)
dg_stim['response'] = trial_responses
print(f"Total dg trials: {len(dg_stim)}, "
      f"NaN direction: {int(dg_stim['direction'].isna().sum())}, "
      f"NaN response: {int(dg_stim['response'].isna().sum())}")

dg_stim = dg_stim.dropna(subset=['direction', 'response'])
dg_stim['orientation'] = dg_stim['direction'].round().astype(int) % 180
step1 = dg_stim.groupby('orientation')['response'].mean()

# --- Pull the same ROI out of the Step 2 batch output ---------------------
tune = pd.read_parquet('/scratch/v1dd_step2/tuning_curves_long.parquet')
sel = tune[(tune.session_name == SESSION) & (tune.plane == 3) & (tune.roi_idx == roi_idx)]
step2 = sel.set_index('orientation')['mean_response'].sort_index()

cmp = pd.DataFrame({'step1': step1, 'step2': step2})
cmp['abs_diff'] = (cmp.step1 - cmp.step2).abs()
print('\nStep 1 vs Step 2 tuning curve for this ROI:')
print(cmp.to_string())
print(f"\nmax |difference| = {cmp.abs_diff.max():.3e}")
assert cmp.abs_diff.max() < 1e-9, 'Batch does NOT reproduce the Step 1 curve'
print('PASS: batch reproduces the Step 1 single-ROI curve exactly.')

# --- Check the roi identifier vs array index really do differ ------------
roi_tbl = dff_obj.rois.to_dataframe().reset_index(drop=True)
n_mismatch = int((roi_tbl['roi'].values != np.arange(len(roi_tbl))).sum())
print(f"\nplane-3: {n_mismatch}/{len(roi_tbl)} ROIs where roi identifier != array index")
