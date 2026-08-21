
# === Step 1: Orientation Tuning Curve ===
# Session:  409828_2018-11-06_14-02-59
# Stimulus: drifting_gratings_full
# Response: mean dF/F in a data-driven window centered on the PSTH peak, baseline-corrected
# TF / SF:  averaged across all conditions
# Output:   orientation tuning curve (direction % 180 collapses opposite directions)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hdmf_zarr import NWBZarrIO
from pathlib import Path

# ── 1. Load NWB file ──────────────────────────────────────────────────────────
session_name = '409828_2018-11-06_14-02-59_filtered_2026-04-09_04-59-00'
nwb_filename = '409828_2018-11-06_14-02-59.nwb.zarr'
nwb_path     = str(Path('/data/409828_V1DD_Filtered') / session_name / nwb_filename)

io      = NWBZarrIO(nwb_path, mode='r')
nwbfile = io.read()

# ── 2. Extract epoch and stimulus tables ──────────────────────────────────────
# epoch_table: one row per stimulus block (stim_name, start_time, stop_time)
# stim_table:  one row per individual trial (start_time, stop_time, direction, TF, SF, ...)
epoch_table = nwbfile.intervals['epochs'].to_dataframe()
stim_table  = nwbfile.intervals['stimulus_table'].to_dataframe()

dg_stim = stim_table[stim_table['stim_name'] == 'drifting_gratings_full'].copy()
print(f"drifting_gratings_full — {len(dg_stim)} trials")
print(f"  Directions (deg): {sorted(dg_stim['direction'].dropna().unique())}")
print(f"  TF (Hz):          {sorted(dg_stim['temporal_frequency'].dropna().unique())}")
print(f"  SF (cpd):         {sorted(dg_stim['spatial_frequency'].dropna().unique())}")

# Epoch-level time spans for ROI responsiveness scoring
dg_epochs    = epoch_table[epoch_table['stim_name'] == 'drifting_gratings_full']
spont_epochs = epoch_table[epoch_table['stim_name'] == 'spontaneous']

# ── 3. Select the most responsive soma across all imaging planes ──────────────
# SNR = mean(dF/F during grating epochs) / std(dF/F during spontaneous epochs)
# Higher SNR → more grating-driven activity relative to baseline fluctuations.
best_snr      = -np.inf
best_roi_info = None

for plane_name in [f'plane-{i}' for i in range(6)]:
    try:
        dff_obj   = nwbfile.processing[plane_name]['dff']
        ts        = dff_obj.timestamps[:]   # imaging frame timestamps (s); zero = session start
        roi_table = dff_obj.rois.to_dataframe()

        # Boolean frame masks for each condition
        grating_mask = np.zeros(len(ts), dtype=bool)
        for _, row in dg_epochs.iterrows():
            grating_mask |= (ts >= row['start_time']) & (ts <= row['stop_time'])

        spont_mask = np.zeros(len(ts), dtype=bool)
        for _, row in spont_epochs.iterrows():
            spont_mask |= (ts >= row['start_time']) & (ts <= row['stop_time'])

        # Fall back to all non-grating frames if no spontaneous epoch recorded
        if not spont_mask.any():
            spont_mask = ~grating_mask
            print(f"  {plane_name}: no spontaneous epoch — using non-grating frames as baseline")

        if not grating_mask.any():
            print(f"  {plane_name}: no grating frames — skipping")
            continue

        # Load full dF/F matrix for this plane, score ROIs, then free memory
        dff_all      = dff_obj.data[:]                 # (nframes, n_rois)
        grating_mean = dff_all[grating_mask].mean(axis=0)
        spont_std    = dff_all[spont_mask].std(axis=0)
        spont_std[spont_std < 1e-6] = 1e-6            # guard against /0
        snr          = grating_mean / spont_std
        best_idx     = int(np.argmax(snr))
        del dff_all                                    # release memory before next plane

        roi_id_best = int(roi_table.iloc[best_idx]['roi'])
        print(f"  {plane_name}: {len(roi_table):3d} ROIs | "
              f"best SNR = {snr[best_idx]:.2f}  (roi_id = {roi_id_best})")

        if snr[best_idx] > best_snr:
            best_snr      = float(snr[best_idx])
            best_roi_info = dict(
                plane      = plane_name,
                array_idx  = best_idx,
                roi_id     = roi_id_best,
                snr        = best_snr,
                timestamps = ts,
            )

    except Exception as e:
        print(f"  {plane_name}: {e}")

print(f"\n→ Using ROI {best_roi_info['roi_id']} "
      f"({best_roi_info['plane']}, SNR = {best_roi_info['snr']:.2f})")

# ── 4. Load single-ROI dF/F trace ────────────────────────────────────────────
dff_obj_sel = nwbfile.processing[best_roi_info['plane']]['dff']
roi_dff     = np.array(dff_obj_sel.data[:, best_roi_info['array_idx']])  # (nframes,)
timestamps  = best_roi_info['timestamps']

# ── 5. Build average PSTH → determine data-driven response window ─────────────
# Align every trial to its onset, interpolate to a common time grid, then average.
# The window is placed at ±0.5 s around the PSTH peak, which automatically
# absorbs the ~0.5–1 s calcium indicator lag.
stim_dur = float((dg_stim['stop_time'] - dg_stim['start_time']).median())
pre_win  = 1.0   # s of pre-stimulus context (for display and baseline)
post_win = 2.0   # s after offset to capture GCaMP decay

dt     = float(np.median(np.diff(timestamps)))         # inter-frame interval (s)
t_grid = np.arange(-pre_win, stim_dur + post_win, dt)  # common time axis

psth_matrix = np.full((len(dg_stim), len(t_grid)), np.nan)
for i, (_, row) in enumerate(dg_stim.iterrows()):
    onset  = row['start_time']
    t_rel  = timestamps - onset
    in_win = (t_rel >= -pre_win) & (t_rel <= stim_dur + post_win)
    idx    = np.where(in_win)[0]
    if len(idx) < 2:
        continue
    # Interpolate to common grid so misaligned frames can be averaged across trials
    psth_matrix[i] = np.interp(t_grid, t_rel[idx], roi_dff[idx])

mean_psth = np.nanmean(psth_matrix, axis=0)
sem_psth  = (np.nanstd(psth_matrix, axis=0)
             / np.sqrt(np.sum(~np.isnan(psth_matrix), axis=0).clip(1)))

# Locate peak within [0, stim_dur + 1 s] (the extra 1 s allows for calcium lag)
in_stim    = (t_grid >= 0) & (t_grid <= stim_dur + 1.0)
peak_lat   = float(t_grid[in_stim][np.argmax(mean_psth[in_stim])])
half_win   = 0.5
resp_start = max(0.0,            peak_lat - half_win)
resp_end   = min(stim_dur + 1.0, peak_lat + half_win)

print(f"PSTH peak latency : {peak_lat:.2f} s after onset")
print(f"Response window   : [{resp_start:.2f} – {resp_end:.2f}] s relative to onset")

# ── 6. Compute per-trial responses (window mean minus pre-stim baseline) ──────
trial_responses = []
for _, row in dg_stim.iterrows():
    onset  = row['start_time']
    t_rel  = timestamps - onset

    # 0.5 s pre-stimulus window for baseline estimate
    base_idx = np.where((t_rel >= -0.5) & (t_rel < 0.0))[0]
    baseline = float(np.nanmean(roi_dff[base_idx])) if len(base_idx) > 0 else 0.0

    resp_idx = np.where((t_rel >= resp_start) & (t_rel <= resp_end))[0]
    if len(resp_idx) == 0:
        trial_responses.append(np.nan)
    else:
        trial_responses.append(float(np.nanmean(roi_dff[resp_idx])) - baseline)

dg_stim = dg_stim.copy()
dg_stim['response'] = trial_responses
print(f"Missing responses : {dg_stim['response'].isna().sum()} / {len(dg_stim)}")

# ── 7. Orientation tuning: collapse direction → orientation (direction % 180) ─
# Gratings moving in opposite directions (e.g. 0° and 180°) share the same
# spatial orientation.  Averaging them gives orientation selectivity rather
# than direction selectivity.  All TF and SF conditions are pooled.
dg_stim = dg_stim.dropna(subset=['direction', 'response'])
dg_stim['orientation'] = (dg_stim['direction'].round().astype(int) % 180)

tuning = (
    dg_stim.groupby('orientation')['response']
    .agg(
        mean_response = 'mean',
        sem           = lambda x: x.std(ddof=1) / np.sqrt(x.count()),
        n_trials      = 'count',
    )
    .reset_index()
)
print("\nOrientation tuning table:")
print(tuning.to_string(index=False))

# ── 8. Visualize ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 4))

# 8a — Average PSTH with response window annotated
ax1 = fig.add_subplot(1, 3, 1)
ax1.axvspan(0,          stim_dur,  alpha=0.10, color='royalblue', label='Stimulus')
ax1.axvspan(resp_start, resp_end,  alpha=0.25, color='orange',    label='Response window')
ax1.fill_between(t_grid, mean_psth - sem_psth, mean_psth + sem_psth,
                 color='steelblue', alpha=0.25)
ax1.plot(t_grid, mean_psth, color='steelblue', lw=1.8)
ax1.axvline(0, color='royalblue', ls='--', lw=0.8, alpha=0.6)
ax1.axhline(0, color='gray',      ls=':',  lw=0.7)
ax1.set_xlabel('Time from onset (s)')
ax1.set_ylabel('Mean dF/F')
ax1.set_title(f"PSTH  –  ROI {best_roi_info['roi_id']} ({best_roi_info['plane']})")
ax1.legend(fontsize=7)

# 8b — Cartesian orientation tuning curve (mean ± SEM)
ax2 = fig.add_subplot(1, 3, 2)
oris  = tuning['orientation'].values
means = tuning['mean_response'].values
sems  = tuning['sem'].values
ax2.errorbar(oris, means, yerr=sems, fmt='o-', capsize=4,
             color='steelblue', ecolor='steelblue', lw=1.8, ms=5)
ax2.axhline(0, color='gray', ls=':', lw=0.7)
ax2.set_xticks(oris)
ax2.set_xlabel('Orientation (°)')
ax2.set_ylabel('Mean dF/F (baseline-corrected)')
ax2.set_title(f"Orientation Tuning  –  ROI {best_roi_info['roi_id']}")

# 8c — Polar orientation tuning (mirrored to full 360° circle)
# The 0–180° curve is duplicated at 180–360° to give a symmetric polar display.
ax3 = fig.add_subplot(1, 3, 3, projection='polar')
angles_h   = np.deg2rad(oris)
angles_all = np.concatenate([angles_h, angles_h + np.pi])
resp_all   = np.tile(means, 2)
loop_ang   = np.append(angles_all, angles_all[0])   # close the loop
loop_resp  = np.append(resp_all,   resp_all[0])
# Shift so all polar radii >= 0 (matplotlib polar axes require non-negative radii)
offset = max(0.0, -loop_resp.min())
ax3.plot(loop_ang, loop_resp + offset, 'o-', color='steelblue', lw=1.8, ms=4)
ax3.fill(loop_ang, loop_resp + offset, alpha=0.2, color='steelblue')
ax3.set_theta_zero_location('E')   # 0 deg points East (rightward grating)
ax3.set_theta_direction(-1)        # clockwise (standard visual field convention)
ax3.set_title(f"Polar  –  ROI {best_roi_info['roi_id']}", pad=15)
if offset > 0:
    ax3.text(0.5, -0.08, f'(radii offset +{offset:.3f})',
             transform=ax3.transAxes, ha='center', fontsize=7, color='gray')

plt.suptitle(
    f"Session 409828_2018-11-06  |  drifting_gratings_full  |  "
    f"ROI {best_roi_info['roi_id']} ({best_roi_info['plane']})",
    fontsize=9, y=1.01,
)
plt.tight_layout()
plt.show()