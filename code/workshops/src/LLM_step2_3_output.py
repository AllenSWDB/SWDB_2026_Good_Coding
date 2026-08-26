# GENERATED CODE - DO NOT RUN (VERY SLOW)

# === Step 2: Batch orientation tuning across all ROIs and all sessions ===
# Generalizes "Simplified code for Step 1" from one ROI in one session to every
# ROI in every session, and audits data completeness along the way.
#
# Four things had to change from Step 1 to be correct and survivable at scale:
#   * Step 1 used `roi = 16` as a COLUMN INDEX into the dF/F matrix. The `roi`
#     field in the ROI table is a separate global identifier -- the one used by
#     coregistration_1196.feather. Both are kept here: `roi_idx` (array column)
#     and `roi` (identifier), alongside column/volume/plane, which together form
#     the key that links back to the EM tables in Step 4.
#   * Sessions are stored in two backends (.nwb.zarr and plain .nwb/HDF5), so we
#     use pynwb.read_nwb(), which dispatches on the file type, rather than
#     NWBZarrIO, which reads only the zarr stores.
#   * Every failure (missing session, plane, stimulus, or trial) is recorded
#     rather than raised, so one bad session cannot abort the batch and the
#     completeness report reflects what actually happened.
#   * Each session is CHECKPOINTED to /scratch as soon as it finishes, so a
#     crash costs one session rather than the whole ~45 min of NWB reading.
#
# What gets checkpointed is ONE ROW PER ROI-SESSION, holding mean / sd / n for
# every orientation and every direction. Individual trials are reduced on the
# fly and not stored: mean, sd and n are sufficient statistics for everything
# Steps 3-5 need -- tuning curves, error bars, OSI/DSI, tuning width, and a
# one-way ANOVA across orientations for responsiveness gating (the between- and
# within-group sums of squares are exactly recoverable from mean/sd/n).
# Direction is kept alongside orientation because `direction % 180` is lossy and
# DSI needs it. Storing raw trials would only add bootstrap CIs, split-half
# reliability, noise correlations, and per-TF/SF re-analysis -- none of which
# these steps use. Note that changing RESP_START/RESP_END still requires
# re-reading the NWB files either way.

import json
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pynwb

warnings.filterwarnings('ignore')

# ── Configuration (same response windows as "Simplified code for Step 1") ─────
DATA_ROOT = Path('/data/409828_V1DD_Filtered')
OUT_DIR   = Path('/scratch/v1dd_step2')
SHARD_DIR = OUT_DIR / 'shards'          # per-session checkpoints
STIM_NAME = 'drifting_gratings_full'
RESP_START, RESP_END = 0.7, 2.5    # s after trial onset
BASE_START, BASE_END = -0.5, 0.0   # s relative to trial onset
DFF_KEY    = 'dff'
MAX_PLANES = 8                     # probe plane-0 .. plane-7; only existing ones are used
RESUME     = True                  # skip sessions already checkpointed;
                                   # delete SHARD_DIR to force a full recompute

for sub in ('tuning', 'meta'):
    (SHARD_DIR / sub).mkdir(parents=True, exist_ok=True)

KEY = ['session_name', 'column', 'volume', 'plane', 'roi', 'roi_idx']


def find_nwb_path(session_dir):
    """Return the NWB store inside a session folder (zarr or HDF5), or None."""
    hits = sorted(list(session_dir.glob('*.nwb.zarr')) + list(session_dir.glob('*.nwb')))
    return hits[0] if hits else None


def trial_window_indices(timestamps, onsets, t0, t1):
    """Frame index bounds [lo, hi) covering [onset+t0, onset+t1] for each trial.

    Timestamps are monotonically increasing, so searchsorted is exact and far
    cheaper than the per-trial boolean mask used in Step 1."""
    lo = np.searchsorted(timestamps, onsets + t0, side='left')
    hi = np.searchsorted(timestamps, onsets + t1, side='right')
    return lo, hi


def window_means(dff, lo, hi):
    """Per-trial, per-ROI nanmean of dF/F over frame windows [lo, hi).

    Returns (n_trials, n_rois); NaN where a trial had no frames or only NaNs."""
    out = np.full((len(lo), dff.shape[1]), np.nan)
    for i in range(len(lo)):
        if hi[i] > lo[i]:
            seg = dff[lo[i]:hi[i]]
            # An all-NaN column stays NaN -- recorded as missing, never as 0
            out[i] = np.nanmean(seg, axis=0) if np.isfinite(seg).any() else np.nan
    return out


def condition_stats(trial_resp, levels, prefix):
    """Reduce (n_trials, n_rois) responses to mean/sd/n per stimulus level.

    Returns a dict of column name -> length-n_rois array, so each ROI ends up as
    a single row. `levels` is the per-trial stimulus value (orientation or
    direction); trials are pooled across TF and SF, as chosen in Step 1."""
    cols = {}
    for lev in np.unique(levels):
        sel = trial_resp[levels == lev]                      # (n_repeats, n_rois)
        n = np.isfinite(sel).sum(axis=0)
        with np.errstate(invalid='ignore'):
            cols[f'{prefix}_mean_{lev:g}'] = np.nanmean(sel, axis=0)
            # ddof=1 is undefined for n<2; leave those as NaN rather than 0
            cols[f'{prefix}_sd_{lev:g}'] = np.where(n > 1, np.nanstd(sel, axis=0, ddof=1), np.nan)
        cols[f'{prefix}_n_{lev:g}'] = n
    return cols


def checkpoint(session_name, rec, tuning_part):
    """Persist one session's results so a later crash cannot cost this work."""
    if tuning_part is not None and len(tuning_part):
        tuning_part.to_parquet(SHARD_DIR / 'tuning' / f'{session_name}.parquet', index=False)
    # meta is written LAST and is what RESUME tests, so a partial write is retried
    (SHARD_DIR / 'meta' / f'{session_name}.json').write_text(json.dumps(rec, default=str))


# ── Main loop over sessions ───────────────────────────────────────────────────
session_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
print(f"Found {len(session_dirs)} session folders under {DATA_ROOT}\n")

for s_i, session_dir in enumerate(session_dirs):
    session_name = session_dir.name
    meta_path = SHARD_DIR / 'meta' / f'{session_name}.json'
    print(f"[{s_i + 1:2d}/{len(session_dirs)}] {session_name}")

    if RESUME and meta_path.exists():
        print("    (checkpoint exists -- skipping)")
        continue

    rec = dict(session_name=session_name)
    session_issues = []

    nwb_path = find_nwb_path(session_dir)
    if nwb_path is None:
        rec.update(status='no_nwb_file', n_planes=0, n_rois=0, n_dg_trials=0,
                   issues=[f"{session_name}: no .nwb or .nwb.zarr found"])
        checkpoint(session_name, rec, None)
        print("    !! no NWB file")
        continue

    rec['nwb_file'] = nwb_path.name
    rec['backend']  = 'zarr' if nwb_path.name.endswith('.zarr') else 'hdf5'

    try:
        nwb = pynwb.read_nwb(str(nwb_path))
    except Exception as e:
        rec.update(status=f'read_failed: {e}', n_planes=0, n_rois=0, n_dg_trials=0,
                   issues=[f"{session_name}: read_nwb failed -- {e}"])
        checkpoint(session_name, rec, None)
        print(f"    !! read failed: {e}")
        continue

    rec['session_id'] = getattr(nwb, 'session_id', None)

    # ── Stimulus / epoch tables ───────────────────────────────────────────────
    if 'stimulus_table' not in nwb.intervals:
        rec.update(status='no_stimulus_table', n_planes=0, n_rois=0, n_dg_trials=0,
                   issues=[f"{session_name}: no stimulus_table"])
        checkpoint(session_name, rec, None)
        continue

    stim_table  = nwb.intervals['stimulus_table'].to_dataframe()
    epoch_table = (nwb.intervals['epochs'].to_dataframe()
                   if 'epochs' in nwb.intervals else pd.DataFrame())
    rec['stim_names_present'] = ';'.join(sorted(stim_table['stim_name'].dropna().unique()))
    rec['n_epochs'] = len(epoch_table)

    dg = stim_table[stim_table['stim_name'] == STIM_NAME].copy()
    rec['n_dg_trials'] = len(dg)
    if len(dg) == 0:
        rec.update(status=f'no_{STIM_NAME}', n_planes=0, n_rois=0,
                   issues=[f"{session_name}: stimulus '{STIM_NAME}' absent "
                           f"(present: {rec['stim_names_present']})"])
        checkpoint(session_name, rec, None)
        print(f"    !! no {STIM_NAME}")
        continue

    # Trials with no direction cannot contribute to an orientation curve.
    # NOTE: Step 1 printed "Missing responses: 0 / 192" BEFORE dropping these,
    # so its trial count was 192 while only 184 trials carry a direction. That
    # difference is recorded explicitly here rather than left implicit.
    rec['n_dg_trials_missing_direction'] = int(dg['direction'].isna().sum())
    dg = dg.dropna(subset=['direction']).reset_index(drop=True)
    dg['orientation'] = dg['direction'].round().astype(int) % 180

    rec['directions']           = ';'.join(str(x) for x in sorted(dg['direction'].unique()))
    rec['n_directions']         = dg['direction'].nunique()
    rec['n_orientations']       = dg['orientation'].nunique()
    rec['temporal_frequencies'] = ';'.join(str(x) for x in sorted(dg['temporal_frequency'].dropna().unique()))
    rec['spatial_frequencies']  = ';'.join(str(x) for x in sorted(dg['spatial_frequency'].dropna().unique()))
    rec['n_dg_trials_used']     = len(dg)
    rec['dg_trial_duration_s']  = float((dg['stop_time'] - dg['start_time']).median())
    rec['dg_epoch_count']       = int((epoch_table['stim_name'] == STIM_NAME).sum()) if len(epoch_table) else None

    onsets       = dg['start_time'].to_numpy(float)
    directions   = dg['direction'].to_numpy(float)
    orientations = dg['orientation'].to_numpy(int)
    n_tr         = len(dg)

    # ── Per-plane processing ──────────────────────────────────────────────────
    plane_names = [f'plane-{i}' for i in range(MAX_PLANES) if f'plane-{i}' in nwb.processing]
    rec['planes_present'] = ';'.join(plane_names)
    rec['n_planes'] = len(plane_names)
    if not plane_names:
        session_issues.append(f"{session_name}: no plane-* processing modules")

    n_rois_session = 0
    tuning_parts = []

    for plane_name in plane_names:
        try:
            module = nwb.processing[plane_name]
            if DFF_KEY not in module.data_interfaces:
                session_issues.append(f"{session_name}/{plane_name}: no '{DFF_KEY}' interface "
                                      f"(has: {sorted(module.data_interfaces)})")
                continue

            dff_obj    = module[DFF_KEY]
            timestamps = np.asarray(dff_obj.timestamps[:], dtype=float)
            dff        = np.asarray(dff_obj.data[:], dtype=float)   # (n_frames, n_rois)
            roi_tbl    = dff_obj.rois.to_dataframe().reset_index(drop=True)

            # Shape sanity: a mismatch means the ROI mapping is not trustworthy
            if dff.shape[0] != len(timestamps):
                session_issues.append(f"{session_name}/{plane_name}: dff frames {dff.shape[0]} != "
                                      f"timestamps {len(timestamps)} -- truncating to the shorter")
                n = min(dff.shape[0], len(timestamps))
                dff, timestamps = dff[:n], timestamps[:n]
            if dff.shape[1] != len(roi_tbl):
                session_issues.append(f"{session_name}/{plane_name}: dff columns {dff.shape[1]} != "
                                      f"ROI table rows {len(roi_tbl)} -- SKIPPED (ambiguous ROI mapping)")
                continue

            # Trials falling outside this plane's recorded time range are unusable
            n_out = int(((onsets + RESP_START) > timestamps[-1]).sum()
                        + ((onsets + RESP_END) < timestamps[0]).sum())
            if n_out:
                session_issues.append(f"{session_name}/{plane_name}: {n_out} {STIM_NAME} trials "
                                      f"outside the dF/F time range")

            # ── Per-trial responses: window mean minus pre-stimulus baseline ──
            r_lo, r_hi = trial_window_indices(timestamps, onsets, RESP_START, RESP_END)
            b_lo, b_hi = trial_window_indices(timestamps, onsets, BASE_START, BASE_END)
            resp = window_means(dff, r_lo, r_hi)      # (n_trials, n_rois)
            base = window_means(dff, b_lo, b_hi)
            # Step 1 substituted baseline = 0 when the pre-stimulus window was
            # empty; the same convention is kept so batch and single-ROI agree
            base = np.nan_to_num(base, nan=0.0)
            trial_resp = resp - base
            n_rois = trial_resp.shape[1]
            n_valid = np.isfinite(trial_resp).sum(axis=0)

            # ── One row per ROI: identity + QC + orientation and direction curves
            tuning_parts.append(pd.DataFrame({
                'session_name':   session_name,
                'column':         roi_tbl['column'].astype(str).values,
                'volume':         roi_tbl['volume'].astype(str).values,
                'plane':          roi_tbl['plane'].values,
                'roi':            roi_tbl['roi'].values,
                'roi_idx':        np.arange(n_rois),
                'plane_name':     plane_name,
                'is_soma':        (roi_tbl['is_soma'].values if 'is_soma' in roi_tbl else np.nan),
                'n_frames':       dff.shape[0],
                'frac_nan_dff':   np.isnan(dff).mean(axis=0),
                'n_trials_total': n_tr,
                'n_trials_valid': n_valid,
                **condition_stats(trial_resp, orientations, 'ori'),
                **condition_stats(trial_resp, directions,   'dir'),
            }))
            n_rois_session += n_rois
            print(f"    {plane_name}: {n_rois:4d} ROIs, "
                  f"{n_valid.min():3d}-{n_valid.max():3d} valid trials/ROI")

            del dff, resp, base, trial_resp

        except Exception as e:
            session_issues.append(f"{session_name}/{plane_name}: EXCEPTION {e}\n"
                                  f"{traceback.format_exc(limit=3)}")
            print(f"    !! {plane_name}: {e}")

    rec['n_rois'] = n_rois_session
    rec['status'] = 'ok' if n_rois_session > 0 else 'no_rois_processed'
    rec['issues'] = session_issues

    checkpoint(session_name, rec,
               pd.concat(tuning_parts, ignore_index=True) if tuning_parts else None)
    del tuning_parts

# ── Assemble outputs from the checkpoints ─────────────────────────────────────
metas = [json.loads(p.read_text()) for p in sorted((SHARD_DIR / 'meta').glob('*.json'))]
sessions_df = pd.DataFrame(metas).drop(columns=['issues'], errors='ignore')
all_issues = [i for m in metas for i in m.get('issues', [])]

shards = sorted((SHARD_DIR / 'tuning').glob('*.parquet'))
tuning = (pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
          if shards else pd.DataFrame())

sessions_df.to_csv(OUT_DIR / 'session_inventory.csv', index=False)
tuning.to_parquet(OUT_DIR / 'roi_tuning.parquet', index=False)
(OUT_DIR / 'issues_log.txt').write_text('\n'.join(all_issues) if all_issues else '(none)\n')

ori_mean_cols = sorted([c for c in tuning.columns if c.startswith('ori_mean_')],
                       key=lambda c: float(c.rsplit('_', 1)[1]))
dir_mean_cols = sorted([c for c in tuning.columns if c.startswith('dir_mean_')],
                       key=lambda c: float(c.rsplit('_', 1)[1]))

# ── Completeness report ───────────────────────────────────────────────────────
lines = []
add = lines.append
add('=' * 78)
add('STEP 2 BATCH -- DATA COMPLETENESS REPORT')
add('=' * 78)
add(f"Session folders found        : {len(sessions_df)}")
add(f"Sessions processed OK        : {(sessions_df['status'] == 'ok').sum()}")
bad = sessions_df[sessions_df['status'] != 'ok']
add(f"Sessions NOT processed       : {len(bad)}")
for _, r in bad.iterrows():
    add(f"    - {r['session_name']}: {r['status']}")
add('')
add(f"Storage backends             : {sessions_df['backend'].value_counts().to_dict()}")
add(f"Rows in roi_tuning (ROI x session): {len(tuning)}")
if len(tuning):
    add(f"Distinct (col,vol,plane,roi) : {tuning[['column','volume','plane','roi']].drop_duplicates().shape[0]}")
    add(f"is_soma == True              : {int((tuning['is_soma'] == True).sum())} / {len(tuning)}")
    add(f"ROIs with any NaN in dF/F    : {int((tuning['frac_nan_dff'] > 0).sum())}")
    add(f"ROIs missing >=1 trial       : {int((tuning['n_trials_valid'] < tuning['n_trials_total']).sum())}")
    add(f"ROIs with zero valid trials  : {int((tuning['n_trials_valid'] == 0).sum())}")
    add(f"Orientation columns          : {len(ori_mean_cols)}  {ori_mean_cols}")
    add(f"Direction columns            : {len(dir_mean_cols)}")
    add(f"ROIs with any NaN in curve   : {int(tuning[ori_mean_cols].isna().any(axis=1).sum())}")
add('')
add('-' * 78)
add('PER-SESSION INVENTORY')
add('-' * 78)
add(sessions_df.reindex(columns=['session_name', 'backend', 'n_planes', 'n_rois',
                                 'n_dg_trials', 'n_dg_trials_missing_direction',
                                 'n_dg_trials_used', 'n_directions', 'n_orientations',
                                 'status']).to_string(index=False))
add('')
add('-' * 78)
add('STIMULUS CONDITIONS PER SESSION')
add('-' * 78)
add(sessions_df.reindex(columns=['session_name', 'dg_trial_duration_s',
                                 'temporal_frequencies', 'spatial_frequencies',
                                 'directions']).to_string(index=False))
add('')
add('-' * 78)
add('ROI AVAILABILITY ACROSS SESSIONS')
add('-' * 78)
if len(tuning):
    per_key = (tuning.groupby(['column', 'volume', 'plane', 'roi'])['session_name']
                     .nunique().rename('n_sessions').reset_index())
    add(f"Distinct ROI keys (column,volume,plane,roi): {len(per_key)}")
    add('Number of sessions each ROI key appears in:')
    add(per_key['n_sessions'].value_counts().sort_index().to_string())
    add('')
    # Count ROWS, not nunique('roi'): `roi` ids are only unique WITHIN a plane
    # and collide across planes of the same volume, so nunique would undercount
    # (e.g. 914 ROIs in one session share only 407 distinct roi values). This is
    # also why the Step 4 merge must join on all of column+volume+plane+roi.
    add('ROI counts by (column, volume) x session:')
    add(tuning.groupby(['column', 'volume', 'session_name'])
              .size().rename('n_rois').reset_index().to_string(index=False))
    add('')
    dup_roi = tuning.duplicated(['session_name', 'column', 'volume', 'roi']).sum()
    add(f"ROI ids that repeat within a session (across planes): {int(dup_roi)}")
    add(f"Duplicates of the full key (session,col,vol,plane,roi): "
        f"{int(tuning.duplicated(['session_name','column','volume','plane','roi']).sum())}")
add('')
add(f"Anomalies logged: {len(all_issues)}  (full text in {OUT_DIR / 'issues_log.txt'})")
for line in all_issues[:40]:
    add(f"    - {line.splitlines()[0]}")

report = '\n'.join(lines)
(OUT_DIR / 'completeness_report.txt').write_text(report)
print('\n' + report)
print(f"\nSaved outputs to {OUT_DIR}")
print(f"Per-session checkpoints in {SHARD_DIR}/tuning (re-running skips completed sessions)")


*****************************************

# GENERATED CODE - DO NOT RUN

# === Step 3: Compute Tuning Metrics ===
# Formulas follow the Allen Institute NeuroAnalysisTools / v1dd_physiology conventions
# (github.com/zhuangjun1981/NeuroAnalysisTools, SingleCellAnalysis.py):
#
# OSI  = (R_pref − R_orth) / (R_pref + R_orth)
#   R_pref: mean response at the preferred orientation (argmax of orientation curve)
#   R_orth: mean response at the orthogonal orientation (pref + 90° mod 180°)
#
# gOSI (global OSI / 1 − circular variance, Ringach et al. 2002):
#   gOSI = | Σ R_i · exp(2i·θ_i) | / Σ max(R_i, 0)
#   More robust than OSI when some responses are negative (noise floor).
#   Preferred orientation is also recovered from this vector's phase angle.
#
# Tuning width (FWHH) — two-peak von Mises fit to the 12-direction curve:
#   R(θ) = r0 + a1·exp(k·(cos(θ−θ_pref)−1)) + a2·exp(k·(cos(θ−θ_pref−π)−1))
#   FWHH = 2·arccos(ln(0.5)/k + 1)·(180/π)  degrees
#   The fit uses the rectified (negative-clipped) direction curve for stability.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

OUT_DIR = Path('/scratch/v1dd_step2')

# ── 1. Load data ──────────────────────────────────────────────────────────────
tuning_df = pd.read_parquet(OUT_DIR / 'roi_tuning.parquet')

ORIENTATIONS  = np.array([0, 30, 60, 90, 120, 150], dtype=float)
DIRECTIONS    = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], dtype=float)
ORI_MEAN_COLS = [f'ori_mean_{int(o)}' for o in ORIENTATIONS]
DIR_MEAN_COLS = [f'dir_mean_{int(d)}' for d in DIRECTIONS]
KEY_COLS      = ['session_name', 'column', 'volume', 'plane', 'roi', 'roi_idx', 'plane_name']

R_ori = tuning_df[ORI_MEAN_COLS].to_numpy(dtype=float)   # (n_rois, 6)
R_dir = tuning_df[DIR_MEAN_COLS].to_numpy(dtype=float)   # (n_rois, 12)

# ── 2. OSI — Allen convention: (R_pref − R_orth) / (R_pref + R_orth) ─────────
# Preferred orientation: argmax of the mean orientation responses.
# Orthogonal: (pref + 90°) mod 180°, which always lands exactly on one of the
# 6 sampled orientations since they are spaced 30° apart.
pref_idx     = np.argmax(R_ori, axis=1)                   # column index into ORIENTATIONS
pref_ori     = ORIENTATIONS[pref_idx]                     # degrees (discrete argmax)
orth_ori     = (pref_ori + 90) % 180
orth_idx     = np.searchsorted(ORIENTATIONS, orth_ori)    # exact match guaranteed

R_pref    = R_ori[np.arange(len(R_ori)), pref_idx]
R_orth    = R_ori[np.arange(len(R_ori)), orth_idx]
denom_osi = R_pref + R_orth
# OSI can exceed [0,1] when the orthogonal response is negative (noise floor).
# Allen also provides an OSI_rec variant on rectified curves for stability.
OSI = np.where(np.abs(denom_osi) > 1e-9, (R_pref - R_orth) / denom_osi, np.nan)

# ── 3. gOSI — global OSI via vector sum ───────────────────────────────────────
theta_rad = np.deg2rad(2 * ORIENTATIONS)   # doubled angles: maps 180° period → 360°
R_pos     = np.clip(R_ori, 0, None)        # clip negatives before summing
vec_sum   = (R_pos * np.exp(1j * theta_rad)).sum(axis=1)
denom_g   = R_pos.sum(axis=1)
gOSI      = np.where(denom_g > 0, np.abs(vec_sum) / denom_g, np.nan)

# Preferred orientation from the continuous vector (not binned to 30° steps)
pref_ori_vec = (np.rad2deg(np.angle(vec_sum)) / 2) % 180

# ── 4. Tuning width — two-peak von Mises fit to direction data ────────────────
dir_rad = np.deg2rad(DIRECTIONS)

def two_peak_von_mises(theta, r0, a1, a2, k, peak_dir):
    """Two von Mises peaks separated by 180° (Allen NeuroAnalysisTools convention)."""
    peak_arc = np.deg2rad(peak_dir)
    return (r0
            + a1 * np.exp(k * (np.cos(theta - peak_arc)         - 1))
            + a2 * np.exp(k * (np.cos(theta - peak_arc - np.pi) - 1)))

def fit_tuning_width(dir_responses, pref_dir_deg):
    """
    Fit two-peak von Mises to the rectified direction curve.
    Returns FWHH in degrees, or NaN on failure.
    FWHH = 2·arccos(ln(0.5)/k + 1)·(180/π)
    """
    R_rect = np.clip(dir_responses, 0, None)
    A_init = R_rect.max() - R_rect.min()
    p0     = [R_rect.min(), max(A_init, 1e-6), max(A_init * 0.1, 1e-6), 1.5, pref_dir_deg]
    bounds = ([-np.inf, 0,       0,       0.01, 0  ],
              [ np.inf, np.inf,  np.inf,  50,   360 ])
    try:
        popt, _ = curve_fit(two_peak_von_mises, dir_rad, R_rect,
                             p0=p0, bounds=bounds, maxfev=3000)
        k   = popt[3]
        arg = np.log(0.5) / k + 1
        if not (-1 <= arg <= 1):   # arccos domain — k too small → flat tuning
            return np.nan
        return 2 * np.arccos(arg) * 180 / np.pi
    except Exception:
        return np.nan

# Initialise each fit at the argmax direction
pref_dir_discrete = DIRECTIONS[np.argmax(R_dir, axis=1)]

print(f"Fitting tuning widths for {len(R_dir):,} ROIs ...")
tuning_width = np.array([
    fit_tuning_width(R_dir[i], pref_dir_discrete[i])
    for i in range(len(R_dir))
])
print("Done.")

# ── 5. Assemble per-ROI metrics dataframe ─────────────────────────────────────
# The composite key (session_name, column, volume, plane, roi) matches
# coregistration_1196.feather for the Step 4 merge — 'roi' alone is NOT unique.
metrics_df = tuning_df[KEY_COLS].copy()
metrics_df['pref_ori']     = pref_ori        # discrete argmax orientation (deg)
metrics_df['pref_ori_vec'] = pref_ori_vec    # vector-sum preferred orientation (deg)
metrics_df['OSI']          = OSI             # (R_pref - R_orth) / (R_pref + R_orth)
metrics_df['gOSI']         = gOSI            # vector-sum magnitude
metrics_df['tuning_width'] = tuning_width    # FWHH in degrees (von Mises fit)

# Attach orientation mean columns for downstream tuning curve plots (Step 4)
metrics_df = pd.concat([metrics_df, tuning_df[ORI_MEAN_COLS]], axis=1)

n = len(metrics_df)
print(f"\nROIs total          : {n:,}")
for col in ['OSI', 'gOSI', 'tuning_width']:
    ok = np.isfinite(metrics_df[col]).sum()
    print(f"  {col:16s}: {ok:,} valid ({100*ok/n:.1f}%)  "
          f"median={metrics_df[col].median():.3f}  mean={metrics_df[col].mean():.3f}")

metrics_df.to_parquet(OUT_DIR / 'roi_metrics.parquet', index=False)
print(f"\nSaved -> {OUT_DIR / 'roi_metrics.parquet'}")

# ── 6. Scatter plot: tuning width vs OSI ──────────────────────────────────────
valid = metrics_df['OSI'].notna() & metrics_df['tuning_width'].notna()
m = metrics_df[valid]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: scatter coloured by preferred orientation
ax = axes[0]
sc = ax.scatter(m['tuning_width'], m['OSI'],
                c=m['pref_ori_vec'], cmap='hsv',
                s=3, alpha=0.25, linewidths=0, rasterized=True)
ax.set_xlabel('Tuning width FWHH (deg)')
ax.set_ylabel('OSI')
ax.set_title('Tuning width vs OSI\n(colour = preferred orientation)')
plt.colorbar(sc, ax=ax, label='Preferred orientation (deg)')

# Right: 2-D density histogram
ax2 = axes[1]
h = ax2.hist2d(m['tuning_width'], m['OSI'],
               bins=[np.linspace(0, 180, 60), np.linspace(-1, 2, 50)],
               cmap='Blues', density=True)
plt.colorbar(h[3], ax=ax2, label='Density')
ax2.set_xlabel('Tuning width FWHH (deg)')
ax2.set_ylabel('OSI')
ax2.set_title('2-D density: tuning width vs OSI')

plt.suptitle(f'Orientation tuning metrics  -  {valid.sum():,} ROIs', fontsize=11)
plt.tight_layout()
plt.show()