"""Batch tuning-curve extraction across all sessions and ROIs (Step 2)."""
import json
import traceback
import warnings

import numpy as np
import pandas as pd

from . import artifacts, io
from .responses import trial_window_indices, window_means, condition_stats

STIM_NAME = 'drifting_gratings_full'
RESP_START, RESP_END = 0.7, 2.5     # s after trial onset
BASE_START, BASE_END = -0.5, 0.0    # s relative to trial onset


def _checkpoint(shard_dir, session_name, rec, tuning_part):
    """Persist one session's tuning shard and metadata JSON."""
    if tuning_part is not None and len(tuning_part):
        tuning_part.to_parquet(shard_dir / 'tuning' / f'{session_name}.parquet', index=False)
    (shard_dir / 'meta' / f'{session_name}.json').write_text(json.dumps(rec, default=str))


def process_plane(nwb, plane_name, onsets, orientations, directions, n_tr):
    """Return one-row-per-ROI tuning stats for a single imaging plane."""
    dff, timestamps, roi_tbl = io.plane_dff(nwb, plane_name)
    if dff.shape[0] != len(timestamps):
        n = min(dff.shape[0], len(timestamps))
        dff, timestamps = dff[:n], timestamps[:n]
    if dff.shape[1] != len(roi_tbl):
        return None

    r_lo, r_hi = trial_window_indices(timestamps, onsets, RESP_START, RESP_END)
    b_lo, b_hi = trial_window_indices(timestamps, onsets, BASE_START, BASE_END)
    resp = window_means(dff, r_lo, r_hi)
    base = np.nan_to_num(window_means(dff, b_lo, b_hi), nan=0.0)
    trial_resp = resp - base
    n_rois = trial_resp.shape[1]

    return pd.DataFrame({
        'column': roi_tbl['column'].astype(str).values,
        'volume': roi_tbl['volume'].astype(str).values,
        'plane': roi_tbl['plane'].values,
        'roi': roi_tbl['roi'].values,
        'roi_idx': np.arange(n_rois),
        'plane_name': plane_name,
        'is_soma': (roi_tbl['is_soma'].values if 'is_soma' in roi_tbl else np.nan),
        'n_frames': dff.shape[0],
        'frac_nan_dff': np.isnan(dff).mean(axis=0),
        'n_trials_total': n_tr,
        'n_trials_valid': np.isfinite(trial_resp).sum(axis=0),
        **condition_stats(trial_resp, orientations, 'ori'),
        **condition_stats(trial_resp, directions, 'dir'),
    })


def process_session(nwb, session_name):
    """Return the concatenated per-ROI tuning table for one loaded session."""
    stim, _ = io.stimulus_epoch_tables(nwb)
    dg = io.grating_trials(stim, STIM_NAME).dropna(subset=['direction']).reset_index(drop=True)
    dg['orientation'] = dg['direction'].round().astype(int) % 180
    onsets = dg['start_time'].to_numpy(float)
    directions = dg['direction'].to_numpy(float)
    orientations = dg['orientation'].to_numpy(int)

    parts = []
    for plane_name in io.plane_names(nwb):
        try:
            part = process_plane(nwb, plane_name, onsets, orientations, directions, len(dg))
            if part is not None:
                part.insert(0, 'session_name', session_name)
                parts.append(part)
        except Exception:
            traceback.print_exc(limit=2)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def run_batch(data_root=None, out_dir=None, resume=True):
    """Process every session to the combined roi_tuning.parquet, checkpointing each."""
    warnings.filterwarnings('ignore')
    data_root = data_root or artifacts.DATA_ROOT
    out_dir = out_dir or artifacts.STEP2_DIR
    shard_dir = out_dir / 'shards'
    for sub in ('tuning', 'meta'):
        artifacts.ensure_dir(shard_dir / sub)

    for session_dir in sorted(d for d in data_root.iterdir() if d.is_dir()):
        session_name = session_dir.name
        meta_path = shard_dir / 'meta' / f'{session_name}.json'
        if resume and meta_path.exists():
            continue
        nwb_path = io.find_nwb_path(session_dir)
        if nwb_path is None:
            _checkpoint(shard_dir, session_name, dict(status='no_nwb_file'), None)
            continue
        try:
            nwb = io.load_nwb(nwb_path)
            part = process_session(nwb, session_name)
        except Exception as e:
            _checkpoint(shard_dir, session_name, dict(status=f'failed: {e}'), None)
            continue
        _checkpoint(shard_dir, session_name, dict(status='ok', n_rois=len(part)), part)

    shards = sorted((shard_dir / 'tuning').glob('*.parquet'))
    tuning = (pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
              if shards else pd.DataFrame())
    tuning.to_parquet(out_dir / 'roi_tuning.parquet', index=False)
    return tuning
