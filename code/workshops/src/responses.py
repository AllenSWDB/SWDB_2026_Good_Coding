"""Per-trial response extraction and tuning-curve aggregation."""
import numpy as np
import pandas as pd


def trial_window_indices(timestamps, onsets, t0, t1):
    """Frame index bounds [lo, hi) covering [onset+t0, onset+t1] per trial."""
    lo = np.searchsorted(timestamps, onsets + t0, side='left')
    hi = np.searchsorted(timestamps, onsets + t1, side='right')
    return lo, hi


def window_means(dff, lo, hi):
    """Per-trial, per-ROI nanmean of dF/F over frame windows [lo, hi)."""
    out = np.full((len(lo), dff.shape[1]), np.nan)
    for i in range(len(lo)):
        if hi[i] > lo[i]:
            seg = dff[lo[i]:hi[i]]
            out[i] = np.nanmean(seg, axis=0) if np.isfinite(seg).any() else np.nan
    return out


def condition_stats(trial_resp, levels, prefix):
    """Reduce (n_trials, n_rois) responses to mean/sd/n per stimulus level."""
    cols = {}
    for lev in np.unique(levels):
        sel = trial_resp[levels == lev]
        n = np.isfinite(sel).sum(axis=0)
        with np.errstate(invalid='ignore'):
            cols[f'{prefix}_mean_{lev:g}'] = np.nanmean(sel, axis=0)
            cols[f'{prefix}_sd_{lev:g}'] = np.where(n > 1, np.nanstd(sel, axis=0, ddof=1), np.nan)
        cols[f'{prefix}_n_{lev:g}'] = n
    return cols


def single_roi_responses(roi_dff, timestamps, dg_stim,
                         resp_start, resp_end, base_start=-0.5, base_end=0.0):
    """Per-trial baseline-corrected window-mean response for one ROI."""
    responses = []
    for _, row in dg_stim.iterrows():
        t_rel = timestamps - row['start_time']
        base_idx = np.where((t_rel >= base_start) & (t_rel < base_end))[0]
        baseline = float(np.nanmean(roi_dff[base_idx])) if len(base_idx) else 0.0
        resp_idx = np.where((t_rel >= resp_start) & (t_rel <= resp_end))[0]
        if len(resp_idx) == 0:
            responses.append(np.nan)
        else:
            responses.append(float(np.nanmean(roi_dff[resp_idx])) - baseline)
    out = dg_stim.copy()
    out['response'] = responses
    return out


def add_orientation(dg_stim):
    """Collapse direction to orientation (direction % 180), dropping NaNs."""
    out = dg_stim.dropna(subset=['direction', 'response']).copy()
    out['orientation'] = out['direction'].round().astype(int) % 180
    return out


def tuning_table(dg_stim, level='orientation'):
    """Aggregate per-trial responses to mean/sem/n per stimulus level."""
    return (
        dg_stim.groupby(level)['response']
        .agg(mean_response='mean',
             sem=lambda x: x.std(ddof=1) / np.sqrt(x.count()),
             n_trials='count')
        .reset_index()
    )
