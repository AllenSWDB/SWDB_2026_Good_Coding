"""Trial-averaged PSTH construction and data-driven response-window detection."""
import numpy as np


def build_psth(roi_dff, timestamps, onsets, stim_dur, pre_win=1.0, post_win=2.0):
    """Align trials to onset, interpolate to a common grid, and average."""
    dt = float(np.median(np.diff(timestamps)))
    t_grid = np.arange(-pre_win, stim_dur + post_win, dt)

    psth = np.full((len(onsets), len(t_grid)), np.nan)
    for i, onset in enumerate(onsets):
        t_rel = timestamps - onset
        in_win = (t_rel >= -pre_win) & (t_rel <= stim_dur + post_win)
        idx = np.where(in_win)[0]
        if len(idx) < 2:
            continue
        psth[i] = np.interp(t_grid, t_rel[idx], roi_dff[idx])

    mean_psth = np.nanmean(psth, axis=0)
    n_valid = np.sum(~np.isnan(psth), axis=0).clip(1)
    sem_psth = np.nanstd(psth, axis=0) / np.sqrt(n_valid)
    return t_grid, psth, mean_psth, sem_psth


def detect_response_window(t_grid, mean_psth, stim_dur, half_win=0.5, lag=1.0):
    """Center a ±half_win window on the PSTH peak within [0, stim_dur+lag]."""
    in_stim = (t_grid >= 0) & (t_grid <= stim_dur + lag)
    peak_lat = float(t_grid[in_stim][np.argmax(mean_psth[in_stim])])
    resp_start = max(0.0, peak_lat - half_win)
    resp_end = min(stim_dur + lag, peak_lat + half_win)
    return peak_lat, resp_start, resp_end
