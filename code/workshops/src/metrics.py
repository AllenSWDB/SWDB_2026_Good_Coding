"""Orientation-selectivity metrics and von Mises tuning-curve fits."""
import numpy as np
from scipy.optimize import curve_fit

ORIENTATIONS = np.array([0, 30, 60, 90, 120, 150], dtype=float)
DIRECTIONS = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], dtype=float)
ORI_MEAN_COLS = [f'ori_mean_{int(o)}' for o in ORIENTATIONS]
DIR_MEAN_COLS = [f'dir_mean_{int(d)}' for d in DIRECTIONS]


def preferred_orientation(R_ori):
    """Preferred orientation (deg) as the argmax of the orientation curve."""
    return ORIENTATIONS[np.argmax(R_ori, axis=-1)]


def compute_osi(R_ori):
    """Allen OSI = (R_pref - R_orth) / (R_pref + R_orth) per ROI."""
    R_ori = np.atleast_2d(R_ori)
    pref_idx = np.argmax(R_ori, axis=1)
    orth_ori = (ORIENTATIONS[pref_idx] + 90) % 180
    orth_idx = np.searchsorted(ORIENTATIONS, orth_ori)
    R_pref = R_ori[np.arange(len(R_ori)), pref_idx]
    R_orth = R_ori[np.arange(len(R_ori)), orth_idx]
    denom = R_pref + R_orth
    return np.where(np.abs(denom) > 1e-9, (R_pref - R_orth) / denom, np.nan)


def compute_gosi(R_ori):
    """Global OSI (1 - circular variance) and vector-sum preferred orientation."""
    R_ori = np.atleast_2d(R_ori)
    theta = np.deg2rad(2 * ORIENTATIONS)
    R_pos = np.clip(R_ori, 0, None)
    vec = (R_pos * np.exp(1j * theta)).sum(axis=1)
    denom = R_pos.sum(axis=1)
    gosi = np.where(denom > 0, np.abs(vec) / denom, np.nan)
    pref_vec = (np.rad2deg(np.angle(vec)) / 2) % 180
    return gosi, pref_vec


def two_peak_von_mises(theta, r0, a1, a2, k, peak_dir):
    """Two von Mises peaks separated by 180 deg (direction-space model)."""
    peak_arc = np.deg2rad(peak_dir)
    return (r0
            + a1 * np.exp(k * (np.cos(theta - peak_arc) - 1))
            + a2 * np.exp(k * (np.cos(theta - peak_arc - np.pi) - 1)))


def orientation_von_mises(theta, r0, a, k, peak_ori):
    """Single von Mises with 180 deg period (orientation-space model)."""
    return r0 + a * np.exp(k * (np.cos(2 * np.deg2rad(theta - peak_ori)) - 1))


def fwhh_from_kappa(k):
    """Full width at half height (deg) for a von Mises of concentration k."""
    arg = np.log(0.5) / k + 1
    if not (-1 <= arg <= 1):
        return np.nan
    return 2 * np.arccos(arg) * 180 / np.pi


def fit_tuning_width(dir_responses, pref_dir_deg):
    """Fit two-peak von Mises to the rectified direction curve; return FWHH, popt."""
    R = np.clip(np.asarray(dir_responses, float), 0, None)
    A = R.max() - R.min()
    p0 = [R.min(), max(A, 1e-6), max(A * 0.1, 1e-6), 1.5, pref_dir_deg]
    bounds = ([-np.inf, 0, 0, 0.01, 0], [np.inf, np.inf, np.inf, 50, 360])
    try:
        popt, _ = curve_fit(two_peak_von_mises, np.deg2rad(DIRECTIONS), R,
                            p0=p0, bounds=bounds, maxfev=3000)
    except Exception:
        return np.nan, None
    return fwhh_from_kappa(popt[3]), popt


def fit_orientation_curve(ori_responses, pref_ori_deg):
    """Fit a single 180 deg-period von Mises to an orientation curve; return FWHH, popt."""
    R = np.asarray(ori_responses, float)
    A = R.max() - R.min()
    p0 = [R.min(), max(A, 1e-6), 1.5, pref_ori_deg]
    bounds = ([-np.inf, 0, 0.01, 0], [np.inf, np.inf, 50, 180])
    try:
        popt, _ = curve_fit(orientation_von_mises, ORIENTATIONS, R,
                            p0=p0, bounds=bounds, maxfev=5000)
    except Exception:
        return np.nan, None
    return fwhh_from_kappa(popt[2]), popt


def compute_metrics(tuning_df, key_cols=None):
    """Build the per-ROI metrics table (pref ori, OSI, gOSI, tuning width)."""
    import pandas as pd
    key_cols = key_cols or ['session_name', 'column', 'volume', 'plane',
                            'roi', 'roi_idx', 'plane_name']
    R_ori = tuning_df[ORI_MEAN_COLS].to_numpy(float)
    R_dir = tuning_df[DIR_MEAN_COLS].to_numpy(float)

    gosi, pref_vec = compute_gosi(R_ori)
    pref_dir = DIRECTIONS[np.argmax(R_dir, axis=1)]
    widths = np.array([fit_tuning_width(R_dir[i], pref_dir[i])[0]
                       for i in range(len(R_dir))])

    out = tuning_df[key_cols].copy()
    out['pref_ori'] = preferred_orientation(R_ori)
    out['pref_ori_vec'] = pref_vec
    out['OSI'] = compute_osi(R_ori)
    out['gOSI'] = gosi
    out['tuning_width'] = widths
    return pd.concat([out, tuning_df[ORI_MEAN_COLS]], axis=1)
