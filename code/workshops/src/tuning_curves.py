"""Reshaping and aligning stored tuning curves for plotting."""
import numpy as np
import pandas as pd

ORI_COLS = ['ori_mean_0', 'ori_sd_0', 'ori_n_0', 'ori_mean_30', 'ori_sd_30',
            'ori_n_30', 'ori_mean_60', 'ori_sd_60', 'ori_n_60', 'ori_mean_90',
            'ori_sd_90', 'ori_n_90', 'ori_mean_120', 'ori_sd_120', 'ori_n_120',
            'ori_mean_150', 'ori_sd_150', 'ori_n_150']


def reshape_tuning(df):
    """Reshape wide ori_mean/sd/n columns into a long per-orientation table."""
    wide = df[ORI_COLS].copy()
    wide.index.name = 'curve_id'
    long = (pd.wide_to_long(wide.reset_index(),
                            stubnames=['ori_mean', 'ori_sd', 'ori_n'],
                            i='curve_id', j='orientation', sep='_', suffix=r'\d+')
            .reset_index()
            .rename(columns={'ori_mean': 'mean_response', 'ori_sd': 'sd', 'ori_n': 'n'})
            .sort_values(['curve_id', 'orientation']))
    long['sem'] = np.where(long['n'] > 0, long['sd'] / np.sqrt(long['n']), np.nan)
    return long


def align_and_normalize(tc):
    """Peak-normalise a 6-point orientation curve and centre it at 0 deg."""
    tc = np.asarray(tc, dtype=float)
    rolled = np.roll(tc, -int(np.argmax(tc)))
    if rolled[0] > 0:
        rolled = rolled / rolled[0]
    return rolled[[3, 4, 5, 0, 1, 2, 3]]     # -> [-90,-60,-30,0,+30,+60,+90]
