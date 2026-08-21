"""Convert the already-computed per-trial shards into the one-row-per-ROI format,
so switching the checkpoint granularity does not force a recompute."""
import glob, os
import numpy as np
import pandas as pd

SRC = '/scratch/v1dd_step2/shards/trials'
DST = '/scratch/v1dd_step2/shards/tuning'
ROIS = '/scratch/v1dd_step2/shards/rois'
os.makedirs(DST, exist_ok=True)

KEY = ['session_name', 'column', 'volume', 'plane', 'roi', 'roi_idx']

for f in sorted(glob.glob(f'{SRC}/*.parquet')):
    name = os.path.basename(f)
    trials = pd.read_parquet(f)
    rois = pd.read_parquet(f'{ROIS}/{name}')

    out = rois.rename(columns={'n_orientations': '_drop'}).drop(columns='_drop', errors='ignore')

    for prefix, col in (('ori', 'orientation'), ('dir', 'direction')):
        g = trials.groupby(KEY + [col], observed=True)['response']
        stats = g.agg(mean='mean', sd='std', n='count').reset_index()
        for stat in ('mean', 'sd', 'n'):
            wide = stats.pivot(index=KEY, columns=col, values=stat)
            wide.columns = [f'{prefix}_{stat}_{c:g}' for c in wide.columns]
            out = out.merge(wide.reset_index(), on=KEY, how='left')

    # match the live code: sd undefined for n<2 -> NaN (pandas std already gives NaN)
    out.to_parquet(f'{DST}/{name}', index=False)
    print(f'{name}: {len(trials)} trial rows -> {len(out)} ROI rows, {out.shape[1]} cols')

print('\nmigrated', len(glob.glob(f'{DST}/*.parquet')), 'sessions')
