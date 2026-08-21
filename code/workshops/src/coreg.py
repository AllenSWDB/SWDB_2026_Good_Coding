"""Coregistration of functional metrics with EM cell types (Step 4)."""
import numpy as np
import pandas as pd

from . import artifacts
from .metrics import ORI_MEAN_COLS


def load_em_tables(data_path=None):
    """Load the coregistration and cell-type feather tables."""
    data_path = data_path or artifacts.EM_DATA_PATH
    coreg = pd.read_feather(f'{data_path}/coregistration_1196.feather')
    cell_type = pd.read_feather(f'{data_path}/soma_and_cell_type_1196.feather')
    return coreg, cell_type


def merge_metrics_with_cell_type(metrics_df, coreg, cell_type):
    """Join per-ROI metrics to EM root IDs and cell types on column/volume/plane/roi."""
    m = metrics_df.copy()
    m['column'] = m['column'].astype(int)
    m['volume'] = m['volume'].astype(int)
    coreg_u = coreg[['pt_root_id', 'column', 'volume', 'plane', 'roi']].drop_duplicates()
    merged = m.merge(coreg_u, on=['column', 'volume', 'plane', 'roi'], how='inner')
    ct = (cell_type[['pt_root_id', 'cell_type']]
          .dropna(subset=['cell_type']).drop_duplicates(subset='pt_root_id'))
    return merged.merge(ct, on='pt_root_id', how='inner')


def filter_responsive(merged, threshold=0.2):
    """Keep cells whose peak orientation response exceeds a dF/F threshold."""
    return merged[merged[ORI_MEAN_COLS].max(axis=1) > threshold].copy()


def cell_type_summary(merged_filt, metrics=('gOSI', 'tuning_width')):
    """Mean/SEM/count of each metric per cell type."""
    rows = []
    for metric in metrics:
        g = merged_filt.groupby('cell_type')[metric].agg(['mean', 'sem', 'count']).round(3)
        g.columns = pd.MultiIndex.from_tuples([(metric, c) for c in g.columns])
        rows.append(g)
    return pd.concat(rows, axis=1)
