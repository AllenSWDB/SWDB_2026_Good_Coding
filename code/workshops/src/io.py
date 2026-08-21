"""Loading NWB sessions and extracting stimulus, epoch, and dF/F data."""
from pathlib import Path

import numpy as np
import pynwb


def find_nwb_path(session_dir):
    """Return the NWB store inside a session folder (zarr or HDF5), or None."""
    session_dir = Path(session_dir)
    hits = sorted(list(session_dir.glob('*.nwb.zarr')) + list(session_dir.glob('*.nwb')))
    return hits[0] if hits else None


def load_nwb(path):
    """Open an NWB file, dispatching on file type (zarr or HDF5)."""
    return pynwb.read_nwb(str(path))


def stimulus_epoch_tables(nwb):
    """Return (stimulus_table, epochs) as dataframes; epochs empty if absent."""
    import pandas as pd
    stim = nwb.intervals['stimulus_table'].to_dataframe()
    epochs = (nwb.intervals['epochs'].to_dataframe()
              if 'epochs' in nwb.intervals else pd.DataFrame())
    return stim, epochs


def grating_trials(stim_table, stim_name='drifting_gratings_full'):
    """Return the subset of the stimulus table for one grating stimulus."""
    return stim_table[stim_table['stim_name'] == stim_name].copy()


def plane_names(nwb, max_planes=8):
    """List the imaging-plane processing modules present in the session."""
    return [f'plane-{i}' for i in range(max_planes) if f'plane-{i}' in nwb.processing]


def plane_dff(nwb, plane_name, dff_key='dff'):
    """Return (dff matrix, timestamps, roi_table) for one imaging plane."""
    dff_obj = nwb.processing[plane_name][dff_key]
    dff = np.asarray(dff_obj.data[:], dtype=float)          # (n_frames, n_rois)
    timestamps = np.asarray(dff_obj.timestamps[:], dtype=float)
    roi_table = dff_obj.rois.to_dataframe().reset_index(drop=True)
    return dff, timestamps, roi_table


def roi_dff_trace(nwb, plane_name, roi_idx, dff_key='dff'):
    """Return (single-ROI dF/F trace, timestamps) for one column index."""
    dff_obj = nwb.processing[plane_name][dff_key]
    trace = np.asarray(dff_obj.data[:, roi_idx], dtype=float)
    timestamps = np.asarray(dff_obj.timestamps[:], dtype=float)
    return trace, timestamps
