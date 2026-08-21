"""Selecting the most grating-responsive ROI across imaging planes by SNR."""
import numpy as np

from . import io


def epoch_frame_mask(timestamps, epochs):
    """Boolean mask of frames falling inside any of the given epoch intervals."""
    mask = np.zeros(len(timestamps), dtype=bool)
    for _, row in epochs.iterrows():
        mask |= (timestamps >= row['start_time']) & (timestamps <= row['stop_time'])
    return mask


def roi_snr(dff_all, grating_mask, spont_mask):
    """SNR per ROI: mean dF/F during gratings over std during spontaneous."""
    grating_mean = dff_all[grating_mask].mean(axis=0)
    spont_std = dff_all[spont_mask].std(axis=0)
    spont_std[spont_std < 1e-6] = 1e-6                 # guard against /0
    return grating_mean / spont_std


def select_most_responsive(nwb, dg_epochs, spont_epochs, planes=None):
    """Return info for the highest-SNR ROI across all imaging planes."""
    planes = planes if planes is not None else io.plane_names(nwb)
    best_snr = -np.inf
    best = None

    for plane_name in planes:
        dff_all, ts, roi_table = io.plane_dff(nwb, plane_name)

        grating_mask = epoch_frame_mask(ts, dg_epochs)
        spont_mask = epoch_frame_mask(ts, spont_epochs)
        if not spont_mask.any():
            spont_mask = ~grating_mask                  # fall back to non-grating frames
        if not grating_mask.any():
            continue

        snr = roi_snr(dff_all, grating_mask, spont_mask)
        idx = int(np.argmax(snr))
        if snr[idx] > best_snr:
            best_snr = float(snr[idx])
            best = dict(plane=plane_name, array_idx=idx,
                        roi_id=int(roi_table.iloc[idx]['roi']),
                        snr=best_snr, timestamps=ts)
    return best
