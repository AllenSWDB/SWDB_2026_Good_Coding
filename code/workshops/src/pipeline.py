"""End-to-end drivers that compute and persist pipeline artifacts."""
import numpy as np
import pandas as pd

from . import artifacts, io, selection, psth, responses, metrics

STIM_NAME = 'drifting_gratings_full'


def run_step1(session_name, out_dir=None, half_win=0.5):
    """Compute and save all single-ROI Step 1 artifacts for one session."""
    out_dir = artifacts.ensure_dir(out_dir or artifacts.STEP1_DIR)
    session_dir = artifacts.DATA_ROOT / session_name
    nwb = io.load_nwb(io.find_nwb_path(session_dir))

    stim, epochs = io.stimulus_epoch_tables(nwb)
    dg_stim = io.grating_trials(stim, STIM_NAME)
    dg_epochs = epochs[epochs['stim_name'] == STIM_NAME]
    spont_epochs = epochs[epochs['stim_name'] == 'spontaneous']

    best = selection.select_most_responsive(nwb, dg_epochs, spont_epochs)
    roi_dff, timestamps = io.roi_dff_trace(nwb, best['plane'], best['array_idx'])

    stim_dur = float((dg_stim['stop_time'] - dg_stim['start_time']).median())
    onsets = dg_stim['start_time'].to_numpy(float)
    t_grid, _, mean_psth, sem_psth = psth.build_psth(roi_dff, timestamps, onsets, stim_dur)
    peak_lat, resp_start, resp_end = psth.detect_response_window(
        t_grid, mean_psth, stim_dur, half_win=half_win)

    dg_resp = responses.single_roi_responses(roi_dff, timestamps, dg_stim, resp_start, resp_end)
    dg_resp = responses.add_orientation(dg_resp)
    ori_tuning = responses.tuning_table(dg_resp, 'orientation')
    dir_tuning = responses.tuning_table(dg_resp, 'direction')

    fwhh_ori, popt_ori = metrics.fit_orientation_curve(
        ori_tuning['mean_response'].to_numpy(), float(ori_tuning.loc[ori_tuning['mean_response'].idxmax(), 'orientation']))
    fwhh_dir, popt_dir = metrics.fit_tuning_width(
        dir_tuning['mean_response'].to_numpy(), float(dir_tuning.loc[dir_tuning['mean_response'].idxmax(), 'direction']))

    meta = dict(session_name=session_name, stim_name=STIM_NAME,
                plane=best['plane'], array_idx=best['array_idx'],
                roi_id=best['roi_id'], snr=best['snr'], stim_dur=stim_dur,
                peak_lat=peak_lat, resp_start=resp_start, resp_end=resp_end,
                fit_ori=dict(fwhh=fwhh_ori, popt=None if popt_ori is None else list(popt_ori)),
                fit_dir=dict(fwhh=fwhh_dir, popt=None if popt_dir is None else list(popt_dir)))

    artifacts.save_json(meta, out_dir / 'selected_roi.json')
    artifacts.save_table(pd.DataFrame({'time': timestamps, 'dff': roi_dff}), out_dir / 'roi_trace.parquet')
    artifacts.save_table(epochs.reset_index(drop=True), out_dir / 'epochs.parquet')
    artifacts.save_table(dg_resp.reset_index(drop=True), out_dir / 'stim_dg.parquet')
    artifacts.save_table(pd.DataFrame({'t': t_grid, 'mean': mean_psth, 'sem': sem_psth}), out_dir / 'psth.parquet')
    artifacts.save_table(ori_tuning, out_dir / 'ori_tuning.parquet')
    artifacts.save_table(dir_tuning, out_dir / 'dir_tuning.parquet')
    return meta


def run_step3(out_dir=None):
    """Compute and save the per-ROI metrics table from roi_tuning.parquet."""
    out_dir = out_dir or artifacts.STEP2_DIR
    tuning = artifacts.load_table(out_dir / 'roi_tuning.parquet')
    m = metrics.compute_metrics(tuning)
    artifacts.save_table(m, out_dir / 'roi_metrics.parquet')
    return m


def run_step4(out_dir=None):
    """Merge metrics with EM cell types and cache the responsive-cell table."""
    from . import coreg
    out_dir = out_dir or artifacts.STEP2_DIR
    m = artifacts.load_table(out_dir / 'roi_metrics.parquet')
    cg, ct = coreg.load_em_tables()
    merged = coreg.filter_responsive(coreg.merge_metrics_with_cell_type(m, cg, ct))
    artifacts.save_table(merged, out_dir / 'roi_cell_type.parquet')
    return merged
