import json, sys

NB = '/root/capsule/code/workshops/Code_verification_V1DD_Functional_Data.ipynb'
CELL = '4e86438b'

OLD = """# ── Assemble and persist outputs ──────────────────────────────────────────────
sessions_df = pd.DataFrame(session_rows)
rois_df     = pd.concat(roi_rows,    ignore_index=True) if roi_rows    else pd.DataFrame()
tuning_df   = pd.concat(tuning_rows, ignore_index=True) if tuning_rows else pd.DataFrame()

# Wide tuning table: one row per ROI, one column per orientation (input to Step 3)
if len(tuning_df):
    key = ['session_name', 'column', 'volume', 'plane', 'roi', 'roi_idx']
    tuning_wide = (tuning_df.pivot_table(index=key, columns='orientation',
                                         values='mean_response', dropna=False)
                            .reset_index())
    tuning_wide.columns = [c if isinstance(c, str) else f'ori_{c}' for c in tuning_wide.columns]
else:
    tuning_wide = pd.DataFrame()

sessions_df.to_csv(OUT_DIR / 'session_inventory.csv', index=False)
rois_df.to_parquet(OUT_DIR / 'roi_inventory.parquet', index=False)
tuning_df.to_parquet(OUT_DIR / 'tuning_curves_long.parquet', index=False)
tuning_wide.to_parquet(OUT_DIR / 'tuning_curves_wide.parquet', index=False)
(OUT_DIR / 'issues_log.txt').write_text('\\n'.join(issues) if issues else '(none)\\n')
"""

NEW = """# ── Assemble and persist outputs ──────────────────────────────────────────────
sessions_df = pd.DataFrame(session_rows)
rois_df     = pd.concat(roi_rows,    ignore_index=True) if roi_rows    else pd.DataFrame()
tuning_df   = pd.concat(tuning_rows, ignore_index=True) if tuning_rows else pd.DataFrame()

# Write the expensive results BEFORE any reshaping, so a failure below cannot
# throw away ~45 min of NWB reading.
sessions_df.to_csv(OUT_DIR / 'session_inventory.csv', index=False)
rois_df.to_parquet(OUT_DIR / 'roi_inventory.parquet', index=False)
tuning_df.to_parquet(OUT_DIR / 'tuning_curves_long.parquet', index=False)
(OUT_DIR / 'issues_log.txt').write_text('\\n'.join(issues) if issues else '(none)\\n')

# Wide tuning table: one row per ROI, one column per orientation (input to Step 3).
# Do NOT use pivot_table(..., dropna=False) here. With this 6-column index it
# materializes the full CARTESIAN PRODUCT of the index levels (25 sessions x
# every distinct roi id x every roi_idx x every plane x ...) -- hundreds of
# millions of empty rows, which exhausts memory and kills the kernel. unstack()
# creates only the key combinations that actually occur.
if len(tuning_df):
    key = ['session_name', 'column', 'volume', 'plane', 'roi', 'roi_idx']
    assert not tuning_df.duplicated(key + ['orientation']).any(), \\
        'duplicate (ROI, orientation) rows -- the ROI key is not unique'
    tuning_wide = (tuning_df.set_index(key + ['orientation'])['mean_response']
                            .unstack('orientation')
                            .reset_index())
    tuning_wide.columns = [c if isinstance(c, str) else f'ori_{c}' for c in tuning_wide.columns]
else:
    tuning_wide = pd.DataFrame()
tuning_wide.to_parquet(OUT_DIR / 'tuning_curves_wide.parquet', index=False)
"""

OLD2 = """    # Trials with no direction cannot contribute to an orientation curve
    rec['n_dg_trials_missing_direction'] = int(dg['direction'].isna().sum())"""

NEW2 = """    # Trials with no direction cannot contribute to an orientation curve.
    # NOTE: Step 1 printed "Missing responses: 0 / 192" BEFORE dropping these,
    # so its trial count was 192 while only 184 trials carry a direction. That
    # difference is recorded explicitly here rather than left implicit.
    rec['n_dg_trials_missing_direction'] = int(dg['direction'].isna().sum())"""

OLD3 = """add(sessions_df.reindex(columns=['session_name', 'backend', 'n_planes', 'n_rois',
                                 'n_dg_trials', 'n_dg_trials_used', 'n_orientations',
                                 'status']).to_string(index=False))"""

NEW3 = """add(sessions_df.reindex(columns=['session_name', 'backend', 'n_planes', 'n_rois',
                                 'n_dg_trials', 'n_dg_trials_missing_direction',
                                 'n_dg_trials_used', 'n_orientations',
                                 'status']).to_string(index=False))"""

nb = json.load(open(NB))
cell = next(c for c in nb['cells'] if c['id'] == CELL)
src = ''.join(cell['source'])

for old, new, label in [(OLD, NEW, 'assembly/pivot'), (OLD2, NEW2, 'direction comment'),
                        (OLD3, NEW3, 'report columns')]:
    if old not in src:
        sys.exit(f'PATCH FAILED: could not find block "{label}"')
    src = src.replace(old, new, 1)

cell['source'] = src.splitlines(keepends=True)
json.dump(nb, open(NB, 'w'), indent=1, ensure_ascii=False)
open(NB, 'a').write('\n')
print('Patched cell', CELL)
print('pivot_table remaining:', src.count('pivot_table'))
print('unstack present:', 'unstack' in src)
