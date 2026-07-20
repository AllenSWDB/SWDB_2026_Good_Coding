import os
import argparse
from os.path import join as pjoin

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
DATA_DIR = "/root/capsule/data/v1dd_1196"
OUT_DIR = "/root/capsule/scratch/SWB_test_0"
MAT = 1196
os.makedirs(OUT_DIR, exist_ok=True)

# Analysis parameters
BIN_WIDTH_UM = 25.0        # distance bin width
MAX_DISTANCE_UM = 900.0    # cap plotted distance range
MIN_PAIRS_PER_BIN = 200    # bins below this are drawn faded (unreliable estimate)


def read_feather(path, columns=None):
    """Read a Feather/Arrow-IPC file, optionally selecting columns."""
    with open(path, "rb") as fh:
        table = ipc.open_file(fh).read_all()
    if columns is not None:
        table = table.select(columns)
    return table.to_pandas()


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% confidence interval for a binomial proportion."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lo = np.clip(center - half, 0, 1)
    hi = np.clip(center + half, 0, 1)
    return lo, hi


def main(buggy=True):
    print("Loading cell / proofreading / synapse data ...")

    cell = read_feather(pjoin(DATA_DIR, f"soma_and_cell_type_{MAT}.feather"))
    proof_axon = np.load(pjoin(DATA_DIR, f"proofread_axon_list_{MAT}.npy"))
    proof_dend = np.load(pjoin(DATA_DIR, f"proofread_dendrite_list_{MAT}.npy"))

    # Excitatory cells with a valid segment id and soma position
    if buggy:
        #BUG 1: Using cortical depth only (not eucludean soma distance)
        pos_cols = ["pt_position_trform_y"]
        # #BUG 2: Using untransformed coordinates (need to be transformed relative to pia/white matter bounds)
        # pos_cols = ["pt_position_y"]
    else:
        pos_cols = ["pt_position_trform_x", "pt_position_trform_y", "pt_position_trform_z"]

    ecells = cell[(cell["cell_type_coarse"] == "E") & (cell["pt_root_id"] != 0)].copy()
    ecells = ecells.dropna(subset=pos_cols)
    # one soma row per root id (dataset already has unique root ids among typed cells)
    ecells = ecells.drop_duplicates("pt_root_id")

    axon_set = set(proof_axon.tolist())
    dend_set = set(proof_dend.tolist())

    pre_cells = ecells[ecells["pt_root_id"].isin(axon_set)].copy()   # complete outputs
    post_cells = ecells[ecells["pt_root_id"].isin(dend_set)].copy()  # complete inputs

    pre_ids = pre_cells["pt_root_id"].to_numpy()
    post_ids = post_cells["pt_root_id"].to_numpy()

    #TODO this bug seemed to really break things... don't include it. 
    # if buggy:
    #     #BUG 3: Incorrectly assume neurons must have both a proofread axon and
    #     # a proofread dendrite to be included in the analysis.
    #     shared_ids = np.intersect1d(pre_ids, post_ids)
    #     pre_ids = shared_ids
    #     post_ids = shared_ids

    print(f"  presynaptic pool (E, proofread axon):     {len(pre_ids):,}")
    print(f"  postsynaptic pool (E, proofread dendrite): {len(post_ids):,}")

    # Positions (nm) -> micrometers, indexed by root id
    if buggy:
        #BUG 4: Assuming coordinates are already in microns (it's the name of the dataset after all!)
        pos_um = (ecells.set_index("pt_root_id")[pos_cols].to_numpy(dtype=np.float64)) 
    else:
        pos_um = (ecells.set_index("pt_root_id")[pos_cols].to_numpy(dtype=np.float64) / 1000.0)

    root_to_row = {r: i for i, r in enumerate(ecells["pt_root_id"].to_numpy())}
    pre_pos = pos_um[[root_to_row[r] for r in pre_ids]]
    post_pos = pos_um[[root_to_row[r] for r in post_ids]]

    # --- Connected (pre, post) pairs from the synapse table ------------------ #
    print("Loading synapse table ...")
    syn = read_feather(
        pjoin(DATA_DIR, f"syn_df_all_to_proofread_to_all_{MAT}.feather"),
        columns=["pre_pt_root_id", "post_pt_root_id"],
    )
    pre_id_set = set(pre_ids.tolist())
    post_id_set = set(post_ids.tolist())
    mask = syn["pre_pt_root_id"].isin(pre_id_set) & syn["post_pt_root_id"].isin(
        post_id_set
    )
    syn_ee = syn.loc[mask]
    # unique connected ordered pairs, excluding self-pairs
    conn_pairs = syn_ee.drop_duplicates(["pre_pt_root_id", "post_pt_root_id"])

    if buggy:
        #BUG 5: not removing self connection pairs
        conn_pairs = conn_pairs
    else:
        conn_pairs = conn_pairs[conn_pairs["pre_pt_root_id"] != conn_pairs["post_pt_root_id"]] 

    print(f"  synapses in E->E proofread subset: {len(syn_ee):,}")
    print(f"  unique connected E->E pairs:       {len(conn_pairs):,}")

    # pre root id -> array of post row indices it connects to
    post_id_to_row = {r: i for i, r in enumerate(post_ids)}
    connected_by_pre = {}
    for pr, grp in conn_pairs.groupby("pre_pt_root_id"):
        rows = [post_id_to_row[p] for p in grp["post_pt_root_id"].to_numpy()
                if p in post_id_to_row]
        connected_by_pre[pr] = np.array(rows, dtype=np.int64)

    # --- Accumulate distances into bins ------------------------------------- #
    edges = np.arange(0.0, MAX_DISTANCE_UM + BIN_WIDTH_UM, BIN_WIDTH_UM)
    n_bins = len(edges) - 1
    total_counts = np.zeros(n_bins, dtype=np.int64)      # denominator per bin
    connected_counts = np.zeros(n_bins, dtype=np.int64)  # numerator per bin

    print("Accumulating distances over all ordered pairs ...")
    for i, pre_root in enumerate(pre_ids):
        # distance from this pre cell to every post cell (um)
        d = np.linalg.norm(post_pos - pre_pos[i], axis=1)

        #BUG 5: not removing self connection pairs
        if not buggy:
            # exclude self-pair (same root id)
            self_row = post_id_to_row.get(pre_root, None)
            if self_row is not None:
                d[self_row] = np.nan  # will fall outside bins -> ignored 

        valid = (d < MAX_DISTANCE_UM) & ~np.isnan(d)
        bin_idx = np.floor(d[valid] / BIN_WIDTH_UM).astype(np.int64)
        total_counts += np.bincount(bin_idx, minlength=n_bins)[:n_bins]

        conn_rows = connected_by_pre.get(pre_root, None)
        if conn_rows is not None and len(conn_rows):
            cd = d[conn_rows]
            cvalid = (cd < MAX_DISTANCE_UM) & ~np.isnan(cd)
            cbin = np.floor(cd[cvalid] / BIN_WIDTH_UM).astype(np.int64)
            connected_counts += np.bincount(cbin, minlength=n_bins)[:n_bins]

        if (i + 1) % 100 == 0:
            print(f"    processed {i + 1}/{len(pre_ids)} presynaptic cells")

    # --- Per-bin probability + CI ------------------------------------------- #
    centers = (edges[:-1] + edges[1:]) / 2
    with np.errstate(invalid="ignore", divide="ignore"):

        if buggy:
            # BUG 6: denominator includes eligible pairs across all distance bins,
            # rather than only eligible pairs in the corresponding distance bin.
            all_eligible_pairs = total_counts.sum()
            prob = connected_counts / all_eligible_pairs
        else:
            # Correct: connected pairs in bin / eligible pairs in the same bin
            prob = np.where( total_counts > 0, connected_counts / total_counts, np.nan)


    if buggy:
        # BUG 7: use all eligible pairs as the denominator for every bin
        prob = connected_counts / total_counts.sum()
        ci_denominator = np.full_like(connected_counts,total_counts.sum())
        lo, hi = wilson_ci(connected_counts, ci_denominator)

    else:
        # Correct: denominator is restricted to each distance bin
        prob = np.where( total_counts > 0, connected_counts / total_counts, np.nan)
        lo, hi = wilson_ci(connected_counts, total_counts)


    table = pd.DataFrame(
        {
            "dist_bin_center_um": centers,
            "dist_bin_low_um": edges[:-1],
            "dist_bin_high_um": edges[1:],
            "n_pairs": total_counts,
            "n_connected": connected_counts,
            "connection_probability": prob,
            "ci95_low": lo,
            "ci95_high": hi,
        }
    )
    if buggy:
        csv_path = pjoin(OUT_DIR, "connection_probability_vs_distance_buggy.csv")
    else:
        csv_path = pjoin(OUT_DIR, "connection_probability_vs_distance.csv")
    table.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    overall_pairs = int(total_counts.sum())
    overall_conn = int(connected_counts.sum())
    overall_p = overall_conn / overall_pairs if overall_pairs else float("nan")

    # --- Plot --------------------------------------------------------------- #
    reliable = total_counts >= MIN_PAIRS_PER_BIN
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=200)

    m = reliable & np.isfinite(prob)
    ax.fill_between(centers[m], lo[m] * 100, hi[m] * 100, color="#4C72B0",
                    alpha=0.25, linewidth=0, label="95% CI (Wilson)")
    ax.plot(centers[m], prob[m] * 100, "-o", color="#2A4D80", ms=4, lw=1.6,
            label="E$\\to$E connection probability")

    # faded points for low-N bins (only labelled if any exist)
    mlow = (~reliable) & np.isfinite(prob) & (total_counts > 0)
    if mlow.any():
        ax.plot(centers[mlow], prob[mlow] * 100, "o", color="#2A4D80", ms=3,
                alpha=0.25, label=f"< {MIN_PAIRS_PER_BIN} pairs/bin")

    # ax.axhline(overall_p * 100, color="grey", ls="--", lw=1,
    #            label=f"overall = {overall_p * 100:.2f}%")

    ax.set_xlabel("Soma–soma distance (µm)")
    ax.set_ylabel("Connection probability (%)")
    ax.set_title(
        "Excitatory→Excitatory connection probability vs. distance\n"
        f"V1DD {MAT}  ·  {len(pre_ids)} proofread-axon E pre  ×  "
        f"{len(post_ids)} E post  ·  {overall_pairs:,} pairs"
    )
    ax.set_xlim(0, MAX_DISTANCE_UM)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    if buggy: 
        png_path = pjoin(OUT_DIR, "connection_probability_vs_distance_buggy.png")
        pdf_path = pjoin(OUT_DIR, "connection_probability_vs_distance_buggy.pdf")
    else:
        png_path = pjoin(OUT_DIR, "connection_probability_vs_distance.png")
        pdf_path = pjoin(OUT_DIR, "connection_probability_vs_distance.pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    print(f"Wrote {png_path}")

    # --- Summary ------------------------------------------------------------ #
    summary = f"""V1DD {MAT}: Excitatory -> Excitatory connection probability vs soma-soma distance
=================================================================================

Pools (fair-denominator requirement):
  presynaptic  = excitatory cells with a PROOFREAD AXON (complete outputs): {len(pre_ids)}
  postsynaptic = excitatory cells with a PROOFREAD DENDRITE (complete inputs): {len(post_ids)}

Pairs:
  ordered (pre, post) pairs, self-pairs excluded, distance < {MAX_DISTANCE_UM:.0f} um: {overall_pairs:,}
  connected pairs (>=1 synapse):                                                 {overall_conn:,}
  overall E->E connection probability:                                          {overall_p*100:.3f}%

Distance:
  3D Euclidean soma-soma distance in the cortically-aligned frame
  (pt_position_trform_{{x,y,z}}), converted nm -> um. Bin width = {BIN_WIDTH_UM:.0f} um.

Connectivity definition:
  binary - a pair is connected if at least one synapse exists from pre to post.

Caveats:
  * Even with proofread axons, axonal branches leaving the imaged volume can cause
    connection probability to be underestimated at large distances. Per-bin pair
    counts are reported (n_pairs) so sparse/unreliable bins are visible; bins with
    < {MIN_PAIRS_PER_BIN} pairs are de-emphasized in the plot.
  * Cell-type labels are automated (cell_type_multifeature_v1).
  * Distance uses the cortically-aligned (warped) coordinate frame; the raw EM frame
    gives near-identical distances since the transform is approximately rigid.

Output files:
  connection_probability_vs_distance.png / .pdf
  connection_probability_vs_distance.csv
"""
    if buggy:
        summary_path = pjoin(OUT_DIR, "summary_buggy.txt")
    else:
        summary_path = pjoin(OUT_DIR, "summary.txt")
    with open(summary_path, "w") as fh:
        fh.write(summary)
    print("\n" + summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--buggy",
        action="store_true",
        help="Run the intentionally buggy version."
    )
    args = parser.parse_args()

    main(buggy=args.buggy)