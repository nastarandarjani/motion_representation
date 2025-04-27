from tkinter import font
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy import stats
import os
import itertools
from scipy.stats import kendalltau
from mne.stats import permutation_cluster_test
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from nilearn.plotting import plot_surf_roi

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change to a relative directory from the script's location
os.chdir(script_dir)

ROIList = ["V1", "pFS", "LO", "EBA", "MTSTS", "infIPS", "SMG_lh"]
status = "dynamic"


def bootstraping(data_A, data_B):
    def my_statistic(data_A, data_B):
        return np.mean(data_A, axis=0) - np.mean(data_B, axis=0)

    T_obs, clusters, cluster_p_values, _ = permutation_cluster_test(
        [data_A, data_B],
        n_permutations=1000,
        tail=0,
        stat_fun=my_statistic,
        out_type="mask",
        threshold=0.1,
        verbose=False,
    )

    significant = np.zeros(T_obs.shape, dtype=bool)
    for i, p_value in enumerate(cluster_p_values):
        if p_value < 0.05:
            significant[clusters[i]] = True

    return significant


def load_rsa_data(region, model_name, data_path, hem):
    folder = f"result/RSA/{data_path}{model_name}/pearson/{region}/"
    data = []

    if region == "behavior":
        with open(f"{folder}S02_{status}_RSA.pkl", "rb") as File:
            RSA = pickle.load(File)
        data = [[tup[1] for tup in rsa_dict.values()] for rsa_dict in RSA]
        RSA = RSA[0]
    else:
        for sub in range(2, 18):
            if sub == 8:
                continue
            subject = f"S{sub:02d}"
            with open(f"{folder}{subject}_{hem}_{status}_RSA.pkl", "rb") as File:
                RSA = pickle.load(File)
            data.append([tup[1] for tup in RSA.values()])

    return np.array(data), RSA


def filter_rsa_data(RSA, cond):
    def is_valid_key(key):
        return "act_a" not in key and "act_b" not in key

    if cond in ["S_wx", "S_nox"]:
        return [
            key
            for key in RSA.keys()
            if "multipathway_blocks.0" in key and is_valid_key(key)
        ]
    elif cond == "F_wx":
        return [
            key
            for key in RSA.keys()
            if "multipathway_blocks.1" in key and is_valid_key(key)
        ]
    elif cond == "fusion":
        return [
            key
            for key in RSA.keys()
            if "multipathway_fusion" in key and is_valid_key(key)
        ]

    return [key for key in RSA.keys() if is_valid_key(key)]


def plot_significance(ax, sig, count):
    consecutive_significant_indices = {i: [] for i in range(count + 1)}
    single_significant_indices = {i: [] for i in range(count + 1)}

    for col in range(count + 1):
        significant_indices = np.where(sig[col, :])[0]
        if len(significant_indices) == 1:
            single_significant_indices[col].append(significant_indices[0])
        elif len(significant_indices) == 0:
            continue
        else:
            start = significant_indices[0]
            for i in range(1, len(significant_indices)):
                if significant_indices[i] != significant_indices[i - 1] + 1:
                    if start == significant_indices[i - 1]:
                        single_significant_indices[col].append(start)
                    else:
                        consecutive_significant_indices[col].append(
                            (start, significant_indices[i - 1])
                        )
                    start = significant_indices[i]
            if start == significant_indices[-1]:
                single_significant_indices[col].append(start)
            else:
                consecutive_significant_indices[col].append(
                    (start, significant_indices[-1])
                )

    for col in range(count + 1):
        for start_idx, end_idx in consecutive_significant_indices[col]:
            ax.plot(
                np.arange(start_idx, end_idx + 1),
                np.zeros((end_idx - start_idx + 1)) + col - 0.25,
                color="black",
                lw=1.3,
            )
        for idx in single_significant_indices[col]:
            ax.scatter(
                idx, col - 0.25, color="black", facecolor="black", s=0.5, marker="o"
            )


def compute_noise_ceiling(region, hem):
    if region == "behavior":
        RDM_folder = f"result/fMRI RDM/pearson/{region}"
        with open(f"{RDM_folder}/S02_RDM_{status}.pkl", "rb") as File:
            dynamic_RDM = pickle.load(File)

        triu_indices = np.triu_indices(dynamic_RDM.shape[1], k=1)
        fmri_data = dynamic_RDM[:, triu_indices[0], triu_indices[1]]
    else:
        fmri_data = []
        for sub in range(2, 18):
            if sub == 8:
                continue
            subject = f"S{sub:02d}"

            RDM_folder = f"result/fMRI RDM/pearson/{region}"
            with open(f"{RDM_folder}/{subject}_RDM_{hem}_{status}.pkl", "rb") as File:
                dynamic_RDM = pickle.load(File)

            fmri_data.append(dynamic_RDM[np.triu_indices(dynamic_RDM.shape[0], k=1)])
        fmri_data = np.array(fmri_data)

    correlations = []
    for i in range(fmri_data.shape[0]):
        mask = np.ones(fmri_data.shape[0], dtype=bool)
        mask[i] = False

        correlation, _ = kendalltau(
            fmri_data[i, :], np.mean(fmri_data[mask, :], axis=0)
        )
        correlations.append(correlation)

    return np.mean(correlations), stats.sem(correlations)


def main(hem, color):
    for datas in dataset:
        fig, axes = plt.subplots(
            2,
            int(np.ceil(len(ROIList) / 2)),
            figsize=(7, 2.25),
            sharey=True,
            sharex=True,
        )
        fig.canvas.draw()

        data_path = f"{datas}/" if datas != "k400" else ""

        cmap = cm.get_cmap("cold_hot", len(ROIList) + 2)

        for r, region in enumerate(ROIList):
            row, col = divmod(r + 1, 4)
            ax = axes[row, col]
            ax.set_axisbelow(True)
            ax.grid(False)
            ttest_data = []

            if "SMG" in region:
                region, hem = region.split("_")
                ROIList[r] = rf"${{\text{{{region}}}}}_{{{hem}}}$"
            else:
                hem = "all"

            for c, cond in enumerate(condition):
                if cond in ["S_wx", "F_wx"]:
                    model_name = "slowfast_r50"
                elif cond == "S_nox":
                    model_name = "slow_r50"
                else:
                    model_name = "res_r50"
                data, RSA = load_rsa_data(region, model_name, data_path, hem)
                filtered_list = filter_rsa_data(RSA, cond)

                indices = [
                    i for i, key in enumerate(RSA.keys()) if key in filtered_list
                ]

                filtered_data = data[:, indices]
                ttest_data.append(filtered_data)
                SEM = stats.sem(filtered_data, axis=0)
                filtered_data = np.mean(filtered_data, axis=0)

                if cond == condition[0]:
                    nc_avg, nc_sem = compute_noise_ceiling(region, hem)
                    ax.fill_between(
                        range(len(filtered_data)),
                        nc_avg - nc_sem,
                        nc_avg + nc_sem,
                        color="gray",
                        alpha=0.2,
                    )
                    ax.plot(
                        range(len(filtered_data)),
                        np.ones((len(filtered_data))) * nc_avg,
                        color="gray",
                        lw=1,
                    )
                    ax.plot(
                        range(len(filtered_data)),
                        np.ones((len(filtered_data))) * (nc_avg - nc_sem),
                        color="gray",
                        lw=0.2,
                    )
                    ax.plot(
                        range(len(filtered_data)),
                        np.ones((len(filtered_data))) * (nc_avg + nc_sem),
                        color="gray",
                        lw=0.2,
                    )

                ax.fill_between(
                    range(len(filtered_data)),
                    filtered_data - SEM,
                    filtered_data + SEM,
                    alpha=0.3,
                    label=f"{cond}",
                    color=color[c],
                    lw=0.8,
                )

                ax.plot(filtered_data, label=None, color=color[c], lw=0.8)

            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

            ax.axhline(y=0, color="black", lw=0.8, ls="--")

            if region == "MTSTS":
                ROIList[r] = "${\\text{LOT}}_{bio}$"

            text_x = 0.5
            text_y = 1.02

            text_obj = ax.text(
                text_x,
                text_y,
                ROIList[r],
                transform=ax.transAxes,
                fontsize=8,
                ha="center",
                va="bottom",
            )

            renderer = fig.canvas.get_renderer()
            bbox = text_obj.get_window_extent(renderer=renderer)

            bbox_data = ax.transAxes.inverted().transform(
                [[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]]
            )
            text_height = bbox_data[1][1] - bbox_data[0][1]

            patch_x = bbox_data[0][0] - 0.1
            patch_y = text_y + text_height / 3

            rect = mpatches.Rectangle(
                (patch_x, patch_y),
                0.08,
                text_height / 2,
                transform=ax.transAxes,
                facecolor=cmap(r + 1),
                edgecolor="black",
                linewidth=0.3,
                fill=True,
                clip_on=False,
            )
            ax.add_patch(rect)

            # ax.set_title(f"{ROIList[r]}", pad=1)
            ax.tick_params(axis="x", direction="in", length=2)
            ax.tick_params(axis="y", direction="in", length=2)

            for fusion in [1, 4, 8, 14]:
                ax.axvline(x=fusion, color="red", lw=0.5)

            ax.set_ylim(-0.35, 0.75)
            ax.set_xlim(0, len(filtered_data) - 1)

            combinations = list(itertools.combinations(range(len(condition)), 2))
            sig = np.zeros((len(combinations), len(filtered_data)))
            for count, ind in enumerate(combinations):
                sig[count, :] = bootstraping(ttest_data[ind[0]], ttest_data[ind[1]])
            plot_significance(ax, sig, count)

            latex_conditions = [
                r"$" + item.replace("_", r"_{") + r"}$" for item in condition
            ]
            h, _ = ax.get_legend_handles_labels()
            fig.legend(
                handles=h,
                labels=latex_conditions,
                ncols=2,
                loc="lower right",
                frameon=False,
                columnspacing=0.8,
                handletextpad=0.3,
            )

        # fig.suptitle(datas, fontweight="bold")
        fig.supylabel("Correlation (Kendall's Tau)", fontweight="bold")
        plt.tight_layout(pad=0.5)
        axes[0, 1].yaxis.set_tick_params(labelleft=True)
        fig.supxlabel("ReLU Layers", fontweight="bold")
        fig.subplots_adjust(bottom=0.13)

        axes[0, 0].axis("off")
        axes[0, 0].text(
            0,
            1.15,
            "A",
            transform=axes[0, 0].transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        axes[0, 1].text(
            -0.15,
            1.15,
            "B",
            transform=axes[0, 1].transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )

    plt.savefig(f"plot/{datas}_{hem}_{condition}.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "axes.labelsize": 8,  # Axis labels
            "axes.titlesize": 8,  # Titles
            "xtick.labelsize": 7,  # Tick labels
            "ytick.labelsize": 7,  # Tick labels
            "legend.fontsize": 7,  # Legend text
            "figure.titlesize": 8,  # Suptitle (if used)
            "figure.labelsize": 8,
        }
    )

    dataset = ["k400"]
    hem = "all"

    condition = ["S_wx", "F_wx"]
    color = ["tab:orange", "tab:blue"]

    # condition = ["S_nox", "S_wx"]
    # color = ["tab:green", "tab:orange"]

    # condition = ["S_nox", "S_1"]
    # color = ["tab:green", "tab:red"]
    main(hem, color)
