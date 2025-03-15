import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy import stats
import os
from statsmodels.stats.multitest import multipletests
import itertools
from scipy.stats import kendalltau

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change to a relative directory from the script's location
os.chdir(script_dir)

ROIList = ["V1", "pFS", "LO", "EBA", "MTSTS", "infIPS", "SMG", "behavior"]


def bootstraping(data_A, data_B, num_samples=1000):
    def my_statistic(sample1, sample2, axis=-1):
        return np.mean(sample1, axis=axis) - np.mean(sample2, axis=axis)

    p_values = []
    for i in range(data_A.shape[1]):
        observed_diff = my_statistic(data_A[:, i], data_B[:, i])

        # Combine and create null distribution
        combined = np.concatenate([data_A[:, i], data_B[:, i]])
        null_distribution = []

        for _ in range(num_samples):
            np.random.shuffle(combined)
            sample1 = combined[: len(data_A[:, i])]
            sample2 = combined[len(data_A[:, i]) :]
            null_distribution.append(my_statistic(sample1, sample2))

        null_distribution = np.array(null_distribution)

        # Compute two-tailed p-value
        p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_diff))
        p_values.append(p_value)

    # Apply FDR correction (Benjamini-Hochberg)
    rejected, fdr_corrected_p, _, _ = multipletests(p_values, method="fdr_bh")

    return p_values, fdr_corrected_p, rejected


def load_rsa_data(region, model_name, data_path):
    folder = f"result/RSA/{data_path}{model_name}/pearson/{region}/"
    data = []

    if region == "behavior":
        with open(f"{folder}S02_all_dynamic_RSA.pkl", "rb") as File:
            RSA = pickle.load(File)
        data = [tup[1] for tup in RSA.values()]
    else:
        for sub in range(2, 18):
            if sub == 8:
                continue
            subject = f"S{sub:02d}"
            with open(f"{folder}{subject}_all_dynamic_RSA.pkl", "rb") as File:
                RSA = pickle.load(File)
            data.append([tup[1] for tup in RSA.values()])

    return np.array(data), RSA


def filter_rsa_data(RSA, cond):
    def is_valid_key(key):
        return "act_a" not in key and "act_b" not in key

    if cond in ["SF-S", "SF-S-noF"]:
        return [
            key
            for key in RSA.keys()
            if "multipathway_blocks.0" in key and is_valid_key(key)
        ]
    elif cond == "SF-F":
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


def compute_noise_ceiling(region):
    fmri_data = []
    for sub in range(2, 18):
        if sub == 8:
            continue
        subject = f"S{sub:02d}"

        RDM_folder = f"result/fMRI RDM/pearson/{region}"
        with open(f"{RDM_folder}/{subject}_RDM_all_dynamic.pkl", "rb") as File:
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

    return ((np.min(correlations), np.max(correlations)), np.mean(correlations))


def main(color):
    for datas in dataset:
        fig, axes = plt.subplots(
            2,
            len(ROIList) // 2,
            figsize=(7, 2.63),
            sharey=True,
            sharex=True,
        )
        axes = axes.flatten()
        data_path = f"{datas}/" if datas != "k400" else ""

        for r, region in enumerate(ROIList):
            ax = axes[r]
            ax.set_axisbelow(True)
            ax.grid(True, lw=0.3)
            ttest_data = []

            for c, cond in enumerate(condition):
                if cond in ["SF-S", "SF-F"]:
                    model_name = "slowfast_r50"
                elif cond == "SF-S-noF":
                    model_name = "slow_r50"
                else:
                    model_name = "res_r50"
                data, RSA = load_rsa_data(region, model_name, data_path)
                filtered_list = filter_rsa_data(RSA, cond)

                indices = [
                    i for i, key in enumerate(RSA.keys()) if key in filtered_list
                ]

                if region == "behavior":
                    filtered_data = data[indices]
                else:
                    filtered_data = data[:, indices]
                    ttest_data.append(filtered_data)
                    SEM = stats.sem(filtered_data, axis=0)
                    filtered_data = np.mean(filtered_data, axis=0)

                    if cond == condition[0]:
                        (nc_low, nc_high), nc_avg = compute_noise_ceiling(region)
                        ax.fill_between(
                            range(len(filtered_data)),
                            nc_low,
                            nc_high,
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
                            np.ones((len(filtered_data))) * nc_high,
                            color="gray",
                            lw=0.2,
                        )
                        ax.plot(
                            range(len(filtered_data)),
                            np.ones((len(filtered_data))) * nc_low,
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

            ax.set_title(f"{region}", fontweight="bold", pad=1)
            ax.tick_params(axis="x", direction="in", length=2)
            ax.tick_params(axis="y", direction="in", length=2)

            for fusion in [1, 4, 8, 14]:
                ax.axvline(x=fusion, color="red")

            # ax.set_ylim(-0.3, 0.4)
            ax.set_xlim(0, len(filtered_data) - 1)

            if region != "behavior":
                combinations = list(itertools.combinations(range(len(condition)), 2))
                sig = np.zeros((len(combinations), len(filtered_data)))
                for count, ind in enumerate(combinations):
                    _, _, sig[count, :] = bootstraping(
                        ttest_data[ind[0]], ttest_data[ind[1]]
                    )
                plot_significance(ax, sig, count)

                h, l = ax.get_legend_handles_labels()
                fig.legend(
                    handles=h,
                    labels=l,
                    ncols=2,
                    loc="lower right",
                    frameon=False,
                )

        # fig.suptitle(datas, fontweight="bold")
        fig.supylabel("Correlation (Kendall's Tau)", fontweight="bold")
        plt.tight_layout(pad=0.5)
        fig.supxlabel("Layers", fontweight="bold")
        fig.subplots_adjust(bottom=0.12)

    plt.savefig(f"plot/{datas}_{condition}.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "axes.labelsize": 8,  # Axis labels
            "axes.titlesize": 8,  # Titles
            "xtick.labelsize": 7,  # Tick labels
            "ytick.labelsize": 7,  # Tick labels
            "legend.fontsize": 8,  # Legend text
            "figure.titlesize": 8,  # Suptitle (if used)
            "figure.labelsize": 8,
        }
    )

    dataset = ["k400"]

    condition = ["SF-S", "SF-F"]
    color = ["tab:orange", "tab:blue"]

    # condition = ["SF-S-noF", "SF-S"]
    # color = ["tab:green", "tab:orange"]

    # condition = ["SF-S-noF", "S-only"]
    # color = ["tab:green", "tab:red"]
    main(color)
