import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy import stats
import os
import itertools
from utils.util import compute_noise_ceiling, filter_rsa_data, init_plot, bootstraping

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change to a relative directory from the script's location
os.chdir(script_dir)

status = "dynamic"


def load_rsa_data(region, model_name, hem):
    folder = f"result/RSA/{model_name}/pearson/{region}/"
    data = []

    if region == "behavior":
        with open(f"{folder}S02_{status}_RSA.pkl", "rb") as File:
            RSA = pickle.load(File)
        data = [[tup[1] for tup in rsa_dict.values()] for rsa_dict in RSA]
        RSA = RSA[0]
    else:
        for sub in range(1, 18):
            if sub == 1:
                continue
            if sub == 8:
                continue
            subject = f"S{sub:02d}"
            with open(f"{folder}{subject}_{hem}_{status}_RSA.pkl", "rb") as File:
                RSA = pickle.load(File)
            data.append([tup[1] for tup in RSA.values()])

    return np.array(data), RSA


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


def main(color, condition):
    hemm = "all"

    row = len(condition)
    column = len(ROIList)
    fig, axes = plt.subplots(
        row,
        column,
        figsize=(7, 1.3 * row),
        sharey=True,
    )
    axes = np.atleast_2d(axes)
    roi = ROIList.copy()

    handles = []
    for m, conds in enumerate(condition):
        for r, region in enumerate(ROIList):
            ax = axes[m, r]
            ax.set_axisbelow(True)
            ax.grid(False)
            ttest_data = []

            if "SMG" in region:
                region, hem = region.split("_")
                roi[r] = rf"${{\text{{{region}}}}}_{{{hem}}}$"
            else:
                hem = hemm

            if region == "MTSTS":
                roi[r] = "${\\text{LOT}}_{bio}$"

            for c, cond in enumerate(conds):
                if cond in ["S_wx", "F_wx"]:
                    model_name = "slowfast_r50"
                elif cond == "S_nox":
                    model_name = "slow_r50"
                elif cond == "S_1":
                    model_name = "res_r50"
                else:
                    model_name = "dorsalnet"
                data, RSA = load_rsa_data(region, model_name, hem)
                filtered_list = filter_rsa_data(RSA, cond)

                indices = [
                    i for i, key in enumerate(RSA.keys()) if key in filtered_list
                ]

                filtered_data = data[:, indices]
                ttest_data.append(filtered_data)
                SEM = stats.sem(filtered_data, axis=0)
                filtered_data = np.mean(filtered_data, axis=0)

                if cond == conds[0]:
                    nc_avg, nc_sem = compute_noise_ceiling(region, hem)
                    # ax.fill_between(
                    #     range(len(filtered_data)),
                    #     nc_avg - nc_sem,
                    #     nc_avg + nc_sem,
                    #     color="gray",
                    #     alpha=0.2,
                    # )
                    ax.plot(
                        range(len(filtered_data)),
                        np.ones((len(filtered_data))) * nc_avg,
                        color="gray",
                        lw=1,
                    )
                    # ax.plot(
                    #     range(len(filtered_data)),
                    #     np.ones((len(filtered_data))) * (nc_avg - nc_sem),
                    #     color="gray",
                    #     lw=0.2,
                    # )
                    # ax.plot(
                    #     range(len(filtered_data)),
                    #     np.ones((len(filtered_data))) * (nc_avg + nc_sem),
                    #     color="gray",
                    #     lw=0.2,
                    # )

                ax.fill_between(
                    range(len(filtered_data)),
                    filtered_data - SEM,
                    filtered_data + SEM,
                    color=color[m][c],
                    alpha=0.3,
                    lw=0.8,
                )

                ax.plot(filtered_data, color=color[m][c], label=f"{cond}", lw=0.8)

            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

            ax.axhline(y=0, color="black", lw=0.8, ls="--")
            ax.tick_params(axis="x", direction="in", length=2)
            ax.tick_params(axis="y", direction="in", length=2)

            if m == 0:
                ax.set_title(f"{roi[r]}")
                for fusion in [1, 4, 8, 14]:
                    ax.axvline(x=fusion, color="red", lw=0.5)
            else:
                ax.set_xticks(range(0, len(filtered_data), 2))

            ax.set_ylim(-0.35, 0.75)
            ax.set_yticks(np.arange(-0.3, 0.75, 0.2))
            ax.set_xlim(0, len(filtered_data) - 1)

            combinations = list(itertools.combinations(range(len(conds)), 2))
            if not combinations:
                combinations = [0]
            sig = np.zeros((len(combinations), len(filtered_data)))
            for count, ind in enumerate(combinations):
                if isinstance(ind, tuple):
                    sig[count, :] = bootstraping(ttest_data[ind[0]], ttest_data[ind[1]])
                else:
                    sig[count, :] = bootstraping(ttest_data[ind])
            plot_significance(ax, sig, count)

            latex_conditions = [
                r"${\text{" + c.replace("_", r"}}_{") + r"}$" if "_" in c else c
                for group in condition
                for c in group
            ]
            h, _ = ax.get_legend_handles_labels()
            handles.extend(h)

    fig.legend(
        handles=[handles[0], handles[1], handles[-1]],
        labels=latex_conditions,
        ncols=3,
        loc="lower right",
        frameon=False,
        columnspacing=0.8,
        handletextpad=0.3,
    )

    # fig.suptitle(datas, fontweight="bold")
    fig.supylabel("Correlation (Kendall's Tau)", fontweight="bold", ha="center")
    fig.supxlabel("ReLU Layers", fontweight="bold")
    plt.tight_layout(pad=0.5)
    plt.savefig("plot/fig1.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    init_plot()

    condition = [["S_wx", "F_wx"], ["DorsalNet"]]
    color = [["tab:orange", "tab:blue"], ["tab:green"]]

    # condition = ["S_nox", "S_wx"]
    # color = ["tab:green", "tab:orange"]

    # condition = ["S_nox", "S_1"]
    # color = ["tab:green", "tab:red"]

    ROIList = ["V1", "EBA", "MTSTS", "SMG_lh"]

    main(color, condition)
