from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
import itertools
from utils.util import bootstraping, filter_rsa_data, compute_noise_ceiling, init_plot
from plot_RSA_layers import load_rsa_data


# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change to a relative directory from the script's location
os.chdir(script_dir)

ROIList = ["V1", "EBA", "MTSTS", "SMG_lh"]
status = "dynamic"


def main(color):
    hem = "all"
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(7 / 2, 2.5),
    )

    br = np.arange(len(ROIList))
    bar = np.zeros(len(condition))
    err = np.zeros(len(condition))
    barWidth = 0.25

    for r, region in enumerate(ROIList):
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

            data, RSA = load_rsa_data(region, model_name, hem)
            filtered_list = filter_rsa_data(RSA, cond)

            indices = [i for i, key in enumerate(RSA.keys()) if key in filtered_list]

            filtered_data = data[:, indices]
            ttest_data.append(filtered_data)
            SEM = stats.sem(filtered_data[:, 12], axis=0)
            filtered_data = np.mean(filtered_data[:, 12], axis=0)

            bar[c] = filtered_data
            err[c] = SEM

            ax.bar(
                br[r] + barWidth * c,
                bar[c],
                yerr=err[c],
                color=color[c],
                label=cond,
                width=barWidth,
            )

            if cond == condition[0]:
                nc_avg, nc_sem = compute_noise_ceiling(region, hem)
                # ax.fill_between(
                #     [br[r] - barWidth, br[r] + barWidth * 3],
                #     nc_avg - nc_sem,
                #     nc_avg + nc_sem,
                #     color="gray",
                #     alpha=0.2,
                #     edgecolor="none",
                #     zorder=2,
                # )
                ax.plot(
                    [br[r] - barWidth / 2, br[r] + barWidth / 2 * 5],
                    [nc_avg, nc_avg],
                    color="dimgrey",
                    lw=1,
                )

        print(region, bar, err)

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        ax.tick_params(axis="x", direction="in", length=2)
        ax.tick_params(axis="y", direction="in", length=2)
        ax.axhline(y=0, color="black", lw=0.8)

        combinations = list(itertools.combinations(range(len(condition)), 2))
        combinations = sorted(combinations, key=lambda x: abs(x[0] - x[1]))
        y_pre = np.zeros((len(condition)))
        for count, ind in enumerate(combinations):
            significant_layers = bootstraping(ttest_data[ind[0]], ttest_data[ind[1]])
            if significant_layers[12] > 0:
                x1 = br[r] + barWidth * ind[0]
                x2 = br[r] + barWidth * ind[1]
                y1 = bar[ind[0]] + err[ind[0]]
                y2 = bar[ind[1]] + err[ind[1]]
                y1 = max(y1, y_pre[ind[0]], y1)
                y2 = max(y_pre[ind[1]], y2)
                y_max = max(y1, y2)
                y = [y_max + 0.02, y_max + 0.04, y_max + 0.04, y_max + 0.02]
                ax.plot([x1, x1, x2, x2], y, color="black", linewidth=0.75, zorder=0)
                ax.add_patch(
                    Rectangle(
                        ((x1 + x2) / 2 - 0.075, y[1] - 0.005),
                        0.15,
                        0.01,
                        color="white",
                        zorder=1,
                    )
                )
                ax.text(
                    (x1 + x2) / 2,
                    y[1] - 0.015,
                    "*",
                    ha="center",
                    va="center",
                    zorder=2,
                )
                for i in range(ind[0], ind[1] + 1):
                    y_pre[i] = y[1]

        if region == "MTSTS":
            ROIList[r] = "${\\text{LOT}}_{bio}$"

    ax.set_xticks(br + barWidth)
    ax.set_xticklabels(ROIList)

    ax.set_xlim(0 - barWidth, 4)
    ax.set_ylim(-0.21, 0.71)

    latex_conditions = [
        r"${\text{" + item.replace("_", r"}}_{") + r"}$" for item in condition
    ]

    h, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles=h[: len(condition)],
        labels=latex_conditions,
        ncols=1,
        loc="upper right",
        frameon=False,
        labelspacing=0.1,
        handletextpad=0.3,
    )

    fig.supylabel("Correlation (Kendall's Tau)", fontweight="bold")
    # fig.suptitle("Layer 12", fontweight="bold")
    # fig.supxlabel("Regions", fontweight="bold")
    plt.tight_layout(pad=0.5)
    plt.savefig("plot/fig2.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    init_plot()

    condition = ["S_wx", "S_nox", "S_1"]
    color = ["tab:orange", "tab:red", "tab:purple"]

    main(color)
