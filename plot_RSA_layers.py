import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy import stats
import matplotlib
import os

matplotlib.rc("font", size=24)

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change to a relative directory from the script's location
os.chdir(script_dir)

# 'alexnet', 'resnet50', 'densenet121', 'vgg16'
model_name = "slowfast_r50"  # 'x3d_m', 'slowfast_r50', 'dorsalnet'
# ['slowfast_r101', 'slowfast_16x8_r101_50_50', 'slowfast_4x16_r50']
statistics = "median"
pretrained = False
random_layer = ""

# Define correlation types, regions of interest, condition and status
cor_types = ["pearson"]  # , 'spearman']
ROIList = ["V1", "pFS", "LO", "EBA", "MTSTS", "infIPS", "SMG", "behavior"]
if "slow" in model_name:
    condition = ["slow without fusion", "slow", "fast", "fusion"]
else:
    condition = [""]
status = "dynamic"

ismedian = 1 if statistics == "median" else 0
random_initialized = "random/" if random_layer != "" else ""
is_cpc = "cpc/" if pretrained == "cpc" else ""
pretrained = "untrained/" if not pretrained else ""
imagenet = (
    "imagenet/"
    if (model_name in ["alexnet", "resnet50", "densenet121", "vgg16"])
    else ""
)

# Initialize an empty array for storing max layers
max_layer = np.empty((len(cor_types), len(ROIList)), dtype=object)

color = ["blue", "red"]
for cor in cor_types:
    # Create subplots for plotting
    fig, axes = plt.subplots(
        len(condition), len(ROIList), figsize=(56, 5 * len(condition) + 1), sharey=True
    )

    for c, cond in enumerate(condition):
        # Loop through regions of interest
        for r, region in enumerate(ROIList):
            if len(condition) == 1:
                ax = axes[r]
            else:
                ax = axes[c, r]
            ax.set_axisbelow(True)
            ax.grid(color="gray", linestyle=(10, (25, 10)), axis="y")
            ttest_data = []

            if cond == "slow without fusion":
                model_name = model_name.replace("slowfast_", "slow_")

            # Loop through dynamic and static status
            for s in range(2):
                data = []

                folder = f"result/RSA/{imagenet}{is_cpc}{pretrained}{random_initialized}{model_name}/{cor}/{region}/{random_layer}"
                if region == "behavior":
                    with open(f"{folder}S02_all_{status}_RSA.pkl", "rb") as File:
                        RSA = pickle.load(File)

                    data = list(RSA.values())
                    data = [tup[s] for tup in data]
                else:
                    # Loop through subjects
                    for sub in range(2, 18):
                        if sub == 8:
                            continue
                        subject = f"S{sub:02d}"

                        # Load RSA data from file
                        with open(
                            f"{folder}{subject}_all_{status}_RSA.pkl", "rb"
                        ) as File:
                            RSA = pickle.load(File)
                        data.append([tup[s] for tup in list(RSA.values())])

                data = np.array(data)
                filtered_list = list(RSA.keys())

                if "slow" in model_name:
                    # Filter RSA data based on the specified condition
                    if cond == "slow" or cond == "slow without fusion":
                        filtered_list = [
                            key
                            for key in filtered_list
                            if "multipathway_blocks.0" in key
                        ]
                    elif cond == "fast":
                        filtered_list = [
                            key
                            for key in filtered_list
                            if "multipathway_blocks.1" in key
                        ]
                    elif cond == "fusion":
                        filtered_list = [
                            key for key in filtered_list if "multipathway_fusion" in key
                        ]

                # Get the indices of selected keys
                indices = [
                    index
                    for index, key in enumerate(RSA.keys())
                    if key in filtered_list
                ]

                if region == "behavior":
                    data = data[indices]
                else:
                    data = data[:, indices]
                    # save for ttest
                    ttest_data.append(data)
                    SEM = stats.sem(data, axis=0)
                    # average across subjects
                    data = np.mean(data, axis=0)

                    upper_bound = data + SEM
                    lower_bound = data - SEM

                    ax.fill_between(
                        np.array(range(len(data))),
                        lower_bound,
                        upper_bound,
                        color=color[s],
                        alpha=0.5,
                        label=None,
                    )

                ax.plot(data, color=color[s], label=f"{status}")

                # ax.set_ylabel('Correlation (Kendal Tau)')
                # ax.xaxis.set_major_locator(ticker.NullLocator())
                if len(data) > 15:
                    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
                else:
                    ax.set_xticks(np.arange(len(data)))

                # tick = ax.get_xticklabels()
                # ax.set_xticklabels([])
                # if r != 7:
                #     ax.get_xaxis().set_visible(False)
                if r == 0:
                    ax.set_ylabel(f"{cond}", fontweight="bold", fontsize=30)
                if c == 0:
                    ax.set_title(f"{region}", fontweight="bold")

            # if region != 'behavior':
            #     # perform ttest
            #     _, p_values = stats.ttest_ind(ttest_data[0], ttest_data[1], axis=0)
            #     p_values = fdrcorrection(p_values, alpha=0.05)[1]

            #     sig_x = np.where(p_values < 0.05)[0]

            #     # plot ttest
            #     ax1_divider = make_axes_locatable(ax)
            #     # Add an Axes to the right of the main Axes.
            #     cax1 = ax1_divider.append_axes("bottom", size="10%", pad="2%", sharex=ax)
            #     cax1.eventplot(sig_x, colors = 'gray', linewidths = 600 / len(data))
            #     # cax1.set_xticklabels(tick)
            #     # axes[1, r].set_xticks([])
            #     cax1.set_yticks([])
            #     cax1.spines[['bottom', 'left', 'right', 'top']].set_visible(False)
            #     # axes[1, r].set_ylabel('p-value < 0.05', rotation='horizontal', ha='right')
            #     # axes[1, r].set_xlabel('label')

            if cond == "slow without fusion":
                model_name = model_name.replace("slow_", "slowfast_")

    h, l = ax.get_legend_handles_labels()
    fig.legend(
        handles=h, labels=l, ncols=2, loc="upper right", frameon=False, fontsize=30
    )
    plt.suptitle(model_name, fontweight="bold", fontsize=40)
    plt.tight_layout()
    file_path = f"plot/{imagenet}{is_cpc}{pretrained}{random_layer}{random_initialized}RSA/{model_name}_{cor}.png"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path)
    plt.close()
    print(file_path)
