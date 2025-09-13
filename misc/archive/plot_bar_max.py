import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change to a relative directory from the script's location
os.chdir(script_dir)

# Define constants
model_name = "x3d_m"
ROIList = ["V1", "EBA", "MTSTS", "SMG"]
width = 0.25  # Bar width

condition = [""]
if model_name == "slowfast_r50":
    condition = ["slow wo fusion", "slow", "fast"]

networks = [""]
if model_name == "slowfast_r50" or model_name == "res_r50":
    networks = ["k400", "ssv2"]

# Initialize arrays
num_conditions = len(condition)
num_rois = len(ROIList)
num_networks = len(networks)

max_layer = np.zeros((num_conditions, num_networks, num_rois))
max_layer_indices = np.zeros((num_conditions, num_networks, num_rois), dtype=int)

# Load data and calculate max layers
for n, network in enumerate(networks):
    datas = "ssv2/" if network == "ssv2" else ""

    for c, cond in enumerate(condition):
        for r, region in enumerate(ROIList):
            data = []
            folder = f"result/RSA/{datas}{model_name if cond != 'slow wo fusion' else model_name.replace('slowfast_', 'slow_')}/pearson/{region}/"
            if region == "behavior":
                with open(f"{folder}S02_all_dynamic_RSA.pkl", "rb") as File:
                    RSA = pickle.load(File)
                data = list(RSA.values())
                data = [tup[0] for tup in data]
            else:
                for sub in range(2, 18):
                    if sub == 8:
                        continue
                    subject = f"S{sub:02d}"
                    with open(f"{folder}{subject}_all_dynamic_RSA.pkl", "rb") as File:
                        RSA = pickle.load(File)
                    data.append([tup[0] for tup in list(RSA.values())])

            data = np.array(data)
            filtered_list = list(RSA.keys())

            if "slow" in model_name:
                # Filter RSA data based on the specified condition
                if cond == "slow" or cond == "slow wo fusion":
                    filtered_list = [
                        key for key in filtered_list if "multipathway_blocks.0" in key
                    ]
                elif cond == "fast":
                    filtered_list = [
                        key for key in filtered_list if "multipathway_blocks.1" in key
                    ]
                elif cond == "fusion":
                    filtered_list = [
                        key for key in filtered_list if "multipathway_fusion" in key
                    ]

            indices = [
                index for index, key in enumerate(RSA.keys()) if key in filtered_list
            ]

            if region == "behavior":
                data = data[indices]
            else:
                data = data[:, indices]
                data = np.mean(data, axis=0)

            max_layer[c, n, r] = np.max(data)
            max_layer_indices[c, n, r] = np.argmax(data)

if model_name == "slowfast_r50":
    # Plot settings for individual ROIs
    x = np.arange(num_conditions)  # X positions for conditions

    # Create subplots: One per ROI
    fig, axes = plt.subplots(3, num_rois // 2, figsize=(18, 8), sharey=True)
    axes = axes.flatten()

    for roi_idx, ax in enumerate(axes):
        if roi_idx >= 7:
            continue
        data = max_layer[:, :, roi_idx]
        indices = max_layer_indices[:, :, roi_idx]

        for net_idx, network in enumerate(networks):
            bars = ax.bar(
                x + net_idx * width - width / 2, data[:, net_idx], width, label=network
            )
            for bar, index in zip(bars, indices[:, net_idx]):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{index}",
                    ha="center",
                    va="bottom",
                )

        ax.set_title(f"ROI: {ROIList[roi_idx]}", fontsize=14)
        ax.set_xticks(x + width)
        ax.set_xticklabels(condition)
        if roi_idx == 0 or roi_idx == 3:
            ax.set_ylabel("Max Layer's RSA", fontsize=12)

    axes[0].legend(title="Network")

elif model_name == "res_r50":
    x = np.arange(num_rois)
    fig, ax = plt.subplots(figsize=(10, 6))

    data = max_layer[0, :, :]
    indices = max_layer_indices[0, :, :]

    for net_idx, network in enumerate(networks):
        bars = ax.bar(
            x + net_idx * width - width / 2, data[net_idx, :], width, label=network
        )
        for bar, index in zip(bars, indices[net_idx, :]):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{index}",
                ha="center",
                va="bottom",
            )

    ax.set_title("res_r50", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(ROIList)
    ax.set_xlabel("ROI", fontsize=12)
    ax.set_ylabel("Max Layer's RSA", fontsize=12)
    ax.legend()
else:
    # Plot settings for DorsalNet
    x = np.arange(num_rois)  # X positions for ROIs

    # Create a new figure
    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar(x, max_layer[0, 0, :], width)

    for bar, index in zip(bars, max_layer_indices[0, 0, :]):
        height = bar.get_height()
        height = max(0, height)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(index)}",
            ha="center",
            va="bottom",
        )

    ax.set_title(f"{model_name}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(ROIList)
    ax.set_xlabel("ROI", fontsize=12)
    ax.set_ylabel("Max Layer's RSA", fontsize=12)

plt.tight_layout()
plt.savefig(f"plot/RSA/barplot/{model_name}.png")
