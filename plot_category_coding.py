import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from utils.util import calculate_RSA, init_plot

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change to a relative directory from the script's location
os.chdir(script_dir)


def LOO(rdms, k=1):
    correlations = []
    for i in range(len(rdms)):
        mask = np.ones(len(rdms), dtype=bool)
        mask[i] = False

        correlation = calculate_RSA(
            rdms[i], np.mean(rdms[mask], axis=0), k=k, bootstrap=False
        )
        correlations.append(correlation)

    return np.mean(correlations), stats.sem(correlations)


def compute_noise_ceiling(rdm_behavior_all, model_cm_all_folds):
    upper_bound_behavior, _ = LOO(np.array(rdm_behavior_all))

    model_rdms = []
    for cm in model_cm_all_folds:
        rdm_model = (cm + cm.T) / 2
        rdm_model = 1 - (rdm_model / rdm_model.max())
        model_rdms.append(rdm_model)

    upper_bound_model, _ = LOO(np.array(model_rdms))

    nc = np.sqrt(upper_bound_behavior * upper_bound_model)

    return nc

if __name__ == "__main__":
    init_plot()
    mpl.rcParams["hatch.linewidth"] = 2.5

    model_names = [
        "both",
        "slowfast_r50",
        "fast_r50",
        "slow_r50",
        "res_r50",
        "dorsalnet",
    ]

    acc_means = []
    acc_sems = []
    RSA_means = []
    RSA_sems = []
    NC_means = []

    RDM_folder = "result/fMRI RDM/pearson/behavior"
    with open(f"{RDM_folder}/S02_RDM_dynamic.pkl", "rb") as File:
        rdm_behavior_all = pickle.load(File)
    rdm_behavior = np.mean(rdm_behavior_all, axis=0)

    for model_name in model_names:
        with open(f"result/confusions/{model_name}.pkl", "rb") as File:
            cm = pickle.load(File)

        acc_all_folds = []
        rsa_all_folds = []

        for fold_cm in cm:
            acc = np.diag(fold_cm) / np.sum(fold_cm, axis=1)
            acc = np.mean(acc)
            acc_all_folds.append(acc)

            rdm_model = (fold_cm + fold_cm.T) / 2
            rdm_model = 1 - (rdm_model / rdm_model.max())
            rsa = calculate_RSA(rdm_behavior, rdm_model, bootstrap=False)
            rsa_all_folds.append(rsa)

        acc_all_folds = np.array(acc_all_folds)

        acc_means.append(np.mean(acc_all_folds, axis=0))
        acc_sems.append(stats.sem(acc_all_folds, axis=0))

        nc_avg = compute_noise_ceiling(rdm_behavior_all, cm)
        NC_means.append(nc_avg)

        RSA_means.append(np.mean(rsa_all_folds))
        RSA_sems.append(stats.sem(rsa_all_folds))

    acc_means = np.array(acc_means)
    acc_sems = np.array(acc_sems)

    fig, ax = plt.subplots(1, 2, figsize=(7, 2.5))

    # Plotting
    x = np.arange(len(model_names))
    width = 0.5

    files = sorted([f for f in os.listdir("stimuli") if f.startswith("processed_")])
    colors = [
        "tab:orange",
        "tab:orange",
        "tab:blue",
        "tab:red",
        "tab:purple",
        "tab:green",
    ]
    plt.rcParams["hatch.color"] = "tab:blue"
    bars = ax[0].bar(x, acc_means * 100, width, yerr=acc_sems * 100, color=colors)
    bars[0].set_hatch("///")

    ax[0].axhline(y=(6 / 36) * 100, color="black", lw=0.8, ls="--")

    ax[0].set_ylabel("Accuracy (Percentage)", fontweight="bold")
    ax[0].set_title("Network Accuracy", fontweight="bold")
    ax[0].set_xticks(x)

    print(acc_means * 100, acc_sems * 100)

    x = np.arange(len(model_names))

    # Plot RSA bars
    bars = ax[1].bar(x, RSA_means, width, yerr=RSA_sems, color=colors)
    bars[0].set_hatch("///")

    print(RSA_means, RSA_sems)

    ax[1].axhline(y=0, color="black", lw=0.8)

    # Plot noise ceiling
    for i, nc in enumerate(NC_means):
        ax[1].hlines(
            nc, i - width / 2, i + width / 2, colors="k", linewidth=1, color="gray"
        )

    ax[1].set_ylabel("Correlation (Kendall's Tau)", fontweight="bold")
    ax[1].set_title("Similarity of Model and Behavioral Data", fontweight="bold")
    ax[1].set_xticks(x)

    for i in range(2):
        for spine in ax[i].spines.values():
            spine.set_linewidth(0.5)
        ax[i].tick_params(axis="x", direction="in", length=2)
        ax[i].tick_params(axis="y", direction="in", length=2)
        ax[i].set_xticklabels(
            [
                r"$\text{S}_{wx}$ + F",
                r"$\text{S}_{wx}$",
                "F",
                r"$\text{S}_{nox}$",
                r"$\text{S}_1$",
                "DorsalNet",
            ]
        )

    plt.tight_layout(pad=0.5)
    plt.savefig("plot/fig5.png", dpi=300, bbox_inches="tight")