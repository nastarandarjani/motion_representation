import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kendalltau

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change to a relative directory from the script's location
os.chdir(script_dir)


def calculate_RSA(RDM1, RDM2):
    RDM1 = RDM1[np.triu_indices(RDM1.shape[0], k=0)]
    RDM2 = RDM2[np.triu_indices(RDM2.shape[0], k=0)]

    correlation, _ = kendalltau(RDM1, RDM2)
    return correlation


def LOO(rdms):
    correlations = []
    for i in range(len(rdms)):
        mask = np.ones(len(rdms), dtype=bool)
        mask[i] = False

        correlation = calculate_RSA(rdms[i], np.mean(rdms[mask], axis=0))
        correlations.append(correlation)

    return np.mean(correlations), stats.sem(correlations)


def compute_noise_ceiling(rdm_behavior_all, model_cm_all_folds):
    # r1: behavioral reliability
    # r1_values = LOO(np.array(rdm_behavior_all))

    # r2: model reliability
    model_rdms = []
    for cm in model_cm_all_folds:
        rdm_model = (cm + cm.T) / 2
        rdm_model = 1 - (rdm_model / rdm_model.max())
        model_rdms.append(rdm_model)

    r2_values, re = LOO(np.array(model_rdms))

    return r2_values, re


model_names = ["slowfast_r50", "slow_r50", "res_r50"]
RSA_means = []
RSA_sems = []
NC_means = []
NC_sems = []

RDM_folder = "result/fMRI RDM/pearson/behavior"
with open(f"{RDM_folder}/S02_RDM_dynamic.pkl", "rb") as File:
    rdm_behavior_all = pickle.load(File)
rdm_behavior = np.mean(rdm_behavior_all, axis=0)

for model_name in model_names:
    with open(f"result/confusions/{model_name}.pkl", "rb") as File:
        cm = pickle.load(File)

    nc_avg, nc_sem = compute_noise_ceiling(rdm_behavior_all, cm)
    NC_means.append(nc_avg)
    NC_sems.append(nc_sem)

    rsa_all_folds = []

    for fold_cm in cm:
        # Symmetrize and convert to RDM
        rdm_model = (fold_cm + fold_cm.T) / 2
        rdm_model = 1 - (rdm_model / rdm_model.max())

        rsa = calculate_RSA(rdm_behavior, rdm_model)
        rsa_all_folds.append(rsa)

    RSA_means.append(np.mean(rsa_all_folds))
    RSA_sems.append(stats.sem(rsa_all_folds))

# Plotting
fig, ax = plt.subplots()
x = np.arange(len(model_names))

# Plot RSA bars
bars = ax.bar(
    x,
    RSA_means,
    yerr=RSA_sems,
    color=["tab:green", "tab:orange", "tab:blue"],
)

ax.axhline(y=0, color="black")

# Plot noise ceiling
ax.errorbar(
    x, NC_means, yerr=NC_sems, fmt="o", color="black", label="Noise Ceiling", capsize=2
)

ax.set_ylabel("RSA (Kendall’s τ)")
ax.set_title("Model RDM vs Behavioral RDM (Kendall’s τ)")
ax.set_xticks(x)
ax.set_xticklabels(["S_wx + F", "S_nox", "S_1"])
plt.tight_layout()
plt.savefig("plot/coding.png", dpi=300)
