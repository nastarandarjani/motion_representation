from scipy import stats
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from scipy.stats import rankdata, kendalltau
from utils.util import filter_rsa_data, compute_noise_ceiling, init_plot

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change to a relative directory from the script's location
os.chdir(script_dir)


def partial_corr(y, x1, x2, rank_transform=True):
    if rank_transform:
        y = rankdata(y)
        x1 = rankdata(x1)
        x2 = rankdata(x2)

    r_yx1, _ = kendalltau(y, x1)
    r_yx2, _ = kendalltau(y, x2)
    r_x1x2, _ = kendalltau(x1, x2)

    num = r_yx1 - r_yx2 * r_x1x2
    den = np.sqrt((1 - r_yx2**2) * (1 - r_x1x2**2))
    return num / den if den > 0 else 0.0


init_plot()
ROIList = ["V1", "EBA", "MTSTS", "SMG_lh"]

RDM_folder = "result/model RDM/dynamic"
with open(f"{RDM_folder}/pearson_RDM_slowfast_r50.pkl", "rb") as File:
    slowfast_model = pickle.load(File)
slowfast_model = {
    k: v[np.triu_indices(v.shape[0], k=1)] for k, v in slowfast_model.items()
}

with open(f"{RDM_folder}/pearson_RDM_dorsalnet.pkl", "rb") as File:
    dorsalnet_model = pickle.load(File)
dorsalnet_model = {
    k: v[np.triu_indices(v.shape[0], k=1)] for k, v in dorsalnet_model.items()
}

filtered_list = filter_rsa_data(slowfast_model, "S_wx")
indices = [i for i, key in enumerate(slowfast_model.keys()) if key in filtered_list]
slowfast_model = np.array(list(slowfast_model.values()))[indices, :]

filtered_list = filter_rsa_data(dorsalnet_model, "dorsalnet")
indices = [i for i, key in enumerate(dorsalnet_model.keys()) if key in filtered_list]
dorsalnet_model = np.array(list(dorsalnet_model.values()))[indices, :]

data = []
R2 = []
cor = []
data_sem = []
noise_ceiling = []
hemm = "all"

for r, region in enumerate(ROIList):
    if "SMG" in region:
        region, hem = region.split("_")
        ROIList[r] = rf"${{\text{{{region}}}}}_{{{hem}}}$"
    else:
        hem = hemm

    nc_avg, _ = compute_noise_ceiling(region, hem)
    noise_ceiling.append(nc_avg)

    fmri_data = []
    for sub in range(1, 18):
        if sub == 1 or sub == 8:
            continue
        subject = f"S{sub:02d}"

        RDM_folder = f"result/fMRI RDM/pearson/{region}"
        with open(f"{RDM_folder}/{subject}_RDM_{hem}_dynamic.pkl", "rb") as File:
            dynamic_RDM = pickle.load(File)

        fmri_RDM = dynamic_RDM[np.triu_indices(dynamic_RDM.shape[0], k=1)]
        fmri_data.append(fmri_RDM)

    if region == "MTSTS":
        ROIList[r] = "${\\text{LOT}}_{bio}$"

    R2_total = np.zeros((len(fmri_data)))
    result = np.zeros((len(fmri_data), 2))
    result_cor = np.zeros((len(fmri_data), 2))
    for f, fmri_RDM in enumerate(fmri_data):
        l1 = slowfast_model[10, :]
        l2 = dorsalnet_model[2, :]
        result[f, 0] = partial_corr(fmri_RDM, l1, l2)
        result[f, 1] = partial_corr(fmri_RDM, l2, l1)

        r_yx1, _ = kendalltau(fmri_RDM, l1)
        r_yx2, _ = kendalltau(fmri_RDM, l2)
        result_cor[f, :] = [r_yx1, r_yx2]
        r_x1x2, _ = kendalltau(l1, l2)
        R2_total[f] = (r_yx1**2 + r_yx2**2 - 2 * r_yx1 * r_yx2 * r_x1x2) / (
            1 - r_x1x2**2
        )

    semm = stats.sem(result, axis=0)
    r = np.mean(result, axis=0)
    result_cor = np.mean(result_cor, axis=0)
    R2_total = np.mean(R2_total)

    data.append(r)
    data_sem.append(semm)
    cor.append(result_cor)
    R2.append(R2_total)

data = np.array(data)
cor = np.array(cor)
data_sem = np.array(data_sem)

print(cor, data, data_sem)

n_groups = data.shape[0]
n_bars = data.shape[1]

x = np.arange(n_groups)
width = 0.3
bottom1 = np.zeros(n_groups)
bottom2 = np.zeros(n_groups)

fig, ax = plt.subplots(figsize=(7 / 2, 2.5))

color = ["tab:orange", "tab:green"]

ax.bar(
    x - width / 2,
    cor[:, 0],
    width,
    color="white",
    label=r"$\text{S}_{wx}$",
    hatch="///",
    linewidth=1,
    edgecolor=color[0],
)
ax.bar(
    x - width / 2,
    data[:, 0],
    width,
    yerr=data_sem[:, 0],
    label=r"$\text{S}_{wx}$ | DorsalNet",
    color=color[0],
)
ax.bar(
    x + width / 2,
    cor[:, 1],
    width,
    color="white",
    label="DorsalNet",
    hatch="///",
    linewidth=1,
    edgecolor=color[1],
)
ax.bar(
    x + width / 2,
    data[:, 1],
    width,
    yerr=data_sem[:, 1],
    label=r"DorsalNet | $\text{S}_{wx}$",
    color=color[1],
)

for i, nc in enumerate(noise_ceiling):
    ax.hlines(nc, i - width, i + width, linewidth=1, color="gray")

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(ROIList)
ax.set_ylabel("Correlation (Kendall's Tau)", fontweight="bold")

ax.legend(
    loc="upper left",
    frameon=False,
    labelspacing=0.1,
    handletextpad=0.3,
)

for spine in ax.spines.values():
    spine.set_linewidth(0.5)

ax.axhline(y=0, color="black", lw=0.8)
ax.tick_params(axis="x", direction="in", length=2)
ax.tick_params(axis="y", direction="in", length=2)
plt.tight_layout(pad=0.5)
plt.savefig("plot/fig3.png", dpi=300, bbox_inches="tight")
