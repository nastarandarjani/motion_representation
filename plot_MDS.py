import itertools
import os
import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from scipy.spatial import procrustes
from sklearn.manifold import MDS

from compute_RSA import calculate_RSA
from utils.util import filter_rsa_data, init_plot

init_plot()

ROIList = ["V1", "EBA", "MTSTS", "SMG_lh"]
roi = ["V1", "EBA", "${\\text{LOT}}_{bio}$", "${\\text{SMG}}_{lh}$"]
condition = ["F_wx", "S_wx", "S_nox", "S_1", "dorsal"]
conds = ["F$_{wx}$", "S$_{wx}$", "S$_{nox}$", "S$_1$", "Dorsal"]

if os.path.exists("mds.npy"):
    aligned = np.load("mds.npy")
else:
    MRI_RDMs = np.zeros((15, len(ROIList), 6, 6))

    nsub = 0
    for sub in range(2, 18):
        if sub == 8:
            continue
        subject = f"S{sub:02d}"

        for r, region in enumerate(ROIList):
            hem = "all"

            if "SMG" in region:
                region, hem = region.split("_")

            RDM_folder = f"result/fMRI RDM/pearson/{region}"

            if region == "behavior":
                with open(f"{RDM_folder}/RDM_dynamic.pkl", "rb") as File:
                    dynamic_RDM = pickle.load(File)
            else:
                with open(
                    f"{RDM_folder}/{subject}_RDM_{hem}_dynamic.pkl", "rb"
                ) as File:
                    dynamic_RDM = pickle.load(File)

            MRI_RDMs[nsub, r, :, :] = dynamic_RDM
        nsub += 1

    model_RDMs = np.zeros((len(condition), 6, 6))

    for c, cond in enumerate(condition):
        if cond in ["S_wx", "F_wx"]:
            model_name = "slowfast_r50"
        elif cond == "S_nox":
            model_name = "slow_r50"
        elif cond == "S_1":
            model_name = "res_r50"
        else:
            model_name = "dorsalnet"

        model_path = f"result/model RDM/dynamic/pearson_RDM_{model_name}.pkl"
        with open(model_path, "rb") as pickle_file:
            model_RDM = pickle.load(pickle_file)

        filtered_list = filter_rsa_data(model_RDM, cond)
        indices = [i for i, key in enumerate(model_RDM.keys()) if key in filtered_list]

        model_RDM = np.array(list(model_RDM.values()))
        model_RDM = model_RDM[indices, :, :]
        if "dorsal" in model_name:
            model_RDM = model_RDM[2, :, :]
        else:
            model_RDM = model_RDM[10, :, :]

        model_RDMs[c, :, :] = model_RDM

    model_RDMs = np.repeat(model_RDMs[np.newaxis, :, :, :], 15, axis=0)
    RDMs = np.concatenate((MRI_RDMs, model_RDMs), axis=1)
    whole_RSA = np.ones(
        (15, len(ROIList) + len(condition), len(ROIList) + len(condition))
    )
    combinations = list(itertools.combinations(range(len(ROIList) + len(condition)), 2))
    for s in range(15):
        for c1, c2 in combinations:
            RSA, _ = calculate_RSA(RDMs[s, c1, :, :], RDMs[s, c2, :, :])
            whole_RSA[s, c1, c2] = whole_RSA[s, c2, c1] = RSA

    RDMs = 1 - whole_RSA

    coords_all = []
    for subj in RDMs:
        mds = MDS(
            n_components=2, dissimilarity="precomputed", metric=False, random_state=42
        )
        coords = mds.fit_transform(subj)
        coords_all.append(coords)
        print(f"Stress: {mds.stress_}")
    coords_all = np.array(coords_all)

    ref = coords_all[0]
    aligned = []
    for coords in coords_all:
        mtx1, mtx2, disparity = procrustes(ref, coords)
        aligned.append(mtx2)
    aligned = np.array(aligned)

    np.save("mds.npy", aligned)


theta = np.pi / 2
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

# apply rotation across last dimension
aligned = aligned @ R.T  # shape (15, 9, 2)

coords_mean = np.mean(aligned, axis=0)

fig = plt.figure(figsize=(7 / 2, 4))

colors = ["tab:pink", "tab:blue", "tab:olive", "tab:cyan"]

for subj_coords in aligned:
    for i in range(len(ROIList + condition)):
        if i < len(ROIList):
            face_color = to_rgba(colors[i], alpha=0.3)
            edge_color = to_rgba(colors[i], alpha=0.4)
            plt.scatter(
                subj_coords[i, 0],
                subj_coords[i, 1],
                edgecolors=edge_color,
                s=30,
                color=face_color,
            )

edge_color = to_rgba("gray", alpha=0.6)
plt.scatter(
    coords_mean[: len(ROIList), 0],
    coords_mean[: len(ROIList), 1],
    c=colors,
    s=180,
    edgecolor=edge_color,
)

plt.scatter(
    coords_mean[len(ROIList) :, 0],
    coords_mean[len(ROIList) :, 1],
    c="none",
    s=180,
    alpha=0.6,
    edgecolor="gray",
)

for i, txt in enumerate(conds):
    text = plt.text(
        coords_mean[len(ROIList) + i, 0],
        coords_mean[len(ROIList) + i, 1],
        txt,
        ha="center",
        va="center",
    )

# ax_scatter.set_xlim(-0.23, 0.27)
plt.gca().set_aspect("auto", adjustable="box")
plt.gca().set_box_aspect(1)
# ax_scatter.axis('square')

plt.xticks([])
plt.yticks([])

for spine in plt.gca().spines.values():
    spine.set_visible(False)

handles = [mpatches.Patch(facecolor=colors[i], label=roi[i]) for i in range(len(roi))]

plt.legend(
    handles,
    roi,
    ncols=3,
    loc="lower center",
    frameon=False,
    bbox_to_anchor=(0.5, -0.05),
    columnspacing=0.8,
    handletextpad=0.3,
)
# plt.savefig("plot/fig4.png", dpi=300, bbox_inches="tight")
plt.show()
