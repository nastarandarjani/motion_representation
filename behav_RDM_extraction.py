import os
import pickle

import numpy as np
import scipy
from tqdm import tqdm

from utils.util import calculate_RSA

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change to a relative directory from the script's location
os.chdir(script_dir)

names = [
    "reptile",
    "ball",
    "reptile",
    "ball",
    "ball",
    "ball",
    "reptile",
    "quadruped_mammal",
    "reptile",
    "PS",
    "PS",
    "ball",
    "quadruped_mammal",
    "ball",
    "human",
    "tool",
    "quadruped_mammal",
    "PS",
    "PS",
    "human",
    "quadruped_mammal",
    "PS",
    "quadruped_mammal",
    "human",
    "human",
    "reptile",
    "quadruped_mammal",
    "PS",
    "reptile",
    "human",
    "human",
    "tool",
    "tool",
    "tool",
    "tool",
    "tool",
]
names = np.array(names)


def compute_rdm(data):
    num_triplet = data.shape[0] // 3
    RDM = np.zeros((36, 36))
    count_matrix = np.zeros((36, 36))

    for tr in range(num_triplet):
        for i in range(3):
            ind1 = data[tr * 3 + i, 0] - 1
            ind2 = data[tr * 3 + i, 1] - 1
            sim = data[tr * 3 + i, 2]
            RDM[ind1, ind2] += sim
            RDM[ind2, ind1] += sim
            count_matrix[ind1, ind2] += 1
            count_matrix[ind2, ind1] += 1

    RDM /= count_matrix + np.eye(36)
    return RDM


def sort_ind(names):
    category_order = {
        "human": 0,
        "quadruped_mammal": 1,
        "reptile": 2,
        "tool": 3,
        "PS": 4,
        "ball": 5,
    }

    ind = np.argsort([category_order.get(x) for x in names])
    return ind

if __name__ == "__main__":
    data = scipy.io.loadmat("O1O.mat")
    data = data["table"]

    B = 1000
    bootstrap_rdms = np.zeros((B, 36, 36))

    ind = sort_ind(names)
    for b in tqdm(range(B)):
        sampled_indices = np.random.choice(
            data.shape[0] // 3, size=data.shape[0] // 3, replace=True
        )
        resampled_data = np.vstack([data[i * 3 : (i + 1) * 3] for i in sampled_indices])
        bootstrap_rdm = compute_rdm(resampled_data)
        bootstrap_rdm = bootstrap_rdm[ind, :][:, ind]
        bootstrap_rdms[b, :, :] = bootstrap_rdm

    bootstrap_rdms = bootstrap_rdms.reshape(B, 6, 6, 6, 6)
    bootstrap_rdms = np.mean(bootstrap_rdms, axis=2)
    bootstrap_rdms = np.mean(bootstrap_rdms, axis=3)

    bootstrap_rdms = bootstrap_rdms[:, [5, 0, 1, 4, 2, 3], :]
    bootstrap_rdms = bootstrap_rdms[:, :, [5, 0, 1, 4, 2, 3]]

    RDM_folder = "result/fMRI RDM/pearson/behavior"
    with open(f"{RDM_folder}/S02_RDM_dynamic.pkl", "wb") as File:
        pickle.dump(bootstrap_rdms, File)

    # List of models, correlation types, regions of interest, and hemispheres
    models = ["slow_r50", "slowfast_r50", "res_r50"]  # , 'slow_r50', 'dorsalnet']
    dataset = "k400"
    pretrained = True
    random_layer = ""  #'fusion/'
    isimagenet = False
    correlation_types = ["pearson"]
    names = {"": [""]}  # , '_anim' : ['_animate', '_inanimate']}

    if isimagenet:
        models = ["alexnet", "resnet50", "densenet121", "vgg16"]
        pretrained = True

    random_initialized = "random/" if random_layer != "" else ""
    imagenet = "imagenet/" if isimagenet else ""
    is_cpc = "cpc/" if pretrained == "cpc" else ""
    pretrained = "untrained/" if not pretrained else ""
    data = f"{dataset}/" if not (dataset == "k400") else ""
    # Loop through subjects

    for cor in correlation_types:
        bootstrap_rdms

        for model_name in models:
            # Construct the save folder path
            save_folder = f"result/RSA/{imagenet}{data}{is_cpc}{pretrained}{random_initialized}{model_name}/{cor}/behavior/{random_layer}"

            # Create the save folder if it doesn't exist
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            RDM2 = bootstrap_rdms

            model_path = f"result/model RDM/{imagenet}{data}{is_cpc}{pretrained}{random_initialized}dynamic/{random_layer}{cor}_RDM_{model_name}.pkl"
            with open(model_path, "rb") as pickle_file:
                RDM1 = pickle.load(pickle_file)

            RSA = []
            for rdm2 in tqdm(RDM2):
                RSA_b = {}
                for layer_name, RDM in RDM1.items():
                    if "slow_" in model_name and "multipathway_fusion" in layer_name:
                        continue
                    RSA_b[layer_name] = calculate_RSA(RDM, rdm2)
                RSA.append(RSA_b)

            with open(
                f"{save_folder}S02_dynamic_RSA.pkl",
                "wb",
            ) as File:
                pickle.dump(RSA, File)
