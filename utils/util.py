import pickle
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mne.stats import permutation_cluster_test
from pytorchvideo.models.hub import r2plus1d, slowfast
from scipy import stats
from scipy.stats import kendalltau

from DorsalNet.dorsalnet import DorsalNet


def calculate_RSA(RDM1, RDM2, k=1, bootstrap=True):
    RDM1 = RDM1[np.triu_indices(RDM1.shape[0], k=k)]
    RDM2 = RDM2[np.triu_indices(RDM2.shape[0], k=k)]

    if bootstrap:
        # Calculate Kendall's Tau correlation between the two RDMs
        correlations = []
        for _ in range(100):
            ind = np.random.choice(15, size=15, replace=True)
            correlation, _ = kendalltau(RDM1[ind], RDM2[ind])
            # correlation = np.corrcoef(RDM1, RDM2)[0][1]

            correlations.append(correlation)

        return (np.mean(correlations), np.median(correlations))
    else:
        correlation, _ = kendalltau(RDM1, RDM2)
        return correlation


def compute_noise_ceiling(region, hem):
    status = "dynamic"
    if region == "behavior":
        RDM_folder = f"result/fMRI RDM/pearson/{region}"
        with open(f"{RDM_folder}/S02_RDM_{status}.pkl", "rb") as File:
            dynamic_RDM = pickle.load(File)

        triu_indices = np.triu_indices(dynamic_RDM.shape[1], k=1)
        fmri_data = dynamic_RDM[:, triu_indices[0], triu_indices[1]]
    else:
        fmri_data = []
        for sub in range(1, 18):
            if sub == 1:
                continue
            if sub == 8:
                continue
            subject = f"S{sub:02d}"

            RDM_folder = f"result/fMRI RDM/pearson/{region}"
            with open(f"{RDM_folder}/{subject}_RDM_{hem}_{status}.pkl", "rb") as File:
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

    return np.mean(correlations), stats.sem(correlations)


def filter_rsa_data(RSA, cond=None):
    def is_valid_key(key):
        return "act_a" not in key and "act_b" not in key

    if cond in ["S_wx", "S_nox"]:
        return [
            key
            for key in RSA.keys()
            if "multipathway_blocks.0" in key and is_valid_key(key)
        ]
    elif cond == "F_wx":
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
    elif cond == "S_1":
        return [key for key in RSA.keys() if is_valid_key(key)]

    return RSA.keys()


def load_model(model_name, pretrained=True, dataset="k400"):
    """
    Load a pre-trained PyTorchVideo model.

    Args:
        model_name (str): Name of the model to load.
        pretrained (bool): Whether to load pretrained weights.
        dataset (str): Dataset name for loading specific weights.

    Returns:
        torch.nn.Module: Loaded pre-trained model.
    """
    if model_name == "dorsalnet":
        network = (
            "airsim_dorsalnet_batch2_model.ckpt-3174400-2021-02-12 02-03-29.666899.pt"
        )

        checkpoint = torch.load(
            f"DorsalNet/{network}", map_location=torch.device("cpu")
        )

        subnet_dict = {}
        for k, v in checkpoint.items():
            if k.startswith("fully_connected"):
                continue
            if k.startswith("subnet.") or k.startswith("module."):
                subnet_dict[k[7:]] = v
            else:
                subnet_dict[k] = v

        model = DorsalNet(False, 32)
        if pretrained:
            model.load_state_dict(subnet_dict)
    elif model_name == "vjepa_v2":
        model = torch.hub.load(
            "facebookresearch/vjepa", "vjepa_v2_vit_large", pretrained=pretrained
        )
    elif model_name == "slowfast_4x16_r50":

        def slowfast_4x16_r50(
            pretrained: bool = False,
            progress: bool = True,
            **kwargs: Any,
        ) -> nn.Module:
            return slowfast._slowfast(
                pretrained=pretrained,
                progress=progress,
                checkpoint_path="https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST_4x16_R50.pyth",
                model_depth=50,
                slowfast_fusion_conv_kernel_size=(5, 1, 1),
                slowfast_fusion_conv_stride=(8, 1, 1),
                head_pool_kernel_sizes=((4, 7, 7), (32, 7, 7)),
                **kwargs,
            )

        model = slowfast_4x16_r50(pretrained=True)
    elif model_name == "R2PLUS1D":
        model = r2plus1d.r2plus1d_r50(pretrained=pretrained, progress=True)
    elif pretrained == "cpc":
        model = torch.hub.load(
            "facebookresearch/pytorchvideo", model_name, pretrained=False
        )
        checkpoint = torch.load(
            "../epoch_0010_best.ckpt",
            map_location=torch.device("cpu"),
        )

        state_dict = {}
        old_prefix = "network.backbone.model."
        new_prefix = "blocks."
        for key, value in checkpoint["state_dict"].items():
            if key.startswith(old_prefix):
                # Replace the prefix
                new_key = new_prefix + key[len(old_prefix) :]
            else:
                new_key = key
            state_dict[new_key] = value

        model.load_state_dict(state_dict, strict=False)
    elif model_name == "res_r50":
        if dataset == "k400":
            model = torch.hub.load(
                "facebookresearch/pytorchvideo", "slow_r50", pretrained=pretrained
            )
        else:
            model = torch.hub.load(
                "facebookresearch/pytorchvideo", "slow_r50", pretrained=False
            )
            weight_path = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/ssv2/SLOW_8x8_R50.pyth"
    elif (
        model_name == "slow_r50"
        or model_name == "fast_r50"
        or model_name == "slowfast_r50"
    ):
        if dataset == "k400":
            model = torch.hub.load(
                "facebookresearch/pytorchvideo", "slowfast_r50", pretrained=pretrained
            )
        else:
            model = torch.hub.load(
                "facebookresearch/pytorchvideo", "slowfast_r50", pretrained=False
            )
            weight_path = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/ssv2/SLOWFAST_8x8_R50.pyth"

            state_dict = torch.hub.load_state_dict_from_url(
                weight_path, map_location="cuda"
            )
            filtered_state_dict = {
                k: v
                for k, v in state_dict["model_state"].items()
                if not k.startswith("blocks.6.proj")
            }
            model.load_state_dict(filtered_state_dict, strict=False)

    return model


def init_plot():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "axes.labelsize": 9,  # Axis labels
            "axes.titlesize": 9,  # Titles
            "xtick.labelsize": 8,  # Tick labels
            "ytick.labelsize": 8,  # Tick labels
            "legend.fontsize": 8,  # Legend text
            "figure.titlesize": 9,  # Suptitle (if used)
            "figure.labelsize": 9,
        }
    )


def bootstraping(data_A, data_B=0):
    oneside = False
    if isinstance(data_B, (int, float)) and data_B == 0:
        oneside = True
        data_B = np.zeros_like(data_A)

    def my_statistic(data_A, data_B):
        return np.mean(data_A, axis=0) - np.mean(data_B, axis=0)

    T_obs, clusters, cluster_p_values, _ = permutation_cluster_test(
        [data_A, data_B],
        n_permutations=1000,
        tail=int(oneside),
        stat_fun=my_statistic,
        out_type="mask",
        threshold=0.1,
        verbose=False,
    )

    significant = np.zeros(T_obs.shape, dtype=bool)
    for i, p_value in enumerate(cluster_p_values):
        if p_value < 0.05:
            significant[clusters[i]] = True

    return significant
