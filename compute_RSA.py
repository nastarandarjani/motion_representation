"""This script performs Representational Similarity Analysis (RSA) on fMRI data and behavioral dissimilarity data.
It calculates Representational Dissimilarity Matrices (RDMs) for different brain regions and compares them with model RDMs.

Functions:
    load_MRI(filepath, hemisphere):

    load_behav():

    calculate_RDM(response_patterns, method="euclidean"):

    calculate_RSA(RDM1, RDM2):

    calculate_RSA_layers(RDM1, RDM2, mode, ind):

    filter_RDM(RDM, mode):

Variables:
    models (list): List of model names to be used for RSA.
    dataset (str): Name of the dataset.
    pretrained (bool): Flag indicating whether the models are pretrained.
    random_layer (str): Path to the random layer.
    isimagenet (bool): Flag indicating whether the dataset is ImageNet.
    correlation_types (list): List of correlation types to be used for RSA.
    ROIList (list): List of regions of interest.
    hemispheres (list): List of hemispheres to be analyzed.
    names (dict): Dictionary mapping modes to names.

The script loops through subjects, regions of interest, hemispheres, models, and correlation types to calculate and save RDMs and RSA values.
"""

import pickle
from scipy.stats import kendalltau
from scipy.stats import spearmanr
import scipy.io
import os
from tqdm import tqdm
import numpy as np


# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change to a relative directory from the script's location
os.chdir(script_dir)


def load_MRI(filepath, hemisphere):
    """
    Load data from a MAT file using the specified hemisphere identifier.

    Parameters:
    filepath (str): The path to the MAT file.
    hemisphere (str): The hemisphere identifier, either 'all', 'left', or 'right'.

    Returns:
    tuple: A tuple containing two numpy arrays, dynamic_tstat and static_tstat.
    """
    # Load data from MAT file using the specified hemisphere identifier
    if hemisphere == "all":
        hemisphere = hemisphere.capitalize()
    else:
        hemisphere = hemisphere.upper()
    key = f"condData{hemisphere}"
    data = scipy.io.loadmat(filepath)[key]

    # Separate dynamic and static RDM
    dynamic_tstat = np.mean(data[:6, :, :], axis=1)
    static_tstat = np.mean(data[6:, :, :], axis=1)

    # swap rows to the desired form
    dynamic_tstat = dynamic_tstat[[5, 0, 1, 4, 2, 3], :]

    static_tstat = static_tstat[[5, 0, 1, 4, 2, 3], :]

    return dynamic_tstat, static_tstat


def load_behav():
    """
    Loads and processes behavioral dissimilarity data from CSV files.

    The function reads two CSV files, "dissimilarity_img2.csv" and "dissimilarity_vid2.csv",
    which contain static and dynamic behavioral dissimilarity data, respectively. The data
    is reshaped, averaged, and reordered to match a specific format.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - static_behav (numpy.ndarray): Processed static behavioral dissimilarity data.
            - dynamic_behav (numpy.ndarray): Processed dynamic behavioral dissimilarity data.
    """
    static_behav = np.genfromtxt("dissimilarity_img2.csv", delimiter=",")[1:, :]
    static_behav = static_behav.reshape(6, 6, 6, 6)
    static_behav = np.mean(static_behav, axis=1)
    static_behav = np.mean(static_behav, axis=2)

    dynamic_behav = np.genfromtxt("dissimilarity_vid2.csv", delimiter=",")[1:, :]
    dynamic_behav = dynamic_behav.reshape(6, 6, 6, 6)
    dynamic_behav = np.mean(dynamic_behav, axis=1)
    dynamic_behav = np.mean(dynamic_behav, axis=2)

    static_behav = static_behav[[5, 0, 1, 4, 2, 3], :]
    static_behav = static_behav[:, [5, 0, 1, 4, 2, 3]]

    dynamic_behav = dynamic_behav[[5, 0, 1, 4, 2, 3], :]
    dynamic_behav = dynamic_behav[:, [5, 0, 1, 4, 2, 3]]
    return static_behav, dynamic_behav


def calculate_RDM(response_patterns, method="euclidean"):
    """
    Calculate the Representational Dissimilarity Matrix (RDM) for given response patterns.

    Parameters:
    response_patterns (numpy.ndarray): A 2D array where each row represents a response pattern.
    method (str): The method to use for calculating the RDM. Options are "euclidean", "spearman", and "pearson".
                  Default is "euclidean".

    Returns:
    numpy.ndarray: A 2D array representing the RDM.

    Methods:
    - "euclidean": Computes the RDM using Euclidean distance.
    - "spearman": Computes the RDM using Spearman correlation.
    - "pearson": Computes the RDM using Pearson correlation.
    """
    if method == "euclidean":
        # Compute RDM using vectorized operations
        pairwise_differences = (
            response_patterns[:, np.newaxis, :] - response_patterns[np.newaxis, :, :]
        )
        rdm = np.linalg.norm(pairwise_differences, axis=2)
    elif method == "spearman":
        cor, _ = spearmanr(response_patterns, axis=1)
        rdm = 1 - cor
    elif method == "pearson":
        rdm = 1 - np.corrcoef(response_patterns)

    return rdm


def calculate_RSA(RDM1, RDM2):
    """
    Calculate the Representational Similarity Analysis (RSA) between two Representational Dissimilarity Matrices (RDMs).

    This function computes the Kendall's Tau correlation between the upper triangular parts of two RDMs, excluding the diagonal.
    It performs bootstrapping by randomly sampling with replacement to estimate the correlation distribution.

    Parameters:
    RDM1 (numpy.ndarray): The first Representational Dissimilarity Matrix.
    RDM2 (numpy.ndarray): The second Representational Dissimilarity Matrix.

    Returns:
    tuple: A tuple containing the mean and median of the bootstrapped Kendall's Tau correlations.
    """
    RDM1 = RDM1[np.triu_indices(RDM1.shape[0], k=1)]
    RDM2 = RDM2[np.triu_indices(RDM2.shape[0], k=1)]

    # Calculate Kendall's Tau correlation between the two RDMs
    correlations = []
    for _ in range(100):
        ind = np.random.choice(15, size=15, replace=True)
        correlation, _ = kendalltau(RDM1[ind], RDM2[ind])
        # correlation = np.corrcoef(RDM1, RDM2)[0][1]

        correlations.append(correlation)

    return (np.mean(correlations), np.median(correlations))


def calculate_RSA_layers(RDM1, RDM2, mode, ind):
    """
    Calculate Representational Similarity Analysis (RSA) for each layer.

    Args:
        RDM1 (dict): A dictionary where keys are layer names and values are Representational Dissimilarity Matrices (RDMs).
        RDM2 (ndarray): A Representational Dissimilarity Matrix to compare against.
        mode (str): A mode to filter the RDMs.
        ind (int): An index to select specific elements from the filtered RDM.

    Returns:
        dict: A dictionary where keys are layer names and values are the calculated RSA values.
    """
    global model_name
    RSA = {}
    for layer_name, RDM in RDM1.items():
        if "slow_" in model_name and "multipathway_fusion" in layer_name:
            continue
        RDM = filter_RDM(RDM, mode)[ind]
        RSA[layer_name] = calculate_RSA(RDM, RDM2)
    return RSA


def filter_RDM(RDM, mode):
    """
    Filters and processes a Representational Dissimilarity Matrix (RDM) based on the specified mode.

    Parameters:
    RDM (numpy.ndarray): A 2D array representing the RDM to be filtered.
    mode (str): A string indicating the mode of filtering. It can be one of the following:
        - "": Returns the reordered RDM.
        - "_anim": Returns two submatrices, one for animate objects and one for inanimate objects.
        - "_stim": Returns a tuple of diagonal elements of the RDM.

    Returns:
    tuple: Depending on the mode, it returns:
        - (RDM,) if mode is "".
        - (animate, inanimate) if mode is "_anim", where animate and inanimate are submatrices of the RDM.
        - RDM_tuple if mode is "_stim", where RDM_tuple is a tuple of diagonal elements of the RDM.
    """
    temp = RDM[[1, 2, 4, 5, 3, 0], :]
    RDM = temp[:, [1, 2, 4, 5, 3, 0]]

    if mode == "":
        return (RDM,)

    elif mode == "_anim":
        animate = RDM[0:3, 0:3]
        inanimate = RDM[3:6, 3:6]
        return (animate, inanimate)

    elif mode == "_stim":
        RDM_tuple = tuple(RDM[i, i] for i in range(6))
        return RDM_tuple

if __name__ == "__main__":
    # List of models, correlation types, regions of interest, and hemispheres
    models = ["slow_r50", "slowfast_r50", "res_r50"]  # , 'slow_r50', 'dorsalnet']
    dataset = "k400"
    pretrained = True
    random_layer = ""  #'fusion/'
    isimagenet = False
    correlation_types = ["pearson"]
    ROIList = ["V1", "pFS", "LO", "EBA", "MTSTS", "infIPS", "SMG", "behavior"]
    hemispheres = ["all", "rh", "lh"]
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
    for sub in range(2, 18):
        if sub == 8:
            continue
        subject = f"S{sub:02d}"

        # Loop through regions of interest, hemispheres, models, and correlation types
        for region in ROIList:
            for hem in tqdm(
                hemispheres, desc=f"computing for subject {sub} in region {region}"
            ):
                for cor in correlation_types:
                    RDM_folder = f"result/fMRI RDM/{cor}/{region}"
                    if region == "behavior":
                        if sub == 2:
                            static_RDM, dynamic_RDM = load_behav()

                            if not os.path.exists(RDM_folder):
                                os.makedirs(RDM_folder)
                            with open(f"{RDM_folder}/RDM_dynamic.pkl", "wb") as File:
                                pickle.dump(dynamic_RDM, File)
                            with open(f"{RDM_folder}/RDM_static.pkl", "wb") as File:
                                pickle.dump(static_RDM, File)
                        else:
                            continue
                    else:
                        # Create the MRI RDM if it doesn't exist
                        if not os.path.exists(
                            f"{RDM_folder}/{subject}_RDM_{hem}_dynamic.pkl"
                        ):
                            if not os.path.exists(RDM_folder):
                                os.makedirs(RDM_folder)

                            # Construct the file path for MRI data
                            filepath = (
                                f"fMRI/{subject}/GCSS_noOverlap_{region}_{hem}.mat"
                            )
                            dynamic_tstat, static_tstat = load_MRI(filepath, hem)

                            # calculate RDM from tstat
                            dynamic_RDM = calculate_RDM(dynamic_tstat, cor)
                            static_RDM = calculate_RDM(static_tstat, cor)

                            with open(
                                f"{RDM_folder}/{subject}_RDM_{hem}_dynamic.pkl", "wb"
                            ) as File:
                                pickle.dump(dynamic_RDM, File)
                            with open(
                                f"{RDM_folder}/{subject}_RDM_{hem}_static.pkl", "wb"
                            ) as File:
                                pickle.dump(static_RDM, File)
                        else:
                            with open(
                                f"{RDM_folder}/{subject}_RDM_{hem}_dynamic.pkl", "rb"
                            ) as File:
                                dynamic_RDM = pickle.load(File)
                            with open(
                                f"{RDM_folder}/{subject}_RDM_{hem}_static.pkl", "rb"
                            ) as File:
                                static_RDM = pickle.load(File)

                    for model_name in models:
                        # Construct the save folder path
                        save_folder = f"result/RSA/{imagenet}{data}{is_cpc}{pretrained}{random_initialized}{model_name}/{cor}/{region}/{random_layer}"

                        # Create the save folder if it doesn't exist
                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder)

                        for m, (mode, name) in enumerate(names.items()):
                            dyn_RDM = filter_RDM(dynamic_RDM, mode)
                            stat_RDM = filter_RDM(static_RDM, mode)

                            model_path = f"result/model RDM/{imagenet}{data}{is_cpc}{pretrained}{random_initialized}dynamic/{random_layer}{cor}_RDM_{model_name}.pkl"
                            with open(model_path, "rb") as pickle_file:
                                model_RDM_dyn = pickle.load(pickle_file)

                            for ind, nam in enumerate(name):
                                # Generate dynamic RSA values
                                RSA = calculate_RSA_layers(
                                    model_RDM_dyn, dyn_RDM[ind], mode, ind
                                )
                                with open(
                                    f"{save_folder}{subject}_{hem}_dynamic_RSA{nam}.pkl",
                                    "wb",
                                ) as File:
                                    pickle.dump(RSA, File)