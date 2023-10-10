# This script processes MRI data and performs RSA (Representational Similarity Analysis) for multiple subjects,
# regions of interest, hemispheres, models, and correlation types. It loads MRI data, calculates RDMs for dynamic
# and static t-statistics, and computes RSA values between model RDMs and MRI RDMs.

import numpy as np
from scipy.stats import kendalltau
from scipy.stats import spearmanr
import scipy.io
from tqdm import tqdm
import os
import pickle

def load_MRI(filepath, hemisphere):
    """
    Load MRI data from a MAT file for a specific hemisphere.

    Parameters:
    - filepath (str): Path to the MAT file containing MRI data.
    - hemisphere (str): Hemisphere identifier
            ('rh' for right, 'lh' for left hemisphere, or 'all' for both).

    Returns:
    - dynamic_tstat (numpy.ndarray): Dynamic tstats for the specified hemisphere.
    - static_tstat (numpy.ndarray): Static tstats for the specified hemisphere.
    """
    # Load data from MAT file using the specified hemisphere identifier
    if hemisphere == 'all':
        hemisphere = hemisphere.capitalize()
    else:
        hemisphere = hemisphere.upper()
    key = f'condData{hemisphere}'
    data = scipy.io.loadmat(filepath)[key]

    # Separate dynamic and static RDM
    dynamic_tstat = np.mean(data[:6, :, :], axis = 1)
    static_tstat = np.mean(data[6:, :, :], axis = 1)

    # swap rows to the desired form
    dynamic_tstat = dynamic_tstat[[5, 0, 1, 4, 2, 3], :]
    static_tstat = static_tstat[[5, 0, 1, 4, 2, 3], :]

    return dynamic_tstat, static_tstat

def calculate_RDM(response_patterns, method='euclidean'):
    """
    Calculate Representational Dissimilarity Matrix (RDM).

    Parameters:
    - response_patterns (numpy.ndarray): Response patterns.
    - method (str): Method for RDM calculation ('euclidean', 'spearman', 'pearson').

    Returns:
    - rdm (numpy.ndarray): RDM based on the specified method.
    """

    if method == 'euclidean':
        # Compute RDM using vectorized operations
        pairwise_differences = response_patterns[:, np.newaxis, :] - response_patterns[np.newaxis, :, :]
        rdm = np.linalg.norm(pairwise_differences, axis = 2)
    elif method == 'spearman':
        cor, _ = spearmanr(response_patterns, axis = 1)
        rdm = 1 - cor
    elif method == 'pearson':
        rdm = 1 - np.corrcoef(response_patterns)

    return rdm

def calculate_RSA(RDM1, RDM2):
    """
    Calculate Representational Similarity Analysis (RSA) using Kendall's Tau.

    Parameters:
    - RDM1 (numpy.ndarray): First RDM.
    - RDM2 (numpy.ndarray): Second RDM.

    Returns:
    - correlation (float): Kendall's Tau correlation between the two RDMs.
    """
    # Calculate Kendall's Tau correlation between the two RDMs
    correlation, _ = kendalltau(RDM1.flatten(), RDM2.flatten())
    return correlation

def calculate_RSA_layers(RDM1, RDM2):
    """
    Calculate RSA values for a dictionary of RDMs with different layers.

    Parameters:
    - RDM1 (dict): Dictionary of RDMs.
    - RDM2 (numpy.ndarray): Second RDM from MRI recording.

    Returns:
    - RSA (dict): Dictionary of RSA values for each layer.
    """
    RSA = {}
    for layer_name, RDM in RDM1.items():
        RSA[layer_name] = calculate_RSA(RDM, RDM2)
    return RSA

# List of models, correlation types, regions of interest, and hemispheres
models = ['slowfast_r50', 'x3d_m']
correlation_types = ['pearson', 'spearman', 'euclidean']
ROIList = ['V1', 'pFS', 'LO', 'EBA', 'MTSTS', 'infIPS', 'SMG']
hemispheres = ['all', 'rh', 'lh']

# Loop through subjects
for sub in range(2, 18):
    if sub == 8:
        continue
    subject = f'S{sub:02d}'

    # Loop through regions of interest, hemispheres, models, and correlation types
    for region in ROIList:
        for hem in tqdm(hemispheres, desc=f'computing for subject {sub} in region {region}'):
            # Construct the file path for MRI data
            filepath = f'/content/drive/My Drive/motion_representation/fMRI/{subject}/GCSS_noOverlap_{region}_{hem}.mat'
            dynamic_tstat, static_tstat = load_MRI(filepath, hem)

            for model_name in models:
                for cor in correlation_types:
                    # Construct the file path for model RDM
                    model_path = f'/result/model RDM/{cor}_RDM_{model_name}.pkl'

                    with open(model_path, 'rb') as pickle_file:
                        model_RDM = pickle.load(pickle_file)

                    # calculate RDM from tstat
                    dynamic_RDM = calculate_RDM(dynamic_tstat, cor)
                    static_RDM = calculate_RDM(static_tstat, cor)

                    # save RDM files
                    RDM_folder = f'/result/fMRI RDM/{subject}_{cor}_RDM_{region}'
                    with open(f'{RDM_folder}_dynamic.pkl', 'wb') as File:
                        pickle.dump(dynamic_RDM, File)
                    with open(f'{RDM_folder}_static.pkl', 'wb') as File:
                        pickle.dump(static_RDM, File)

                    # Generate dynamic RSA values
                    RSA = calculate_RSA_layers(model_RDM, dynamic_RDM)

                    # Construct the save folder path
                    save_folder = f'/result/RSA/{model_name}/{cor}/{region}'

                    # Create the save folder if it doesn't exist
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    # Save dynamic RSA dictionary to a pickle file
                    with open(f'{save_folder}/{subject}_{hem}_dynamic_RSA.pkl', 'wb') as File:
                        pickle.dump(RSA, File)

                    # Generate static RSA values
                    RSA = calculate_RSA_layers(model_RDM, static_RDM)

                    # Save static RSA dictionary to a pickle file
                    with open(f'{save_folder}/{subject}_{hem}_static_RSA.pkl', 'wb') as File:
                        pickle.dump(RSA, File)
