# This script processes MRI data and performs RSA (Representational Similarity Analysis) for multiple subjects,
# regions of interest, hemispheres, models, and correlation types. It loads MRI data, calculates RDMs for dynamic
# and static t-statistics, and computes RSA values between model RDMs and MRI RDMs.

# compute RSA and MRI RDM

import numpy as np
from scipy.stats import kendalltau
from scipy.stats import spearmanr
import scipy.io
from tqdm import tqdm
import os
import pickle
import math

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
    dynamic_tstat = dynamic_tstat[:, [5, 0, 1, 4, 2, 3]]

    static_tstat = static_tstat[[5, 0, 1, 4, 2, 3], :]
    static_tstat = static_tstat[:, [5, 0, 1, 4, 2, 3]]

    return dynamic_tstat, static_tstat

def load_behav():
    static_behav = np.genfromtxt ('dissimilarity_img2.csv', delimiter = ',')[1:, :]
    static_behav = static_behav.reshape(6, 6, 6, 6)
    static_behav = np.mean(static_behav, axis = 1)
    static_behav = np.mean(static_behav, axis = 2)

    dynamic_behav = np.genfromtxt ('dissimilarity_vid2.csv', delimiter = ',')[1:, :]
    dynamic_behav = dynamic_behav.reshape(6, 6, 6, 6)
    dynamic_behav = np.mean(dynamic_behav, axis = 1)
    dynamic_behav = np.mean(dynamic_behav, axis = 2)

    static_behav = static_behav[[5, 0, 1, 4, 2, 3], :]
    static_behav = static_behav[:, [5, 0, 1, 4, 2, 3]]

    dynamic_behav = dynamic_behav[[5, 0, 1, 4, 2, 3], :]
    dynamic_behav = dynamic_behav[:, [5, 0, 1, 4, 2, 3]]
    return static_behav, dynamic_behav

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
    RDM1 = RDM1[np.triu_indices(RDM1.shape[0], k = 1)]
    RDM2 = RDM2[np.triu_indices(RDM2.shape[0], k = 1)]
    # Calculate Kendall's Tau correlation between the two RDMs
    correlation, _ = kendalltau(RDM1, RDM2)
    return correlation

def calculate_RSA_layers(RDM1, RDM2, mode, ind):
    RSA = {}
    for layer_name, RDM in RDM1.items():
        RDM = filter_RDM(RDM, mode)[ind]
        RSA[layer_name] = calculate_RSA(RDM, RDM2)
    return RSA

def calculate_semcor_layers(RDM1, RDM2, RDM3, mode, ind):
    r23 = calculate_RSA(RDM2, RDM3)
    cor = {}
    for layer_name, RDM in RDM1.items():
        RDM = filter_RDM(RDM, mode)[ind]
        r12 = calculate_RSA(RDM, RDM2)
        r13 = calculate_RSA(RDM, RDM3)

        cor[layer_name] = (r12 - (r13 * r23)) / (math.sqrt(1 - math.pow(r13, 2)) * math.sqrt(1 - math.pow(r23, 2)))

    return cor

def filter_RDM(RDM, mode):
    temp = RDM[[1, 2, 4, 5, 3, 0], :]
    RDM = temp[:, [1, 2, 4, 5, 3, 0]]

    if mode == '':
        return (RDM, )

    elif mode == '_anim':
        animate = RDM[0:3, 0:3]
        inanimate = RDM[3:6, 3:6]
        return (animate, inanimate)

    elif mode == '_stim':
        RDM_tuple = tuple(RDM[i, i] for i in range(6))
        return RDM_tuple


# List of models, correlation types, regions of interest, and hemispheres
models = ['x3d_m', 'slowfast_r50', 'slow_r50', 'dorsalnet']
correlation_types = ['pearson', 'spearman', 'euclidean']
ROIList = ['V1', 'pFS', 'LO', 'EBA', 'MTSTS', 'infIPS', 'SMG', 'behavior']
hemispheres = ['all', 'rh', 'lh']
names = {'_anim' : ['_animate', '_inanimate'], '': ['']}
israndom = False
isimagenet = True


if isimagenet:
    models = ['alexnet', 'resnet50', 'densenet121', 'vgg16']
    israndom = False

random_initialized = 'random/' if israndom else ''
imagenet = 'imagenet/' if isimagenet else ''
# Loop through subjects
for sub in range(2, 18):
    if sub == 8:
        continue
    subject = f'S{sub:02d}'

    # Loop through regions of interest, hemispheres, models, and correlation types
    for region in ROIList:
        for hem in tqdm(hemispheres, desc=f'computing for subject {sub} in region {region}'):
            for cor in correlation_types:
                RDM_folder = f'result/fMRI RDM/{cor}/{region}'
                if region == 'behavior':
                    if sub == 2:
                        static_RDM, dynamic_RDM = load_behav()

                        if not os.path.exists(RDM_folder):
                            os.makedirs(RDM_folder)
                        with open(f'{RDM_folder}/RDM_dynamic.pkl', 'wb') as File:
                            pickle.dump(dynamic_RDM, File)
                        with open(f'{RDM_folder}/RDM_static.pkl', 'wb') as File:
                            pickle.dump(static_RDM, File)
                    else:
                        continue
                else:
                    # Create the MRI RDM if it doesn't exist
                    if not os.path.exists(f'{RDM_folder}/{subject}_RDM_{hem}_dynamic.pkl'):
                        if not os.path.exists(RDM_folder):
                            os.makedirs(RDM_folder)

                        # Construct the file path for MRI data
                        filepath = f'fMRI/{subject}/GCSS_noOverlap_{region}_{hem}.mat'
                        dynamic_tstat, static_tstat = load_MRI(filepath, hem)

                        # calculate RDM from tstat
                        dynamic_RDM = calculate_RDM(dynamic_tstat, cor)
                        static_RDM = calculate_RDM(static_tstat, cor)

                        with open(f'{RDM_folder}/{subject}_RDM_{hem}_dynamic.pkl', 'wb') as File:
                            pickle.dump(dynamic_RDM, File)
                        with open(f'{RDM_folder}/{subject}_RDM_{hem}_static.pkl', 'wb') as File:
                            pickle.dump(static_RDM, File)
                    else:
                        with open(f'{RDM_folder}/{subject}_RDM_{hem}_dynamic.pkl', 'rb') as File:
                            dynamic_RDM = pickle.load(File)
                        with open(f'{RDM_folder}/{subject}_RDM_{hem}_static.pkl', 'rb') as File:
                            static_RDM = pickle.load(File)

                for model_name in models:
                    # Construct the save folder path
                    save_folder = f'result/RSA/{imagenet}{random_initialized}{model_name}/{cor}/{region}'

                    # Create the save folder if it doesn't exist
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    for m, (mode, name) in enumerate(names.items()):
                        dyn_RDM = filter_RDM(dynamic_RDM, mode)
                        stat_RDM = filter_RDM(static_RDM, mode)

                        model_path = f'result/model RDM/{imagenet}{random_initialized}dynamic/{cor}_RDM_{model_name}.pkl'
                        with open(model_path, 'rb') as pickle_file:
                            model_RDM_dyn = pickle.load(pickle_file)

                        model_path = f'result/model RDM/{imagenet}{random_initialized}static/{cor}_RDM_{model_name}.pkl'
                        with open(model_path, 'rb') as pickle_file:
                            model_RDM_stat = pickle.load(pickle_file)

                        for ind, nam in enumerate(name):
                            # Generate dynamic RSA values
                            RSA = calculate_RSA_layers(model_RDM_dyn, dyn_RDM[ind], mode, ind)
                            with open(f'{save_folder}/{subject}_{hem}_dynamic_RSA{nam}.pkl', 'wb') as File:
                                pickle.dump(RSA, File)

                            # Generate static RSA values
                            RSA = calculate_RSA_layers(model_RDM_stat, stat_RDM[ind], mode, ind)
                            with open(f'{save_folder}/{subject}_{hem}_static_RSA{nam}.pkl', 'wb') as File:
                                pickle.dump(RSA, File)

                            if mode == '':
                                semcor = calculate_semcor_layers(model_RDM_dyn, dyn_RDM[ind], stat_RDM[ind], mode, ind)
                                with open(f'{save_folder}/{subject}_{hem}_dynamic_cor{nam}.pkl', 'wb') as File:
                                    pickle.dump(semcor, File)

                                # Generate semi-partial correlation
                                semcor = calculate_semcor_layers(model_RDM_stat, stat_RDM[ind], dyn_RDM[ind], mode, ind)
                                with open(f'{save_folder}/{subject}_{hem}_static_cor{nam}.pkl', 'wb') as File:
                                    pickle.dump(semcor, File)
