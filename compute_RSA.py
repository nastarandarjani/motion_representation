import numpy as np
from scipy.stats import kendalltau
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
    - dynamic_RDM (numpy.ndarray): Dynamic RDM for the specified hemisphere.
    - static_RDM (numpy.ndarray): Static RDM for the specified hemisphere.
    """
    # Load data from MAT file using the specified hemisphere identifier
    if hemisphere == 'all':
        hemisphere = hemisphere.capitalize()
    else:
        hemisphere = hemisphere.upper()
    key = f'condData{hemisphere}'
    data = scipy.io.loadmat(filepath)[key]

    # Separate dynamic and static RDM
    temp = data[:6, :, :]
    dynamic_RDM = temp[:, :6, :]
    temp = data[6:, :, :]
    static_RDM = temp[:, 6:, :]

    # swap rows to the desired form
    temp = dynamic_RDM[[5, 0, 1, 4, 2, 3], :, :]
    dynamic_RDM = temp[:, [5, 0, 1, 4, 2, 3], :]

    temp = static_RDM[[5, 0, 1, 4, 2, 3], :, :]
    static_RDM = temp[:, [5, 0, 1, 4, 2, 3], :]

    return dynamic_RDM, static_RDM

def calculate_RSA(RDM1, RDM2):
    """
    Calculate Representational Similarity Analysis (RSA) for
    two Representational Dissimilarity Matrices (RDMs).

    Parameters:
    - RDM1 (numpy.ndarray): First RDM.
    - RDM2 (numpy.ndarray): Second RDM with timepoints.

    Returns:
    - correlations (list): List of RSA values for each time point.
    """
    # Calculate RSA for each time point in RDM2
    R1 = RDM1.reshape(1, -1)
    R2 = RDM2.reshape(6*6, -1).swapaxes(0, 1)

    vectorized_kendall_tau = np.vectorize(kendalltau, signature='(m),(m)->(),()')

    # Apply vectorized function to calculate Kendall Tau correlations for each time point
    correlations = vectorized_kendall_tau(R1, R2)[0]

    return correlations

def calculate_RSA_layers(RDM1, RDM2):
    """
    Calculate RSA values for a dictionary of RDMs with different layers.

    Parameters:
    - RDM1 (dict): Dictionary of RDMs.
    - RDM2 (numpy.ndarray): Second RDM with timepoints.

    Returns:
    - RSA (dict): Dictionary of RSA values for each layer.
    """
    RSA = {}
    for layer_name, RDM in RDM1.items():
        RSA[layer_name] = calculate_RSA(RDM, RDM2)
    return RSA

# List of models
models = ['slowfast_r50', 'x3d_s']
# List of correlation types in RDM
correlation_types = ['pearson', 'spearman']

# List of regions of interest
ROIList = ['V1', 'pFS', 'LO', 'EBA', 'MTSTS', 'infIPS', 'SMG']
# List of hemispheres
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
            filepath = f'/fMRI/{subject}/GCSS_noOverlap_{region}_{hem}.mat'
            dynamic_RDM, static_RDM = load_MRI(filepath, hem)

            for model_name in models:
                for cor in correlation_types:
                    # Construct the file path for model RDM
                    model_path = f'/result/{cor}_RDM_{model_name}.pkl'

                    with open(model_path, 'rb') as pickle_file:
                        model_RDM = pickle.load(pickle_file)

                    # Construct the save folder path
                    save_folder = f'/result/{subject}/{model_name}/{cor}'

                    # Create the save folder if it doesn't exist
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    # Generate dynamic RSA values
                    RSA = calculate_RSA_layers(model_RDM, dynamic_RDM)

                    # Save dynamic RSA dictionary to a pickle file
                    with open(f'{save_folder}/{region}_{hem}_dynamic_RSA.pkl', 'wb') as File:
                        pickle.dump(RSA, File)

                    # Generate static RSA values
                    RSA = calculate_RSA_layers(model_RDM, static_RDM)

                    # Save static RSA dictionary to a pickle file
                    with open(f'{save_folder}/{region}_{hem}_static_RSA.pkl', 'wb') as File:
                        pickle.dump(RSA, File)
