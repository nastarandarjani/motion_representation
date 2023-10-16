# plot RDM

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import SubplotSpec
import matplotlib.pyplot as plt
import pickle
import tqdm

def CDF_(x):
    """
    Calculate the Cumulative Distribution Function (CDF) of an input array.

    Parameters:
    x (numpy.ndarray): Input array for which the CDF is calculated.

    Returns:
    numpy.ndarray: CDF values corresponding to the input array.

    This function reshapes the input array if needed and computes the CDF values
    for the elements in the array.
    """
    # Reshape input array x to ensure it has 2 dimensions
    shape = x.shape
    x = x.reshape(1, -1)
    n = len(x)

    # Compute CDF values
    sorted_unique, index = np.unique(x, return_inverse=True)
    value = (1 + np.arange(len(sorted_unique))) / len(sorted_unique)
    data = value[index].reshape(shape)
    return data

# Define models, correlation types, regions of interest, and condition
models = ['x3d_m', 'slowfast_r50']
cor_types = ['pearson', 'spearman', 'euclidean']
ROIList = ['V1', 'pFS', 'LO', 'EBA', 'MTSTS', 'infIPS', 'SMG']

# Specify the condition ('slow', 'fast', 'rmslow', 'fusion', or '')
condition = ['slow', 'fast', 'fusion', 'rmslow', '']

# Specify the path where results will be saved
folder = '/result'

# Define a list of names
name = ['human', 'mammal', 'reptile', 'tool', 'penswi', 'ball']

# Loop through different models
for model_name in models:
    for cond in condition:
        # Set the condition based on the model
        if model_name != 'slowfast_r50' and cond != '':
            continue

        # Load the maximum layer data for the current model and condition
        max_layer = np.load(f'{folder}/max layer/layer_{model_name}_{cond}.npz', allow_pickle=True)['arr_0']

        # Loop through different correlation types
        for c, cor in enumerate(cor_types):
            # Specify the path to save PDF plots
            save_pdf_path = f'/plot/RDM_{model_name}_{cor}_{cond}.pdf'
            pdf_pages = PdfPages(save_pdf_path)

            # Loop through regions of interest
            for r, region in enumerate(ROIList):
                # Create subplots for each region
                fig, axes = plt.subplots(2, 2, figsize=(10, 10))

                # Loop through dynamic and static status
                for s, status in enumerate(['dynamic', 'static']):
                    ax = axes[s, 0]
                    fMRI_RDM = []

                    # Loop through subjects
                    for sub in range(2, 18):
                        if sub == 8:
                            continue
                        subject = f'S{sub:02d}'
                        RDM_folder = f'/result/fMRI RDM/{cor}/{region}'

                        # Load fMRI RDM data from file
                        with open(f'{RDM_folder}/{subject}_RDM_all_{status}.pkl', 'rb') as File:
                            fMRI_RDM.append(pickle.load(File))

                    fMRI_RDM = np.mean(fMRI_RDM, axis=0)

                    temp = fMRI_RDM[[1, 2, 4, 5, 3, 0], :]
                    fMRI_RDM = temp[:, [1, 2, 4, 5, 3, 0]]

                    # plot average fMRI RDM
                    im = ax.imshow(CDF_(fMRI_RDM))
                    ax.set_xticks(range(6), name, rotation=90)
                    ax.set_yticks(range(6), name)
                    ax.set_title('fmri ' + status)
                    plt.colorbar(im, ax=ax)

                    ax = axes[s, 1]
                    model_folder = f'/result/model RDM'

                    # Load model RDM data from file
                    with open(f'{model_folder}/{cor}_RDM_{model_name}.pkl', 'rb') as File:
                        model_RDM = pickle.load(File)

                    for layer_name, RDM in model_RDM.items():
                        if layer_name == max_layer[c, r, s]:
                            layer_RDM = RDM

                    im = ax.imshow(CDF_(layer_RDM))
                    ax.set_xticks(range(6), name, rotation=90)
                    ax.set_yticks(range(6), name)
                    ax.set_title(f'model {status}\n{max_layer[c, r, s]}')
                    plt.colorbar(im, ax=ax)

                fig.suptitle(region)
                plt.tight_layout()

                # Save the PDF page
                pdf_pages.savefig()
                plt.close()

            # Close the PDF file
            pdf_pages.close()
