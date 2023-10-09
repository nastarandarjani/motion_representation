# This script loads RDM (Representational Dissimilarity Matrix) data from a pickle file,
# applies Cumulative Distribution Function (CDF) normalization to each layer's data,
# creates visualization plots, and saves them in a PDF document.

import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

# Function to compute the Cumulative Distribution Function (CDF) of an array
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

if __name__ == "__main__":
    # Specify the desired model name here ('slowfast_r50' or 'x3d_s')
    model_name = 'slowfast_r50'
    # Specify the correlation types in RDM ('pearson', 'spearman' or 'euclidean')
    cor_type = 'euclidean'

    video_names = ['ball', 'human', 'mammal', 'penswi', 'reptile', 'tool']

    # Construct the path to the pickle file containing correlation data
    folder_path = f'../result/{cor_type}_RDM_{model_name}.pkl'

    # Load correlation data from a pickle file
    with open(folder_path, 'rb') as pickle_file:
        correlation = pickle.load(pickle_file)

    # Directory to save the PDF file
    save_pdf_path = f'/{model_name}_{cor_type}.pdf'

    # Create a PDF file to save the plots
    pdf_pages = PdfPages(save_pdf_path)

    # Iterate through layers and plot CDFs
    for layer_name, rdm in tqdm(correlation.items()):
        # Create a CDF plot
        plt.imshow(CDF_(rdm))
        plt.xticks(range(36), video_names, rotation=90)
        plt.yticks(range(36), video_names)
        plt.title(layer_name)

        # Add a colorbar to the plot
        plt.colorbar()

        # Add the current plot to the PDF
        pdf_pages.savefig()

        # Close the plot to release resources
        plt.close()

    # Close the PDF file
    pdf_pages.close()
