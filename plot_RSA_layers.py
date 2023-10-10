import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

model_name = 'slowfast_r50' # 'x3d_m', 'slowfast_r50'

# Define correlation types, regions of interest, and condition
cor_types = ['pearson', 'spearman', 'euclidean']
ROIList = ['V1', 'pFS', 'LO', 'EBA', 'MTSTS', 'infIPS', 'SMG']
condition = ['slow', 'fast', 'fusion', 'rmslow', '']

# Initialize an empty array for storing max layers
max_layer = np.empty((len(cor_types), len(ROIList), 2), dtype=object)

for cond in condition:
    if model_name != 'slowfast_r50' and cond != '':
        continue
    for c, cor in enumerate(cor_types):
        # Create subplots for plotting
        fig, axes = plt.subplots(2, len(ROIList), figsize=(56, 10), sharey=True)

        # Loop through regions of interest
        for r, region in enumerate(ROIList):
            # Loop through dynamic and static status
            for s, status in enumerate(['dynamic', 'static']):
                ax = axes[s, r]
                data = []

                # Loop through subjects
                for sub in range(2, 18):
                    if sub == 8:
                        continue
                    subject = f'S{sub:02d}'

                    folder = f'/result/RSA/{model_name}/{cor}/{region}'
                    # Load RSA data from file
                    with open(f'{folder}/{subject}_all_{status}_RSA.pkl', 'rb') as File:
                        RSA = pickle.load(File)

                    data.append(list(RSA.values()))
                
                data = np.array(data)
                filtered_list = list(RSA.keys())

                if model_name == 'slowfast_r50':
                    # Filter RSA data based on the specified condition
                    if cond == 'slow':
                        filtered_list = [key for key in filtered_list if 'multipathway_blocks.1' in key]
                    elif cond == 'fast':
                        filtered_list = [key for key in filtered_list if 'multipathway_blocks.0' in key]
                    elif cond == 'fusion':
                        filtered_list = [key for key in filtered_list if 'multipathway_fusion' in key]
                    elif cond == 'rmslow':
                        filtered_list = [key for key in filtered_list if 'multipathway_blocks.1' not in key]

                    # Get the indices of selected keys
                    indices = [index for index, key in enumerate(RSA.keys()) if key in filtered_list]
                    data = data[:, indices]

                SEM = np.std(data, axis=0) / np.sqrt(len(data))
                # average across subjects
                data = np.mean(data, axis=0)

                upper_bound = data + SEM
                lower_bound = data - SEM

                ax.plot(data, color='blue')
                ax.fill_between(np.array(range(len(data))), lower_bound, upper_bound, color='blue', alpha=0.5)

                max_key = filtered_list[np.nanargmax(data)]
                max_layer[c, r, s] = max_key

                ax.set_title(f'{status} {region}\n{max_key}')

        fig.supylabel('Correlation (Kendal Tau)')
        fig.supxlabel('Layer')
        plt.tight_layout()
        plt.savefig(f'/plot/{model_name}_{cor}_{cond}.png')
        plt.close()

        # Save the max layers data
        save_folder = f'/result/max layer'
        np.savez(f'{save_folder}/layer_{model_name}_{cor}_{cond}.npz', max_layer)
