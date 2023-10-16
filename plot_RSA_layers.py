import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

model_name = 'x3d_m' # 'x3d_m', 'slowfast_r50'

# Define correlation types, regions of interest, condition and status
cor_types = ['pearson', 'spearman', 'euclidean']
ROIList = ['V1', 'pFS', 'LO', 'EBA', 'MTSTS', 'infIPS', 'SMG']
condition = ['slow', 'fast', 'fusion', 'rmslow', '']
stat = ['dynamic', 'static']

# Initialize an empty array for storing max layers
max_layer = np.empty((len(cor_types), len(ROIList), 2), dtype=object)

color = ['blue', 'red']
for cond in condition:
    if model_name != 'slowfast_r50' and cond != '':
        continue
    for c, cor in enumerate(cor_types):
        # Create subplots for plotting
        fig, axes = plt.subplots(2, len(ROIList), figsize=(56, 5), sharey = 'row', sharex=True, gridspec_kw={'height_ratios': [50, 1]})

        # Loop through regions of interest
        for r, region in enumerate(ROIList):
            ttest_data = []
            ax = axes[0, r]
            # Loop through dynamic and static status
            for s, status in enumerate(stat):
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

                # save for ttest
                ttest_data.append(data)
                SEM = stats.sem(data, axis=0)
                # average across subjects
                data = np.mean(data, axis=0)

                upper_bound = data + SEM
                lower_bound = data - SEM

                ax.plot(data, color = color[s], label = status)
                ax.fill_between(np.array(range(len(data))), lower_bound, upper_bound, color=color[s], alpha=0.5, label=None)

                max_layer[c, r, s] = filtered_list[np.nanargmax(data)]

            ax.set_ylabel('Correlation (Kendal Tau)')
            ax.set_title(f'{region}', fontweight='bold')
            ax.legend()

            # perform ttest
            _, p_values = stats.ttest_ind(ttest_data[0], ttest_data[1], axis=0)
            p_values = fdrcorrection(p_values, alpha=0.05)[1]

            sig_x = np.where(p_values < 0.05)[0]

            # plot ttest
            axes[1, r].scatter(sig_x, np.zeros((len(sig_x))), c='black', marker='.', facecolors='none')
            axes[1, r].set_xticks([])
            axes[1, r].set_yticks([])
            axes[1, r].spines['top'].set_visible(False)
            axes[1, r].spines['right'].set_visible(False)
            axes[1, r].spines['bottom'].set_visible(False)
            axes[1, r].spines['left'].set_visible(False)
            # axes[1, r].set_ylabel('p-value < 0.05', rotation='horizontal', ha='right')
            axes[1, r].set_xlabel('label')

        plt.tight_layout()
        plt.savefig(f'/plot/{model_name}_{cor}_{cond}.png')
        plt.close()

    # Save the max layers data
    save_folder = f'/result/max layer'
    np.savez(f'{save_folder}/layer_{model_name}_{cond}.npz', max_layer)
