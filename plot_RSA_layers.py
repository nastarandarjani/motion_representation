import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

models = ['x3d_m', 'slowfast_r50']
cor_types = ['pearson', 'spearman', 'euclidean']
ROIList = ['V1', 'pFS', 'LO', 'EBA', 'MTSTS', 'infIPS', 'SMG']

# Specify the path 'slow', 'fast', 'rmslow', 'fusion'
cond = 'slow'

max_layer = np.empty((3, 7, 2), dtype=object)
for model_name in models:
    for c, cor in enumerate(cor_types):
        fig, axes = plt.subplots(2, 7, figsize=(56, 10), sharey=True)
        for r, region in enumerate(ROIList):
            for s, status in enumerate(['dynamic', 'static']):
                ax = axes[s, r]
                data = []
                for sub in range(2, 18):
                    if sub == 8:
                        continue
                    subject = f'S{sub:02d}'
                    save_folder = f'/result/{model_name}/{cor}'

                    with open(f'{save_folder}/{region}/{subject}_all_{status}_RSA.pkl', 'rb') as File:
                        RSA = pickle.load(File)

                    data.append(list(RSA.values()))

                filtered_dict = RSA.items()
                if model_name == 'slowfast_r50':
                    condition = cond
                    if cond == 'slow':
                        # slow path: 'multipathway_block.1'
                        filtered_dict = {key: value for key, value in RSA.items() if 'multipathway_block.1' in key}
                    elif cond == 'fast':
                        # slow path: 'multipathway_block.0'
                        filtered_dict = {key: value for key, value in RSA.items() if 'multipathway_block.0' in key}
                    elif cond == 'fusion':
                        filtered_dict = {key: value for key, value in RSA.items() if 'multipathway_fusion' in key}
                    elif cond == 'rmslow':
                        filtered_dict = {key: value for key, value in RSA.items() if 'multipathway_block.1' not in key}
                else:
                    condition = ''
                
                indices = [index for index, key in enumerate(RSA) if key in filtered_dict]

                data = np.array(data)
                data = data[:, indices]

                SEM = np.std(data, axis = 0) / np.sqrt(len(data))
                data = np.mean(data, axis = 0)

                upper_bound = data + SEM
                lower_bound = data - SEM

                ax.plot(data, color='blue')
                ax.fill_between(np.array(range(len(data))), lower_bound, upper_bound, color='blue', alpha=0.5)

                max_key = list(RSA)[np.nanargmax(data)]
                max_layer[c, r, s] = max_key

                ax.set_title(f'{status} {region}\n{max_key}')                

        fig.supylabel('Correlation (Kendal Tau)')
        fig.supxlabel('Layer')
        plt.tight_layout()
        plt.savefig(f'/result/plot/{model_name}_{cor}.png')
        plt.close()

    save_folder = f'/result'
    np.savez(f'{save_folder}/max_layer_{model_name}_{condition}.npz', max_layer)
