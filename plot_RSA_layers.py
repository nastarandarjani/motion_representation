import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

models = ['x3d_s', 'slowfast_r50']
cor_types = ['pearson', 'spearman', 'euclidean']
ROIList = ['V1', 'pFS', 'LO', 'EBA', 'MTSTS', 'infIPS', 'SMG']

max_layer = np.empty((2, 3, 7, 2), dtype=object)
for m, model_name in enumerate(models):
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

                # Remove 'branch2' (slow path), 'multipathway_fusion'
                # filtered_dict = {key: value for key, value in RSA.items() if 'branch2' not in key}
                # indices = [index for index, key in enumerate(RSA) if key in filtered_dict]

                data = np.array(data)
                # data = data[:, indices]

                SEM = np.std(data, axis = 0) / np.sqrt(len(data))
                data = np.mean(data, axis = 0)

                upper_bound = data + SEM
                lower_bound = data - SEM

                ax.plot(data, color='blue')
                ax.fill_between(np.array(range(len(data))), lower_bound, upper_bound, color='blue', alpha=0.5)

                max_key = list(RSA)[np.nanargmax(data)]
                max_layer[m, c, r, s] = max_key

                ax.set_title(f'{status} {region}\n{max_key}')                

        fig.supylabel('Correlation (Kendal Tau)')
        fig.supxlabel('Layer')
        plt.tight_layout()
        plt.savefig(f'/result/plot/{model_name}_{cor}.png')
        plt.close()

save_folder = f'/result'
np.savez(f'{save_folder}/max_layer.npz', max_layer)
