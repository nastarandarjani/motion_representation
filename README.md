# Motion Representation Analysis

This repository contains scripts and data for performing Representational Similarity Analysis (RSA) between fMRI data and behavioral dissimilarity data with model netwroks. The analysis includes calculating Representational Dissimilarity Matrices (RDMs) for different brain regions and comparing them with model RDMs.

## Project Structure

```bash
motion_representation/
├── DorsalNet
│   └── # Contains the DorsalNet model implementation.
├── misc
│   └── # Miscellaneous files and scripts.
├── plot
│   ├── RSA
│   ├── charades
│   ├── cpc
│   ├── fast
│   ├── fusion
│   ├── imagenet
│   ├── slow
│   ├── ssv2
│   └── untrained
│       └── # Contains plots for different models and datasets.
├── result
│   ├── RSA
│   │   ├── charades
│   │   ├── cpc
│   │   ├── dorsalnet
│   │   ├── imagenet
│   │   ├── random
│   │   ├── slow_r50
│   │   ├── slowfast_r50
│   │   ├── ssv2
│   │   └── untrained
│       └── # Contains RSA results for different models and datasets.
│   ├── fMRI RDM
│   │   ├── pearson
│   │   └── spearman
│       └── # Contains fMRI RDM results using different correlation methods.
│   └── model RDM
│       ├── charades
│       ├── cpc
│       ├── dynamic
│       ├── imagenet
│       ├── random
│       ├── ssv2
│       └── untrained
│           └── # Contains model RDM results for different models and datasets.
├── stimuli
│   └── # Contains stimuli videos and images.
├── video
│   └── # Contains sample transformed videos of each model.
├── ImagenetModel_RDM_extraction.py
├── README.md
├── compute_RSA.py
├── dissimilarity_img2.csv
├── dissimilarity_vid2.csv
├── model_RDM_extraction.py
├── plot_MDS_max.py
├── plot_RDMs_max.py
├── plot_RSA_layers.py
└── preprocess.py
```

- `ImagenetModel_RDM_extraction.py`: Script for extracting RDMs from ImageNet models.
- `compute_RSA.py`: Script for retrieving RDMs form fMRI data and behavioral dissimilarity data and performing RSA between these RDMs and model RDMs.
- `model_RDM_extraction.py`: Script for extracting RDMs from models.
- `plot_MDS_max.py`: Script for plotting maximum RSA layer's MDS.
- `plot_RDMs_max.py`: Script for plotting maximum RSA layer's RDMs.
- `plot_RSA_layers.py`: Script for plotting RSA layers.
- `preprocess.py`: Script for preprocessing data.

## Parameters

### `ImagenetModel_RDM_extraction.py`

- `model_name`: Specify the desired model name (e.g., 'AlexNet', 'ResNet50', 'DenseNet121', 'VGG16').
- `status`: Specify the status (e.g., 'dynamic', 'static').

### `model_RDM_extraction.py`

- `model_name`: Specify the desired model name (e.g., 'slowfast_r50', 'x3d_m', 'slow_r50', 'dorsalnet').
- `dataset`: Specify the dataset name (e.g., 'k400', 'ssv2', 'charades').
- `status`: Specify the status (e.g., 'dynamic', 'static').
- `pretrained`: Specify whether to use pretrained weights (True, False, 'cpc').
- `random_layer`: Specify the random layer (e.g., '', 'slow/', 'fast/', 'fusion/').


### `compute_RSA.py`

- `models`: List of model names to be used for RSA (e.g., ["slowfast_r50", "slow_r50"]).
- `dataset`: Name of the dataset (e.g., 'charades').
- `pretrained`: Flag indicating whether the models are pretrained (True, False, 'cpc').
- `random_layer`: Path to the random layer (e.g., 'fusion/').
- `isimagenet`: Flag indicating whether the dataset is ImageNet (True, False).
- `correlation_types`: List of correlation types to be used for RSA (e.g., ["pearson"]).
- `ROIList`: List of regions of interest (e.g., ["V1", "pFS", "LO", "EBA", "MTSTS", "infIPS", "SMG", "behavior"]).
- `hemispheres`: List of hemispheres to be analyzed (e.g., ["all"]).

## Usage

### Extracting RDMs from Models

To extract RDMs from models, run the `model_RDM_extraction.py` script:
```sh
python model_RDM_extraction.py
```

### Extracting RDMs from ImageNet Models

To extract RDMs from ImageNet models, run the `ImagenetModel_RDM_extraction.py` script:
```sh
python ImagenetModel_RDM_extraction.py
```

### Performing RSA

To perform RSA on fMRI data and behavioral dissimilarity data, run the `compute_RSA.py` script:
```sh
python compute_RSA.py
```