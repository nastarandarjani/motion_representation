import numpy as np
import torch
import json
import urllib
import pickle
from tqdm import tqdm
import torch.nn as nn
from scipy.stats import spearmanr
from pytorchvideo.data.encoded_video import EncodedVideo
import os
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

def load_model(model_name):
    weights = models.__dict__[f'{model_name}_Weights'].DEFAULT
    model = models.__dict__[model_name.lower()](weights=weights)
    model.eval()
    model = model.to('cuda')
    return model, weights

def get_relu_modules(model):
    """
    Get the names of ReLU modules within a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model from which to extract module names.

    Returns:
        List[str]: A list of module names that are instances of nn.ReLU.
                   These names can be used to access or manipulate these modules in the model.
    """
    modules = []

    # Iterate through all modules in the model
    for module_name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            modules.append(module_name)

    return modules

def get_activation(model, video_inputs, layer, isLabel = False):
    """
    Get the activation from a specified layer of a pre-trained model.

    Args:
        model (torch.nn.Module): Pre-trained model.
        video_inputs (torch.Tensor): Video input tensor.
        layer (torch.nn.Module): The layer from which to extract activation.

    Returns:
        numpy.ndarray: Activation values as a NumPy array.
    """
    global model_name, weights
    def hook_func(model, input, output):
        nonlocal Layer_output
        Layer_output = output

    # Register the forward hook on the desired layer
    hook = layer.register_forward_hook(hook_func)

    Layer_output = None
    preds = model(video_inputs).softmax(1)
    hook.remove()

    if isLabel:
        class_id = preds.argmax(axis = 1)
        labels = weights.meta["categories"]
        print([labels[i] for i in class_id])

    activations_batch = Layer_output.detach().cpu().numpy().reshape(len(Layer_output), -1)
    return activations_batch


if __name__ == "__main__":
    # Specify the desired model name ('AlexNet', 'ResNet50', 'DenseNet121', 'VGG16')
    model_name = 'DenseNet121'
    status = 'static' # 'dynamic', 'static'

    # List the files in the folder with the proper prefix
    prefix = 'processed_' if status == 'dynamic' else 'img_'
    processed_videos = [file for file in sorted(os.listdir('stimuli')) if file.startswith(prefix)]

    # Load the pre-trained model
    model, weights = load_model(model_name)

    # Retrieve corresponding layers from which to extract activations
    modules = get_relu_modules(model)

    transformed_videos = []
    preprocess = weights.transforms()
    for video_file in tqdm(processed_videos, desc='load data'):
        video_path = os.path.join('stimuli', video_file)
        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec=0, end_sec=3)['video']
        if status == 'static':
            video_data = video_data[:, 1, :, :]
            video_data = torch.unsqueeze(video_data, 1)
        video_data = video_data.permute(1, 0, 2, 3) / 255.0
        video_data = preprocess(video_data)
        transformed_videos.append(video_data)

    euclidean_RDM = {}
    pearson_RDM = {}
    spearman_RDM = {}
    for model_layer in tqdm(modules):
        layer = model
        for attr in model_layer.split('.'):
            layer = getattr(layer, attr)

        activations = []
        # Process videos in batches
        for transformed_video in transformed_videos:
            with torch.no_grad():
                activation = get_activation(model, transformed_video.to('cuda'), layer, isLabel=False)
            if status == 'dynamic':
                activation = np.mean(activation, axis = 0, keepdims = True)
            activations.append(activation)

        activations = np.vstack(activations)

        # average across category
        activations = activations.reshape(6, 6, -1)
        activations = np.mean(activations, axis = 1)

        pearson_RDM[model_layer] = 1 - np.corrcoef(activations)

        cor, _ = spearmanr(activations, axis=1)
        spearman_RDM[model_layer] = 1 - cor

        pairwise_differences = activations[:, np.newaxis, :] - activations[np.newaxis, :, :]
        euclidean_RDM[model_layer] = np.linalg.norm(pairwise_differences, axis=2)

    model_name = model_name.lower()
    # Save the RDM dictionary to a pickle file
    file_path = f'result/model RDM/imagenet/{status}/pearson_RDM_{model_name}.pkl'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as File:
        pickle.dump(pearson_RDM, File)

    file_path = f'result/model RDM/imagenet/{status}/spearman_RDM_{model_name}.pkl'
    with open(file_path, 'wb') as File:
        pickle.dump(spearman_RDM, File)

    file_path = f'result/model RDM/imagenet/{status}/euclidean_RDM_{model_name}.pkl'
    with open(file_path, 'wb') as File:
        pickle.dump(euclidean_RDM, File)
