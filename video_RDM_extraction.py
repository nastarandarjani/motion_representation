# This script loads a pre-trained PyTorchVideo model, processes video clips, extracts
# activations from all BatchNormalization and Relu layer, and saves the RDM in a pickle file.

# transformations are based on :
#    slowfast: https://pytorch.org/hub/facebookresearch_pytorchvideo_slowfast/
#    x3d: https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/

# Necessary installation
# pip install av
# pip install pytorchvideo
# pip install tqdm

import numpy as np
import torch
import av
import json
import urllib
import pickle
from tqdm import tqdm
import torch.nn as nn
from scipy.stats import spearmanr
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
import os

# Define functions

def load_pretrained_model(model_name):
    """
    Load a pre-trained PyTorchVideo model.

    Args:
        model_name (str): Name of the model to load.

    Returns:
        torch.nn.Module: Loaded pre-trained model.
    """
    model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
    model = model.eval()
    model = model.to('cuda')
    return model

def apply_video_transform(model_name, video_data):
    """
    Apply transformations to video data according to model.

    Args:
        model_name (str): Pre-trained model's name.
        video_data (dict): Video data dictionary.

    Returns:
        dict: Transformed video data.
    """

    if model_name == 'slowfast_r50':
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 32
        sampling_rate = 2
        slowfast_alpha = 4
        num_clips = 10
        num_crops = 3

        class PackPathway(torch.nn.Module):
            """
            Transform for converting video frames as a list of tensors.
            """
            def __init__(self):
                super().__init__()

            def forward(self, frames: torch.Tensor):
                fast_pathway = frames
                # Perform temporal sampling from the fast pathway.
                slow_pathway = torch.index_select(
                    frames,
                    1,
                    torch.linspace(
                        0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
                    ).long(),
                )
                frame_list = [slow_pathway, fast_pathway]
                return frame_list

        transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=side_size
                    ),
                    CenterCropVideo(crop_size),
                    PackPathway()
                ]
            ),
        )
    elif model_name ==  'x3d_s':
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        frames_per_second = 30
        side_size = 182
        crop_size = 182
        num_frames = 13
        sampling_rate = 6

        transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size=side_size),
                    CenterCropVideo(crop_size)
                ]
            ),
        )

    transformed_video = transform(video_data)
    return transformed_video

def process_video(model_name, video_path):
    """
    Read, extract and transform a clip from a video file.

    Args:
        model_name (str): Pre-trained model's name.
        video_path (str): Path to the video file.

    Returns:
        transformed_video_data (torch.Tensor): transformed video tensor.
    """
    # Read video from a file
    video = EncodedVideo.from_path(video_path)
    video_data = video.get_clip(start_sec=0, end_sec=9)

    # Apply video transformations
    transformed_video_data = apply_video_transform(model_name, video_data)["video"]

    return transformed_video_data

def get_top_k_predicted_labels(preds, k=5):
    """
    Get the top-k predicted labels from a set of model predictions.

    Args:
        preds (torch.Tensor): Model predictions as a tensor.
        k (int): Number of top labels to retrieve.

    Returns:
        list: Top-k predicted labels.
    """
    def get_labels_name():
        """
        Retrieve the label names from the Kinetics dataset.

        Returns:
            dict: A dictionary mapping label IDs to label names.
        """
        json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
        json_filename = "kinetics_classnames.json"
        try: urllib.URLopener().retrieve(json_url, json_filename)
        except: urllib.request.urlretrieve(json_url, json_filename)

        with open(json_filename, "r") as f:
            kinetics_classnames = json.load(f)

        # Create an id to label name mapping
        kinetics_id_to_classname = {}
        for k, v in kinetics_classnames.items():
            kinetics_id_to_classname[v] = str(k).replace('"', "")

        return kinetics_id_to_classname

    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=k).indices[0]

    kinetics_classnames = get_labels_name()

    pred_class_names = [kinetics_classnames[int(i)] for i in pred_classes]

    print(f"Top 5 predicted labels for {video_file}: {', '.join(pred_class_names)}")

def get_relu_and_batchnorm_modules(model):
    """
    Get the names of ReLU and BatchNorm3d modules within a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model from which to extract module names.

    Returns:
        List[str]: A list of module names that are instances of nn.ReLU or nn.BatchNorm3d.
                   These names can be used to access or manipulate these modules in the model.
    """
    modules = []

    # Iterate through all modules in the model
    for module_name, module in model.named_modules():
        if isinstance(module, nn.ReLU) or isinstance(module, nn.BatchNorm3d):
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
    global model_name
    def hook_func(model, input, output):
        nonlocal Layer_output
        Layer_output = output

    # Register the forward hook on the desired layer
    hook = layer.register_forward_hook(hook_func)

    if model_name == 'slowfast_r50':
        input = [torch.cat([video_input[i] for video_input in video_inputs], dim=0) for i in range(len(video_inputs[0]))]
    elif model_name == 'x3d_s':
        input = torch.cat(video_inputs, dim=0)

    Layer_output = None
    preds = model(input)
    hook.remove()

    if isLabel:
        get_top_k_predicted_labels(preds)

    activations_batch = Layer_output.detach().cpu().numpy().reshape(len(Layer_output), -1)
    return activations_batch

if __name__ == "__main__":
    # Specify the desired model name here ('slowfast_r50' or 'x3d_s')
    model_name = 'slowfast_r50'

    folder_path = '../test videos/'

    # List the files in the folder with the "processed_" prefix
    processed_videos = [file for file in sorted(os.listdir(folder_path)) if file.startswith('processed_')]

    # Remove "processed_" and ".mp4" and save in a NumPy array
    video_names = [os.path.splitext(file)[0].replace('processed_', '') for file in processed_videos]
    video_names = np.array(video_names)
    np.savez('/result/video_names.npz', video_names)

    # Load the pre-trained model
    model = load_pretrained_model(model_name)

    # Retrieve corresponding layers from which to extract activations
    modules = get_relu_and_batchnorm_modules(model)

    # Transform and store all videos
    transformed_videos = []
    for video_file in tqdm(processed_videos, desc='load data'):
        video_path = os.path.join(folder_path, video_file)
        transformed_video = process_video(model_name, video_path)
        transformed_videos.append(transformed_video)
    
    euclidean_RDM = {}
    pearson_RDM = {}
    spearman_RDM = {}
    batch_size = int(36/4)
    for model_layer in tqdm(modules):
        layer = model
        for attr in model_layer.split('.'):
            layer = getattr(layer, attr)

        activations = []
        # Process videos in batches
        for block in range(0, len(transformed_videos), batch_size):
            batch_videos = transformed_videos[block:block+batch_size]

            if model_name == 'slowfast_r50':
                batch_videos = [[j.to('cuda')[None, ...] for j in i] for i in batch_videos]
            elif model_name == 'x3d_s':
                batch_videos = [i.to('cuda')[None, ...] for i in batch_videos]

            batch_activations = get_activation(model, batch_videos, layer)

            activations.extend(batch_activations)

        activations = np.vstack(activations)

        # average across category
        activations = activations.reshape(6, 6, -1)
        activations = np.mean(activations, axis = 1)

        pearson_RDM[model_layer] = 1 - np.corrcoef(activations)
        
        cor, _ = spearmanr(activations, axis=1)
        spearman_RDM[model_layer] = 1 - cor

        pairwise_differences = activations[:, np.newaxis, :] - activations[np.newaxis, :, :]
        euclidean_RDM[model_layer] = np.linalg.norm(pairwise_differences, axis=2)
        
    # Save the RDM dictionary to a pickle file
    file_path = f'/result/pearson_RDM_{model_name}.pkl'
    with open(file_path, 'wb') as File:
        pickle.dump(pearson_RDM, File)

    file_path = f'/result//spearman_RDM_{model_name}.pkl'
    with open(file_path, 'wb') as File:
        pickle.dump(spearman_RDM, File)

    file_path = f'/result/euclidean_RDM_{model_name}.pkl'
    with open(file_path, 'wb') as File:
        pickle.dump(euclidean_RDM, File)
