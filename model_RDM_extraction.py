# transformations are based on https://pytorch.org/hub/facebookresearch_pytorchvideo_slowfast/

# Necessary installation
# pip install av
# pip install pytorchvideo

import numpy as np
import torch
import av
import json
import urllib
import pickle
from tqdm import tqdm
import torch.nn as nn
from scipy.stats import spearmanr
from typing import Any
from pytorchvideo.models.hub import slowfast
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
from sklearn.metrics.pairwise import euclidean_distances
from DorsalNet.dorsalnet import DorsalNet

# Define functions

def load_model(model_name, pretrained=True):
    """
    Load a pre-trained PyTorchVideo model.

    Args:
        model_name (str): Name of the model to load.

    Returns:
        torch.nn.Module: Loaded pre-trained model.
    """
    if model_name == 'dorsalnet':
        network = 'airsim_dorsalnet_batch2_model.ckpt-3174400-2021-02-12 02-03-29.666899.pt'

        checkpoint = torch.load(f'DorsalNet/{network}')

        subnet_dict = {}
        for k, v in checkpoint.items():
            if k.startswith("fully_connected"):
                continue
            if k.startswith("subnet.") or k.startswith("module."):
                subnet_dict[k[7:]] = v
            else:
                subnet_dict[k] = v

        model = DorsalNet(False, 32)
        if pretrained:
            model.load_state_dict(subnet_dict)
    elif model_name == 'slowfast_4x16_r50':
        def slowfast_4x16_r50(pretrained: bool = False, progress: bool = True, **kwargs: Any,) -> nn.Module:
            return slowfast._slowfast(
                pretrained=pretrained,
                progress=progress,
                checkpoint_path="https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST_4x16_R50.pyth",
                model_depth = 50,
                slowfast_fusion_conv_kernel_size=(5, 1, 1),
                slowfast_fusion_conv_stride = (8, 1, 1),
                head_pool_kernel_sizes=((4, 7, 7), (32, 7, 7)),
                **kwargs,
            )

        model = slowfast_4x16_r50(pretrained = True)
    else:
        model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=pretrained)

    model = model.eval()
    model = model.to('cuda')
    return model

def apply_video_transform(model_name, video):

    if 'slowfast' in model_name:
        if model_name == 'slowfast_r50' or model_name == 'slowfast_r101':
            side_size = 256
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            crop_size = 256
            num_frames = 32
            sampling_rate = 2
            slowfast_alpha = 4
            num_clips = 10
            num_crops = 3

        elif model_name == 'slowfast_16x8_r101_50_50':
            side_size = 256
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            crop_size = 256
            num_frames = 64
            sampling_rate = 2
            slowfast_alpha = 4
            num_clips = 10
            num_crops = 3

        elif model_name == 'slowfast_4x16_r50':
            side_size = 256
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            crop_size = 256
            num_frames = 32
            sampling_rate = 2
            slowfast_alpha = 8
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
    elif model_name == 'x3d_m':
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        side_size = 256
        crop_size = 256
        num_frames = 16
        sampling_rate = 5

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
    elif model_name == 'dorsalnet':
        side_size = 112
        crop_size = 112
        mean = [123.0, 123.0, 123.0]
        std = [75.0, 75.0, 75.0]
        num_frames = 30
        sampling_rate = 3
        transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    # Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=side_size
                    ),
                    CenterCropVideo(crop_size=(crop_size, crop_size))
                ]
            ),
        )


    end_sec = (num_frames * sampling_rate)/30
    video_data = video.get_clip(start_sec=0, end_sec=end_sec)
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

    # Apply video transformations
    transformed_video_data = apply_video_transform(model_name, video)["video"]

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
    pred_classes = preds.topk(k=1)[1]

    # Map the predicted classes to the label names
    pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]

    print(f"Top 5 predicted labels for {video_file}: {', '.join(pred_class_names)}")

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
    global model_name
    def hook_func(model, input, output):
        nonlocal Layer_output
        Layer_output = output

    # Register the forward hook on the desired layer
    hook = layer.register_forward_hook(hook_func)

    if 'slowfast' in model_name:
        input = [torch.cat([video_input[i] for video_input in video_inputs], dim=0) for i in range(len(video_inputs[0]))]
    else:
        input = torch.cat(video_inputs, dim=0)

    Layer_output = None
    preds = model(input)
    hook.remove()

    if isLabel:
        get_top_k_predicted_labels(preds)

    activations_batch = Layer_output.detach().cpu().numpy().reshape(len(Layer_output), -1)
    return activations_batch


if __name__ == "__main__":
    # Specify the desired model name ('slowfast_r50', 'x3d_m', 'slow_r50' or 'dorsalnet')
    model_name = 'dorsalnet'
    status = 'dynamic' # 'dynamic'
    pretrained = True

    isslow = False
    if model_name == 'slow_r50':
        isslow = True
        model_name = 'slowfast_r50'

    # List the files in the folder with the proper prefix
    prefix = 'processed_' if status == 'dynamic' else 'img_'
    processed_videos = [file for file in sorted(os.listdir('stimuli')) if file.startswith(prefix)]

    # Load the pre-trained model
    model = load_model(model_name, pretrained = pretrained)

    if isslow:
        for module_name, module in model.named_modules():
            if 'branch1' in module_name and 'multipathway_blocks.0' in module_name:
                for param in module.parameters():
                    param.requires_grad = False
                    if isinstance(module, torch.nn.Conv3d):
                        module.weight.data.fill_(0)
                    elif isinstance(module, torch.nn.BatchNorm3d):
                        module.weight.data.fill_(0)
                        module.bias.data.fill_(0)

    # Retrieve corresponding layers from which to extract activations
    modules = get_relu_modules(model)

    # Transform and store all videos
    transformed_videos = []
    for video_file in tqdm(processed_videos, desc='load data'):
        video_path = os.path.join('stimuli', video_file)
        transformed_video = process_video(model_name, video_path)
        transformed_videos.append(transformed_video)

    euclidean_RDM = {}
    pearson_RDM = {}
    spearman_RDM = {}
    batch_size = int(36/1)
    for model_layer in tqdm(modules):
        layer = model
        for attr in model_layer.split('.'):
            layer = getattr(layer, attr)

        activations = []
        # Process videos in batches
        for block in range(0, len(transformed_videos), batch_size):
            batch_videos = transformed_videos[block:block+batch_size]

            if 'slowfast' in model_name:
                batch_videos = [[j.to('cuda')[None, ...] for j in i] for i in batch_videos]
            else:
                batch_videos = [i.to('cuda')[None, ...] for i in batch_videos]

            with torch.no_grad():
                batch_activations = get_activation(model, batch_videos, layer)

            activations.extend(batch_activations)

        activations = np.vstack(activations)
        del batch_videos

        # average across category
        activations = activations.reshape(6, 6, -1)
        activations = np.mean(activations, axis = 1)

        pearson_RDM[model_layer] = 1 - np.corrcoef(activations)

        cor, _ = spearmanr(activations, axis=1)
        spearman_RDM[model_layer] = 1 - cor
        del cor

        euclidean_RDM[model_layer] = euclidean_distances(activations)
        del activations

    if isslow:
        model_name = 'slow_r50'

    random_initialized = 'random/' if not pretrained else ''
    # Save the RDM dictionary to a pickle file
    file_path = f'result/model RDM/{random_initialized}{status}/pearson_RDM_{model_name}.pkl'
    print(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'wb') as File:
        pickle.dump(pearson_RDM, File)

    file_path = f'result/model RDM/{random_initialized}{status}/spearman_RDM_{model_name}.pkl'
    with open(file_path, 'wb') as File:
        pickle.dump(spearman_RDM, File)

    file_path = f'result/model RDM/{random_initialized}{status}/euclidean_RDM_{model_name}.pkl'
    with open(file_path, 'wb') as File:
        pickle.dump(euclidean_RDM, File)
