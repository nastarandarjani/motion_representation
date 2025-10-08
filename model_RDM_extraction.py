import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import pickle
from tqdm import tqdm
import torch.nn as nn
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
import torch.nn.init as init
from utils.util import load_model


# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change to a relative directory from the script's location
os.chdir(script_dir)

# Define functions

def apply_video_transform(model_name, video):
    """
    Apply video transformations based on the model name.

    Args:
        model_name (str): Name of the model.
        video (EncodedVideo): Encoded video object.

    Returns:
        transformed_video (dict): Transformed video data.
    """
    if 'slowfast' in model_name:
        if model_name == 'slowfast_r50' or model_name == 'slowfast_r101':
            side_size = 256
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            crop_size = 256
            num_frames = 32
            sampling_rate = 2
            slowfast_alpha = 4

        elif model_name == 'slowfast_16x8_r101_50_50':
            side_size = 256
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            crop_size = 256
            num_frames = 64
            sampling_rate = 2
            slowfast_alpha = 4

        elif model_name == 'slowfast_4x16_r50':
            side_size = 256
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            crop_size = 256
            num_frames = 32
            sampling_rate = 2
            slowfast_alpha = 8

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
    elif model_name == "res_r50":
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 8
        sampling_rate = 8

        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size=side_size),
                    CenterCropVideo(crop_size),
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

        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size=side_size),
                    CenterCropVideo(crop_size),
                ]
            ),
        )
    elif model_name == "R2PLUS1D":
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        side_size = 256
        crop_size = 256
        num_frames = 16
        sampling_rate = 4

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


def get_activation(model, video_inputs, layer, model_name):
    """
    Get the activation from a specified layer of a pre-trained model.

    Args:
        model (torch.nn.Module): Pre-trained model.
        video_inputs (torch.Tensor): Video input tensor.
        layer (torch.nn.Module): The layer from which to extract activation.

    Returns:
        numpy.ndarray: Activation values as a NumPy array.
    """

    def hook_func(model, input, output):
        nonlocal Layer_output
        Layer_output = output

    # Register the forward hook on the desired layer
    hook = layer.register_forward_hook(hook_func)

    if "slowfast" in model_name:
        input = [
            torch.cat([video_input[i] for video_input in video_inputs], dim=0)
            for i in range(len(video_inputs[0]))
        ]
    else:
        input = torch.cat(video_inputs, dim=0)

    Layer_output = None
    model(input)
    hook.remove()

    activations_batch = Layer_output.detach().cpu().numpy().reshape(len(Layer_output), -1)
    return activations_batch


if __name__ == "__main__":
    # Specify the desired model name ('slowfast_r50', 'R2PLUS1D', 'x3d_m', 'slow_r50', 'res_r50' or 'dorsalnet')
    model_name = "res_r50"
    dataset = "k400"  # k400, ssv2
    status = "dynamic"  # 'dynamic'
    pretrained = True  # True, False, cpc
    random_layer = ""  # '', 'slow/', 'fast/', 'fusion/'
    folder = "stimuli"  # "stimuli"

    isslow = False
    if model_name == 'slow_r50':
        isslow = True
        model_name = 'slowfast_r50'

    # List the files in the folder with the proper prefix
    prefix = 'processed_' if status == 'dynamic' else 'img_'
    if folder == "stimuli":
        processed_videos = [
            file for file in sorted(os.listdir(folder)) if file.startswith(prefix)
        ]
    else:
        processed_videos = sorted(os.listdir(folder))

    # Load the pre-trained model
    model = load_model(model_name, pretrained=pretrained, dataset=dataset)
    model = model.eval()
    model = model.to("mps")

    for module_name, module in model.named_modules():
        if (
            ("multipathway_blocks.0" in module_name and random_layer == "slow/")
            or ("multipathway_blocks.1" in module_name and random_layer == "fast/")
            or ("multipathway_fusion" in module_name and random_layer == "fusion/")
        ):
            if isinstance(module, torch.nn.Conv3d) or isinstance(
                module, torch.nn.Linear
            ):
                # Reset the weights using PyTorch's initialization functions
                init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    init.constant_(module.bias, 0.0)
            elif isinstance(module, torch.nn.BatchNorm3d):
                # Reset batch normalization parameters
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
                init.constant_(module.running_mean, 0)
                init.constant_(module.running_var, 1)

    if isslow:
        for module_name, module in model.named_modules():
            if "multipathway_fusion" in module_name:
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
        video_path = os.path.join(folder, video_file)
        transformed_video = process_video(model_name, video_path)
        transformed_videos.append(transformed_video)

    euclidean_RDM = {}
    pearson_RDM = {}
    spearman_RDM = {}
    batch_size = int(36 / 2)
    for model_layer in tqdm(modules):
        layer = model
        for attr in model_layer.split('.'):
            layer = getattr(layer, attr)

        activations = []
        # Process videos in batches
        for block in range(0, len(transformed_videos), batch_size):
            batch_videos = transformed_videos[block:block+batch_size]

            if 'slowfast' in model_name:
                batch_videos = [
                    [j.to("mps")[None, ...] for j in i] for i in batch_videos
                ]
            else:
                batch_videos = [i.to("mps")[None, ...] for i in batch_videos]

            with torch.no_grad():
                batch_activations = get_activation(
                    model, batch_videos, layer, model_name
                )

            activations.extend(batch_activations)

        activations = np.vstack(activations)
        del batch_videos

        # average across category
        if folder == "stimuli":
            activations = activations.reshape(6, 6, -1)
            activations = np.mean(activations, axis=1)

        pearson_RDM[model_layer] = 1 - np.corrcoef(activations)

        # cor, _ = spearmanr(activations, axis=1)
        # spearman_RDM[model_layer] = 1 - cor
        # del cor

        # euclidean_RDM[model_layer] = euclidean_distances(activations)
        # del activations

    if isslow:
        model_name = 'slow_r50'

    random_initialized = "random/" if random_layer != "" else ""
    is_cpc = "cpc/" if pretrained == "cpc" else ""
    pretrained = "untrained/" if not pretrained else ""
    data = f"{dataset}/" if not (dataset == "k400") else ""
    # Save the RDM dictionary to a pickle file
    file_path = f"result/model RDM/{folder}/{data}{is_cpc}{pretrained}{random_initialized}{status}/{random_layer}pearson_RDM_{model_name}.pkl"
    print(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'wb') as File:
        pickle.dump(pearson_RDM, File)

    # file_path = f"result/model RDM/{data}{is_cpc}{pretrained}{random_initialized}{status}/{random_layer}spearman_RDM_{model_name}.pkl"
    # with open(file_path, 'wb') as File:
    #     pickle.dump(spearman_RDM, File)

    # file_path = f"result/model RDM/{data}{is_cpc}{pretrained}{random_initialized}{status}/{random_layer}euclidean_RDM_{model_name}.pkl"
    # with open(file_path, 'wb') as File:
    #     pickle.dump(euclidean_RDM, File)