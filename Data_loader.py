# transformations are based on https://pytorch.org/hub/facebookresearch_pytorchvideo_slowfast/

# Necessary installation
# pip install av
# pip install pytorchvideo

import torch
import av
import json
import urllib
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
    model = model.to('cpu')
    return model

def read_video_from_path(video_path, start_sec, end_sec):
    """
    Read and extract a clip from a video file.

    Args:
        video_path (str): Path to the video file.
        start_sec (int): Start time in seconds.
        end_sec (int): End time in seconds.

    Returns:
        pytorchvideo.data.encoded_video.EncodedVideo: Encoded video clip.
    """
    video = EncodedVideo.from_path(video_path)
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
    return video_data

def apply_video_transform(video_data):
    """
    Apply transformations to video data.

    Args:
        video_data (dict): Video data dictionary.

    Returns:
        dict: Transformed video data.
    """
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

    transformed_video = transform(video_data)
    return transformed_video

def get_labels_name():
    import json
    import urllib
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

def get_top_k_predicted_labels(model, video_inputs, k=5):
    """
    Get the top-k predicted labels for the given video inputs.

    Args:
        model (torch.nn.Module): Pre-trained model.
        video_inputs (torch.Tensor): Video input tensor.
        k (int): Number of top labels to retrieve.

    Returns:
        list: Top-k predicted labels.
    """
    video_inputs = [i.to('cpu')[None, ...] for i in video_inputs]
    preds = model(video_inputs)

    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=k).indices[0]

    kinetics_classnames = get_labels_name()

    pred_class_names = [kinetics_classnames[int(i)] for i in pred_classes]
    return pred_class_names

if __name__ == "__main__":
    # Load the pre-trained model
    model = load_pretrained_model('slowfast_r50')

    # Read video from a file
    video_data = read_video_from_path('/content/human_2.mp4', start_sec=0, end_sec=3)

    # Apply video transformations
    transformed_video_data = apply_video_transform(video_data)["video"]

    # Get top 5 predicted labels
    top_predicted_labels = get_top_k_predicted_labels(model, transformed_video_data)
    print("Top 5 predicted labels:", ", ".join(top_predicted_labels))
