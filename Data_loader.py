# code is based on https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet

# Necessary installation
# pip install av
# pip install pytorchvideo

import torch
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18
from torchvision.io import read_video
from torch.utils.data import DataLoader, TensorDataset
import os
import av
import torch
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

# model = r3d_18(pretrained=True)
# Choose the `slowfast_r50` model 
model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
model.eval()  # Set the model to evaluation mode

# TODO: read videos from folder
video = EncodedVideo.from_path('/content/human_2.mp4')
video_data = video.get_clip(start_sec=0, end_sec=3)

# TODO: make new function for transform
# transformation parameters
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 8
sampling_rate = 8
frames_per_second = 30

# Note that this transform is specific to the slow_R50 model.
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
            CenterCropVideo(crop_size=(crop_size, crop_size))
        ]
    ),
)

# get transformed video
v = transform(video_data)
inputs = v["video"]
inputs = [i.to(device)[None, ...] for i in inputs]

# TODO: revise this part
# ---------------------------------------------------------------------
# Pass the input clip through the model
preds = model(inputs[None, ...])

# Get the predicted classes
post_act = torch.nn.Softmax(dim=1)
preds = post_act(preds)
pred_classes = preds.topk(k=5).indices[0]

# Map the predicted classes to the label names
pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
print("Top 5 predicted labels: %s" % ", ".join(pred_class_names))
