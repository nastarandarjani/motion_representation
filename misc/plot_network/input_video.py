import av
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
from DorsalNet.dorsalnet import DorsalNet
import cv2
import torch
import numpy as np
from moviepy.editor import ImageSequenceClip
from IPython.display import display, HTML


def apply_video_transform(model_name, video):
    slowfast_alpha = 0

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
                    ShortSideScale(size=side_size),
                    CenterCropVideo(crop_size)
                ]
            ),
        )
    elif model_name == 'dorsalnet':
        side_size = 256
        crop_size = 256
        num_frames = 30
        sampling_rate = 3
        transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x/255.0),
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
    return num_frames, slowfast_alpha, transformed_video

def process_video(model_name, video_path):
    # Read video from a file
    video = EncodedVideo.from_path(video_path)

    # Apply video transformations
    num_frames, slowfast_alpha, transformed_video_data = apply_video_transform(model_name, video)

    return num_frames, slowfast_alpha, transformed_video_data["video"]

model_name = ['slowfast_r50', 'slowfast_r101', 'slowfast_16x8_r101_50_50', 'slowfast_4x16_r50', 'dorsalnet']

for model in model_name:
    video_file = [file for file in sorted(os.listdir('stimuli')) if file.startswith('processed_')][0]
    video_path = os.path.join('stimuli', video_file)
    num_frames, slowfast_alpha, transformed_video = process_video(model, video_path)

    if 'slowfast' in model:
        x = ['_slow', '_fast']
        fps = [num_frames//slowfast_alpha, num_frames]
    else:
        transformed_video = torch.unsqueeze(transformed_video, 0)
        x = ['']
        fps = [num_frames]

    for ind, typ in enumerate(x):
        images = transformed_video[ind].numpy().transpose(1, 2, 3, 0)
        clip = ImageSequenceClip(list(images), fps = fps[ind])
        clip.write_videofile(f'video/{model}{typ}.mp4', codec='libx264', audio=False)
        # display(clip.ipython_display(width=400))
