import os
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample
)
from torchvision.io import write_video
from PIL import Image
import numpy as np

def preprocess_video(input_folder, target_fps=30, clip_duration=3):
    """
    Process videos in the input folder and save them with prefix "preprocessed_" in the same folder with adjusted FPS and duration.

    Args:
        input_folder (str): Path to the folder containing video files.
        target_fps (int, optional): Target frames per second (default is 30).
        clip_duration (int, optional): Duration of processed clips in seconds (default is 3).

    Returns:
        None
    """
    # List all files in the input folder
    file_list = os.listdir(input_folder)

    # Define a transform to change FPS from 60 to target_fps (default 30)
    transform = ApplyTransformToKey(key="video", transform=Compose([
        UniformTemporalSubsample(int(clip_duration * target_fps))]))

    # Iterate through each file in the folder
    for filename in file_list:
        if '.mp4' in filename:
            # Read the video
            video = EncodedVideo.from_path(os.path.join(input_folder, filename))
    
            # Process the video as needed
            video_data = video.get_clip(start_sec=0, end_sec=clip_duration)
            video_data = transform(video_data)
            video_data = video_data["video"]
    
            # Duplicate the video_data three times along the time dimension
            video_data = video_data.repeat(1, 3, 1, 1)
    
            output_video_path = os.path.join(input_folder, f'processed_{filename}')
    
            # Save the processed video
            write_video(output_video_path, video_data.permute(1, 2, 3, 0), target_fps)

def preprocess_images(input_folder):
    """
    Process images in the input folder and save them with prefix "img_" in the same folder.

    Args:
        input_folder (str): Path to the folder containing video files.

    Returns:
        None
    """
    # List all files in the input folder
    file_list = os.listdir(input_folder)

    counter = 1
    # Iterate through each file in the folder
    for filename in file_list:
        if '.png' in filename:
            # load image
            image = Image.open(os.path.join(input_folder, filename))
            image = np.array(image)
            # find transparent part (aka green color)
            mask = np.all(image != [0, 255, 0], axis=2)
            image = np.transpose(image, [2, 0, 1])

            # # make image black and white
            # d = np.mean(image, axis=2).astype(int)
            # image = np.repeat(d[:, :, np.newaxis], 3, axis=2)

            # select noise from the same named video file
            file_path = os.path.join(input_folder, str(counter) + '.mp4')
            video = EncodedVideo.from_path(file_path)
            video_data = video.get_clip(start_sec=0, end_sec=3)['video']
            noise = video_data.numpy().astype(int)

            # place noise over the image
            image = np.repeat(image[:, np.newaxis, :, :], noise.shape[1], axis = 1)
            noise[:, :, mask] = image[:, :, mask]
            noise = np.transpose(noise, [1, 2, 3, 0])

            filename = filename.split('.')[0]
            output_video_path = os.path.join(input_folder, f'img_{filename}.mp4')

            # Save the processed video
            write_video(output_video_path, noise, 30)
            counter += 1

folder_path = 'stimuli/'
preprocess_video(folder_path)
preprocess_images(folder_path)