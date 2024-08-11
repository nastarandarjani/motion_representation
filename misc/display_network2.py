# pip install torchviz

import torch
from torchviz import make_dot
import sys
sys.path.append("..")
from video_RDM_extraction import process_video, load_pretrained_model

# Choose the model you want to use: 'x3d_m' or 'slowfast_r50'
model_name = 'x3d_m'

# Load the pre-trained model based on the chosen model_name
model = load_pretrained_model(model_name)

# Process the video using
file_path = '/stimuli/processed_ball_1.mp4'
transformed_video = process_video(model_name, file_path)

if model_name == 'slowfast_r50':
    input = [i.to('cuda')[None, ...] for i in transformed_video]
else:
    input = transformed_video.to('cuda')[None, ...]

# Create a graph of the computation
graph = make_dot(input, params=dict(model.named_parameters()))

# Save the graph to a file (optional)
graph.render(f'{model_name}')
