# pip install graphviz
# pip install torchview

import torch
from torchview import draw_graph
import urllib
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

# Display the visual representation of the model graph
model_graph = draw_graph(model, input_data=input, depth=10, expand_nested=True)
model_graph.visual_graph
