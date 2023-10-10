# This script loads a pre-trained deep learning model, processes a video using the model,
# and visualizes the computation graph of the model. The visualization is saved as a PDF file.

# Install the 'torchviz' library if not already installed
# pip install torchviz 
from torchviz import make_dot
import sys

# Add the parent directory to the Python path to access custom modules
sys.path.append('../motion_representation')

# Import necessary functions from a custom module
from motion_representation.video_activation_extraction import (
    load_pretrained_model,
    process_video
)

# Specify the model name ('slowfast_r50' or 'x3d_m')
model_name = 'x3d_m'

# Load the pre-trained model
model = load_pretrained_model(model_name)

# Path to the processed video
video = '../test videos/processed_ball_1.mp4'

# Process the video using the pre-trained model
transformed_video = process_video(model_name, video)

# Convert the video to GPU and add a batch dimension
if model_name == 'slowfast_r50':
    video_inputs = [i.to('cuda')[None, ...] for i in transformed_video]
else:
    video_inputs = transformed_video.to('cuda')[None, ...]

# Forward pass to compute the computation graph
output = model(video_inputs)

# Create a graph of the computation
graph = make_dot(output, params=dict(model.named_parameters()))

# Save the graph as a PDF with the model name as the filename
graph.render(model_name)
