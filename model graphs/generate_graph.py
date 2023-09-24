# pip install torchviz 
from torchviz import make_dot
import sys
sys.path.append('../motion_representation')  # Add the parent directory to the Python path
from motion_representation.video_activation_extraction import (
  load_pretrained_model,
  process_video
)

# Specify the model name ('slowfast_r50' or 'x3d_s')
model_name = 'x3d_s'

# Load the pre-trained model
model = load_pretrained_model(model_name)

# Path to the processed video
video = '../test videos/processed_ball_1.mp4'

# Process the video
transformed_video = process_video(model_name, video)

# Convert the video to CPU and add a batch dimension
video_inputs = transformed_video.to('cpu')[None, ...]

# Forward pass to compute the computation graph
output = model(video_inputs)

# Create a graph of the computation
graph = make_dot(output, params=dict(model.named_parameters()))

# Save the graph as a PDF
graph.render(model_name)
