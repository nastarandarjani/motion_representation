import os

import cv2
import numpy as np
import scipy.io
from tqdm import tqdm

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change to a relative directory from the script's location
os.chdir(script_dir)

mat_root = "flow"
n_dots = 2500
max_life = 17
frames = 180
imW, imH = 720, 460
fps = 60
output_dir = "./generated_videos"
os.makedirs(output_dir, exist_ok=True)

video_list = sorted(os.listdir(mat_root))

# --- Generate and save videos ---
for idx, mat_path in enumerate(tqdm(video_list)):
    name, label = mat_path.split(".")[0].split("_")
    for i in range(10):
        out_path = os.path.join(output_dir, f"{name}_{label}{i}.mp4")

        optFlow = scipy.io.loadmat(os.path.join(mat_root, mat_path))["optFlow"]

        X = np.random.choice(np.arange(imW), n_dots, replace=True)
        Y = np.random.choice(np.arange(imH), n_dots, replace=True)
        Life = np.random.choice(np.arange(1, max_life + 1), n_dots, replace=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (imW, imH))

        for iFrame in range(frames):
            spX = optFlow[Y, X, 0, iFrame]
            spY = optFlow[Y, X, 1, iFrame]
            X = np.round(X + spX).astype(int)
            Y = np.round(Y + spY).astype(int)

            Life -= 1
            out_of_bounds = (X < 0) | (X >= imW) | (Y < 0) | (Y >= imH) | (Life == 0)
            X[out_of_bounds] = np.random.choice(np.arange(imW), np.sum(out_of_bounds), replace=True)
            Y[out_of_bounds] = np.random.choice(np.arange(imH), np.sum(out_of_bounds), replace=True)
            Life[out_of_bounds] = np.random.choice(np.arange(1, max_life + 1), np.sum(out_of_bounds), replace=True)

            # Gray background
            frame = np.ones((imH, imW, 3), dtype=np.uint8) * 128

            # Draw dots (white, radius = 1 pixel → 2 pixel diameter)
            for x, y in zip(X, Y):
                cv2.circle(frame, (x, y), radius=1, color=(255, 255, 255), thickness=-2, lineType=cv2.LINE_AA)

            writer.write(frame)

        writer.release()
        print(f"Saved: {out_path}")