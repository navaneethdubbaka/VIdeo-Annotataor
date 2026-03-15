import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

csv_file = "hand_keypoints.csv"
output_video = "hand_motion_fast.mp4"

df = pd.read_csv(csv_file)

connections = [
(0,1),(1,2),(2,3),(3,4),
(0,5),(5,6),(6,7),(7,8),
(5,9),(9,10),(10,11),(11,12),
(9,13),(13,14),(14,15),(15,16),
(13,17),(17,18),(18,19),(19,20),
(0,17)
]

frames = df["frame"].unique()

width = 800
height = 800

out = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*'mp4v'),
    30,
    (width,height)
)

for frame_id in tqdm(frames):

    canvas = np.zeros((height,width,3), dtype=np.uint8)

    frame_data = df[df["frame"] == frame_id]

    for _, row in frame_data.iterrows():

        xs = []
        ys = []

        for i in range(21):
            xs.append(row[f"x{i}"])
            ys.append(row[f"y{i}"])

        px = (np.array(xs) * width).astype(int)
        py = (np.array(ys) * height).astype(int)

        for i,(x,y) in enumerate(zip(px,py)):
            cv2.circle(canvas,(x,y),4,(0,255,0),-1)

        for c in connections:
            cv2.line(
                canvas,
                (px[c[0]],py[c[0]]),
                (px[c[1]],py[c[1]]),
                (255,255,255),
                2
            )

    out.write(canvas)

out.release()

print("Video created!")