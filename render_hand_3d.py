import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

csv_file = "hand_keypoints.csv"
output_video = "hand_3d_motion.mp4"

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

width = 500
height = 500

out = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*'mp4v'),
    30,
    (width,height)
)

for frame_id in tqdm(frames):

    frame_data = df[df["frame"] == frame_id]

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')

    for _, row in frame_data.iterrows():

        xs = []
        ys = []
        zs = []

        for i in range(21):
            xs.append(row[f"x{i}"])
            ys.append(row[f"y{i}"])
            zs.append(-i*0.01)

        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)

        ax.scatter(xs, ys, zs)

        for c in connections:
            ax.plot(
                [xs[c[0]], xs[c[1]]],
                [ys[c[0]], ys[c[1]]],
                [zs[c[0]], zs[c[1]]]
            )

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(-0.3,0.1)

    fig.canvas.draw()

    img = np.asarray(fig.canvas.buffer_rgba())
    img = img[:,:,:3]

    img = cv2.resize(img,(width,height))

    out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    plt.close()

out.release()

print("3D hand video saved")