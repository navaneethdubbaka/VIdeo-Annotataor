import cv2
import pandas as pd
from tqdm import tqdm

video_file = "E:\video_annotaotor\factory001_worker001_part01\factory001_worker001_00002.mp4"
csv_file = "hand_keypoints.csv"
output_video = "video_with_hands.mp4"

df = pd.read_csv(csv_file)

connections = [
(0,1),(1,2),(2,3),(3,4),
(0,5),(5,6),(6,7),(7,8),
(5,9),(9,10),(10,11),(11,12),
(9,13),(13,14),(14,15),(15,16),
(13,17),(17,18),(18,19),(19,20),
(0,17)
]

cap = cv2.VideoCapture(video_file)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

out = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width,height)
)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for frame_id in tqdm(range(total_frames)):

    ret, frame = cap.read()
    if not ret:
        break

    frame_data = df[df["frame"] == frame_id]

    for _, row in frame_data.iterrows():

        xs = []
        ys = []

        for i in range(21):
            xs.append(int(row[f"x{i}"] * width))
            ys.append(int(row[f"y{i}"] * height))

        # draw joints
        for x,y in zip(xs,ys):
            cv2.circle(frame,(x,y),4,(0,255,0),-1)

        # draw bones
        for c in connections:
            cv2.line(
                frame,
                (xs[c[0]],ys[c[0]]),
                (xs[c[1]],ys[c[1]]),
                (255,255,255),
                2
            )

    out.write(frame)

cap.release()
out.release()

print("Video with hand tracking saved!")