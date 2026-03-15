import cv2
import mediapipe as mp
import csv
from tqdm import tqdm

input_video = video_path
output_video = "tracked_output.mp4"
csv_file = "hand_keypoints_3d.csv"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(input_video)

width = 960
height = 540

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width,height))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# open csv file
with open(csv_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow(["frame","hand","landmark","x","y","z"])

    for frame_id in tqdm(range(total_frames)):

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame,(width,height))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                for i,lm in enumerate(hand_landmarks.landmark):

                    writer.writerow([
                        frame_id,
                        hand_idx,
                        i,
                        lm.x,
                        lm.y,
                        lm.z
                    ])

        out.write(frame)

cap.release()
out.release()

print("Processing finished")