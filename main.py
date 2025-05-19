import cv2
import os
from ultralytics import YOLO
import numpy as np

model = YOLO("./trained_model/lane_detection.pt")

# Input and output video paths
input_video_path = "src.mp4"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
output_video_path = os.path.join(output_dir, "segmented_output.mp4")

# Open input video
cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

mask_color = (128, 0, 0) 
alpha = 0.4

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False, iou=0.8, conf=0.2)[0]

    overlay = frame.copy()

    if results.masks and results.masks.xy:
        for seg in results.masks.xy:
            points = np.array(seg, dtype=np.int32)
            cv2.fillPoly(overlay, [points], color=mask_color)

        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Show the result
    cv2.imshow("YOLO Segmentation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Saved segmented video to: {output_video_path}")
