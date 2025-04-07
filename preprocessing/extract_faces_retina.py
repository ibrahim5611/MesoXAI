import os
import cv2
import torch
from retinaface import RetinaFace
from PIL import Image

input_dir = "D:\Dataset\FF++\FF++"
output_dir = "../processed_data/"
frame_interval = 5
img_size = 256

os.makedirs(output_dir + "real_frames", exist_ok=True)
os.makedirs(output_dir + "fake_frames", exist_ok=True)

def extract_faces_from_video(video_path, label_dir):
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            faces = RetinaFace.extract_faces(img_path=frame, align=True)
            for i, face in enumerate(faces):
                face = cv2.resize(face, (img_size, img_size))
                save_path = os.path.join(output_dir, label_dir, f"{video_name}_{frame_idx}_{i}.jpg")
                cv2.imwrite(save_path, face)
                saved += 1
        frame_idx += 1
    cap.release()
    print(f"Processed {video_name} → {saved} face crops")

# Process real and fake videos
for label, folder in [("real_frames", "real"), ("fake_frames", "fake")]:
    video_folder = os.path.join(input_dir, folder)
    for video in os.listdir(video_folder):
        extract_faces_from_video(os.path.join(video_folder, video), label)
