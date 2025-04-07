# /MesoXAI/preprocessing/extract_frames.py

import cv2
import os
from tqdm import tqdm
from face_align import detect_and_align_faces

# Parameters
FRAME_SKIP = 5  # extract every 5th frame
FRAME_SIZE = (256, 256)  # Resize to this


def extract_frames_from_video(video_path, output_dir, label):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    extracted_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Check if frames already extracted
    existing_frames = [f for f in os.listdir(output_dir) if f.startswith(video_name)]
    if existing_frames:
        print(f"[SKIP] Frames for '{video_name}' already exist. Skipping extraction.")
        cap.release()
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_SKIP == 0:
            # Detect and align face
            faces = detect_and_align_faces(frame)
            for i, face in enumerate(faces):
                resized_face = cv2.resize(face, FRAME_SIZE)
                normalized_face = resized_face / 255.0  # normalize to [0, 1]
                frame_filename = f"{video_name}_{frame_count:04d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, (normalized_face * 255).astype('uint8'))
                extracted_count += 1

        frame_count += 1

    cap.release()
    print(f"[{video_name}] Extracted {extracted_count} frames from {total_frames} total frames.")


def process_dataset(root_dir, output_base_dir):
    for label in ['real', 'fake']:
        video_dir = os.path.join(root_dir, label)
        output_dir = os.path.join(output_base_dir, f"{label}_frames")
        videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

        for video in tqdm(videos, desc=f"Processing {label} videos"):
            video_path = os.path.join(video_dir, video)
            extract_frames_from_video(video_path, output_dir, label)


if __name__ == "__main__":
    input_dir = "D:\Dataset\FF++\FF++"
    output_dir = "D:\Deepfake\MesoXAI\processed_data"
    process_dataset(input_dir, output_dir)
    print("✅ Frame extraction completed.")
    print(f"✅ Frames saved to {output_dir}.")