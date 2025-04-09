# /MesoXAI/preprocessing/extract_frames.py
import cv2
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tqdm import tqdm
from utils.video_utils import get_video_paths, extract_video_name
from preprocessing.face_align import detect_and_crop_face

def extract_frames_from_videos(input_dir, output_dir, frame_interval=5):
    video_paths = get_video_paths(input_dir)
    label = "real" if "real" in input_dir.lower() else "fake"
    save_dir = os.path.join(output_dir, f"{label}_frames")
    os.makedirs(save_dir, exist_ok=True)

    if not video_paths:
        print(f"❌ No videos found in {input_dir}")
        return

    print(f"\n🎞️ Processing {label} videos:")
    for video_path in tqdm(video_paths, desc=f"Processing {label} videos"):
        video_name = extract_video_name(video_path)

        # ✅ Skip if frames for this video already exist
        existing_frames = [f for f in os.listdir(save_dir) if f.startswith(video_name)]
        if len(existing_frames) >= 5:  # You can change this threshold
            print(f"⏩ Skipping {video_name}: frames already extracted ({len(existing_frames)} found)")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠️ Unable to open video: {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        saved_frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                face = detect_and_crop_face(frame)
                if face is not None:
                    face = cv2.resize(face, (256, 256))
                    frame_filename = f"{video_name}_frame_{saved_frame_count:03d}.jpg"
                    cv2.imwrite(os.path.join(save_dir, frame_filename), face)
                    saved_frame_count += 1

            frame_count += 1

        cap.release()
        print(f"✅ {video_name}: {saved_frame_count} frames extracted out of {total_frames}")

    print(f"\n✅ Finished processing {label} directory")

if __name__ == '__main__':
    extract_frames_from_videos("/content/FF++/real", "/content/MesoXAI/processed_data", frame_interval=5)
    extract_frames_from_videos("/content/FF++/fake", "/content/MesoXAI/processed_data", frame_interval=5)
