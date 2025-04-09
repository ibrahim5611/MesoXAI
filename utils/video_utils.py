# /MesoXAI/utils/video_utils.py
import os

def get_video_paths(input_dir):
    """
    Recursively get all video file paths in the given directory.
    Supported formats: .mp4, .avi, .mov, .mkv
    """
    supported_exts = ['.mp4', '.avi', '.mov', '.mkv']
    video_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_exts):
                video_paths.append(os.path.join(root, file))
    return video_paths

def extract_video_name(video_path):
    """
    Extracts the base name of the video file (without extension).
    """
    return os.path.splitext(os.path.basename(video_path))[0]
