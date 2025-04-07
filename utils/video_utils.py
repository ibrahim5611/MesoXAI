# /MesoXAI/utils/video_utils.py

import os

VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']

def is_video_file(filename):
    return any(filename.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)

# utils/video_utils.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob

class FrameDataset(Dataset):
    def __init__(self, folder_path, label, transform=None):
        self.image_paths = glob.glob(os.path.join(folder_path, '*.jpg'))
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(self.label)
