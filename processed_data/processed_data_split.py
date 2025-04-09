# /MesoXAI/preprocessing/split_dataset.py

import os
import random
import shutil
from tqdm import tqdm

def split_dataset(source_dir, output_dir, split_ratio=0.8):
    """
    Splits images in each class subfolder into train and val sets.
    """
    classes = ['real_frames', 'fake_frames']
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = [img for img in os.listdir(cls_path) if img.lower().endswith(('.jpg', '.png'))]
        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        for split, split_imgs in zip(['train', 'val'], [train_imgs, val_imgs]):
            split_dir = os.path.join(output_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            print(f"🔄 Copying {len(split_imgs)} images to {split_dir}...")
            for img in tqdm(split_imgs):
                src = os.path.join(cls_path, img)
                dst = os.path.join(split_dir, img)
                shutil.copy(src, dst)

if __name__ == "__main__":
    source_dir = "/content/MesoXAI/processed_data"      # Change if needed
    output_dir = "/content/MesoXAI/processed_data_split"  # New split directory
    split_dataset(source_dir, output_dir, split_ratio=0.8)
