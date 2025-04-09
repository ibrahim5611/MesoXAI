# /MesoXAI/preprocessing/split_dataset.py

import os
import random
import shutil
from tqdm import tqdm

def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits images in each class subfolder into train, val, and test sets.
    """
    classes = ['real_frames', 'fake_frames']
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = [img for img in os.listdir(cls_path) if img.lower().endswith(('.jpg', '.png'))]
        random.shuffle(images)

        train_end = int(len(images) * train_ratio)
        val_end = train_end + int(len(images) * val_ratio)

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        for split, split_imgs in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            # Rename class folder to 'real' or 'fake'
            new_cls = 'real' if 'real' in cls.lower() else 'fake'
            split_dir = os.path.join(output_dir, split, new_cls)
            os.makedirs(split_dir, exist_ok=True)
            print(f"🔄 Copying {len(split_imgs)} images to {split_dir}...")
            for img in tqdm(split_imgs):
                src = os.path.join(cls_path, img)
                dst = os.path.join(split_dir, img)
                shutil.copy(src, dst)

if __name__ == "__main__":
    source_dir = "/content/MesoXAI/processed_data"       # Path to real_frames and fake_frames
    output_dir = "/content/MesoXAI/processed_data_split" # Output path with train/val/test splits
    split_dataset(source_dir, output_dir)
    print("✅ Dataset split into train, val, and test sets successfully.")
    print(f"📁 Source Directory: {source_dir}")