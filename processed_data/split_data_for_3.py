import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(source_dir, target_dir, split_ratio=(0.7, 0.15, 0.15)):
    real_src = os.path.join(source_dir, "real_frames")
    fake_src = os.path.join(source_dir, "fake_frames")

    for label, src_dir in [("real", real_src), ("fake", fake_src)]:
        all_files = [f for f in os.listdir(src_dir) if f.endswith(".jpg")]
        train_files, valtest = train_test_split(all_files, test_size=1-split_ratio[0], random_state=42)
        val_files, test_files = train_test_split(valtest, test_size=split_ratio[2]/(split_ratio[1]+split_ratio[2]), random_state=42)

        for split, file_list in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            split_dir = os.path.join(target_dir, split, label)
            os.makedirs(split_dir, exist_ok=True)
            for file in file_list:
                shutil.copy(os.path.join(src_dir, file), os.path.join(split_dir, file))

    print("✅ Dataset split into train/val/test successfully.")

if __name__ == "__main__":
    split_dataset(
        source_dir="/content/Celeb-DF",
        target_dir="/content/MesoXAI/processed_data_split"
    )
