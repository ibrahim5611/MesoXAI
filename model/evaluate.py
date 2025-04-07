# /model/evaluate.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import shutil

from mesonet import Meso4

# ========== CONFIG ==========
TEST_DIR = "../processed_data"
WEIGHTS_PATH = "../weights/mesonet_model.pth"
SAVE_MISCLASSIFIED_DIR = "./misclassified"
CONF_MATRIX_SAVE_PATH = "./confusion_matrix.png"
IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 0  # ⚠️ Use 0 for Windows to avoid multiprocessing issues
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== TRANSFORMS ==========
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# ========== CUSTOM COLLATE FN ==========
def collate_skip_corrupted(batch):
    return [sample for sample in batch if sample is not None]

# ========== DATASET ==========
class FrameDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform

        real_path = os.path.join(root_dir, "real_frames")
        fake_path = os.path.join(root_dir, "fake_frames")

        for folder, label in [(real_path, 0), (fake_path, 1)]:
            for img_file in os.listdir(folder):
                img_path = os.path.join(folder, img_file)
                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label, img_path
        except (UnidentifiedImageError, OSError):
            return None

# ========== EVALUATE ==========
def evaluate():
    print("Loading model...")
    model = Meso4()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    test_dataset = FrameDataset(TEST_DIR, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_skip_corrupted  # ✅ SAFE!
    )

    all_preds = []
    all_labels = []
    misclassified = []

    print("Evaluating...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if not batch:
                continue
            images, labels, paths = zip(*batch)
            images = torch.stack(images).to(DEVICE)
            labels = torch.tensor(labels).to(DEVICE)

            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs >= 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for pred, label, path in zip(preds, labels, paths):
                if pred != label:
                    misclassified.append(path)

    # ========== METRICS ==========
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)

    print("\n====== Evaluation Report ======")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision      : {prec:.4f}")
    print(f"Recall         : {rec:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print(f"AUC-ROC        : {auc:.4f}")
    print("================================")

    # ========== CONFUSION MATRIX ==========
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(CONF_MATRIX_SAVE_PATH)
    plt.close()
    print(f"Confusion matrix saved to {CONF_MATRIX_SAVE_PATH}")

    # ========== SAVE MISCLASSIFIED ==========
    if os.path.exists(SAVE_MISCLASSIFIED_DIR):
        shutil.rmtree(SAVE_MISCLASSIFIED_DIR)
    os.makedirs(SAVE_MISCLASSIFIED_DIR, exist_ok=True)

    for img_path in misclassified:
        try:
            img = Image.open(img_path).convert("RGB")
            base_name = os.path.basename(img_path)
            img.save(os.path.join(SAVE_MISCLASSIFIED_DIR, base_name))
        except Exception as e:
            print(f"[!] Error saving misclassified frame: {img_path} — {e}")

    print(f"Misclassified frames saved to {SAVE_MISCLASSIFIED_DIR}")

if __name__ == "__main__":
    evaluate()
