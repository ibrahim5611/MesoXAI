# /MesoXAI/model/evaluate.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve
)

import seaborn as sns
from model.mesonet import Meso4
from utils.visual_utils import save_confusion_matrix_heatmap, save_evaluation_curves

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(test_dir, weight_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = Meso4().to(device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    criterion = nn.BCELoss()

    y_true, y_probs = [], []
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            y_probs.extend(outputs.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    y_probs = np.array(y_probs).flatten()
    y_true = np.array(y_true).flatten()
    y_pred = (y_probs >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_probs)
    cm = confusion_matrix(y_true, y_pred)

    print("\n📊 Evaluation on Test Set:")
    print(f"🎯 Accuracy: {acc:.4f}")
    print(f"📌 Precision: {prec:.4f} | Recall: {rec:.4f} | F1 Score: {f1:.4f}")
    print(f"📈 AUC Score: {auc:.4f}")
    print(f"🔍 Confusion Matrix:\n{cm}")

    # Save confusion matrix heatmap
    save_confusion_matrix_heatmap(cm, labels=["Real", "Fake"], output_path="confusion_matrix_test.png")

    # Save ROC & PR curves
    save_evaluation_curves(y_true, y_probs, output_dir=".")

if __name__ == "__main__":
    evaluate_model(
        test_dir="/content/MesoXAI/processed_data_split/test",
        weight_path="/content/MesoXAI/weights/mesonet_model.pth"
    )
