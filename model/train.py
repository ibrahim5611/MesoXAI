# /MesoXAI/model/train.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.mesonet import Meso4
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             roc_curve, precision_recall_curve)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.early_stopping import EarlyStopping

from PIL import Image

misclassified_dir = "/content/MesoXAI/misclassified"
os.makedirs(misclassified_dir, exist_ok=True)

cm_save_path = "/content/MesoXAI/eval/"
os.makedirs(cm_save_path, exist_ok=True)

roc_pr_dir = "/content/MesoXAI/eval/curves"
os.makedirs(roc_pr_dir, exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_existing_weights(model, weight_path):
    if os.path.exists(weight_path):
        print(f"🧠 Loading existing weights from {weight_path}")
        existing_state = torch.load(weight_path)
        model.load_state_dict(existing_state, strict=False)
    return model

def save_combined_weights(model, old_path, new_path):
    if os.path.exists(old_path):
        old_weights = torch.load(old_path)
        new_weights = model.state_dict()
        combined_weights = {}
        for key in new_weights:
            if key in old_weights:
                combined_weights[key] = (new_weights[key] + old_weights[key]) / 2
            else:
                combined_weights[key] = new_weights[key]
        torch.save(combined_weights, new_path)
        print(f"💾 Combined weights saved to {new_path}")
    else:
        torch.save(model.state_dict(), new_path)
        print(f"💾 Model weights saved to {new_path} (initial save)")

def plot_and_save_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

def save_roc_pr_curves(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(roc_pr_dir, 'roc_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(recall, precision, label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(roc_pr_dir, 'pr_curve.png'))
    plt.close()

def save_misclassified_samples(images, labels, preds, paths):
    for img, label, pred, path in zip(images, labels, preds, paths):
        if int(pred) != int(label):
            img_np = img.permute(1, 2, 0).cpu().numpy() * 255
            img_pil = Image.fromarray(img_np.astype(np.uint8))
            base_name = os.path.basename(path)
            img_pil.save(os.path.join(misclassified_dir, f"{label}_{pred}_{base_name}"))

def compute_metrics(y_true, y_pred_probs):
    y_pred = (y_pred_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_probs)

    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)
    fnr = fn / (fn + tp + 1e-8)
    weighted_prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n📊 Confusion Matrix: \n{cm}")
    print(f"🎯 Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1 Score: {f1:.4f}")
    print(f"📈 AUC: {auc:.4f}")
    print(f"🔍 TPR: {tpr:.4f} | FPR: {fpr:.4f} | TNR: {tnr:.4f} | FNR: {fnr:.4f}")
    print(f"📌 Weighted Precision: {weighted_prec:.4f}")

    plot_and_save_confusion_matrix(y_true, y_pred, cm_save_path)
    save_roc_pr_curves(y_true, y_pred_probs)

def train_model():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset_path = '/content/MesoXAI/processed_data_split'
    train_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'val'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = Meso4().to(device)
    weight_path = "/content/MesoXAI/weights/mesonet_model.pth"
    model = load_existing_weights(model, weight_path)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    num_epochs = 30
    early_stopper = EarlyStopping(patience=3, verbose=True, delta=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, labels in loop:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"📉 Epoch {epoch+1} - Avg Training Loss: {avg_loss:.4f}")

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        val_paths = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        print(f"✅ Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")

        all_preds_np = np.array(all_preds).flatten()
        all_labels_np = np.array(all_labels).flatten()
        compute_metrics(all_labels_np, all_preds_np)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_combined_weights(model, weight_path, weight_path)

        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print("⏹️ Early stopping triggered!")
            break

if __name__ == "__main__":
    train_model()
