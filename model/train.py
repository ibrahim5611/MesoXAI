# /MesoXAI/model/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from tqdm import tqdm
from datetime import datetime
from mesonet import Meso4

# Configuration
DATA_DIR = "../processed_data"
REAL_DIR = os.path.join(DATA_DIR, "real_frames")
FAKE_DIR = os.path.join(DATA_DIR, "fake_frames")
WEIGHTS_DIR = "../weights"
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
IMG_SIZE = 256

os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Custom Dataset
class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir)]
        self.fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir)]
        self.images = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = datasets.folder.default_loader(img_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Dataset & DataLoader
dataset = DeepfakeDataset(REAL_DIR, FAKE_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, Loss, Optimizer
model = Meso4()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weight_path = os.path.join(WEIGHTS_DIR, f"mesonet_model_{timestamp}.pth")
    torch.save(model.state_dict(), weight_path)
    print(f"Saved weights: {weight_path}")

print("Training Complete.")
