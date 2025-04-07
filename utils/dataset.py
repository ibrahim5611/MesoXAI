import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.transform = transform if transform else self.default_transforms()

        self.data = []
        self.labels = []

        # Load real frames
        for filename in os.listdir(self.real_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                self.data.append(os.path.join(self.real_dir, filename))
                self.labels.append(0)  # 0 = Real

        # Load fake frames
        for filename in os.listdir(self.fake_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                self.data.append(os.path.join(self.fake_dir, filename))
                self.labels.append(1)  # 1 = Fake

                
    def default_transforms(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
