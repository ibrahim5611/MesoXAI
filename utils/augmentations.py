# utils/augmentations.py

from torchvision import transforms
import random
from PIL import ImageFilter

class RandomGaussianBlur:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))
        return img

def get_train_transforms(img_size=256):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

def get_val_transforms(img_size=256):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])