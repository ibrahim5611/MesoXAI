# xai/grad_cam.py
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.model.eval()

        # Hook the forward and backward passes
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Grad-CAM calculation
        pooled_grad = torch.mean(self.gradients, dim=[0, 2, 3])
        activation = self.activations.squeeze(0)
        for i in range(len(pooled_grad)):
            activation[i, :, :] *= pooled_grad[i]

        heatmap = torch.mean(activation, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap

def apply_gradcam_on_image(model, image_path, device):
    model.to(device)
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    grad_cam = GradCAM(model, model.layer4)  # MesoNet's final conv block
    heatmap = grad_cam.generate(input_tensor)

    heatmap = cv2.resize(heatmap, (256, 256))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image_np = np.array(image.resize((256, 256)))
    overlay = cv2.addWeighted(image_np, 0.6, heatmap_color, 0.4, 0)

    return overlay, heatmap_color
