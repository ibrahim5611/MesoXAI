# xai/visualize.py
import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.mesonet import Meso4
from grad_cam import apply_gradcam_on_image
from PIL import Image

def visualize_and_save_gradcam(image_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Meso4()
    model.load_state_dict(torch.load('../weights/mesonet_model.pth', map_location=device))
    model.eval()

    result, heatmap = apply_gradcam_on_image(model, image_path, device)

    os.makedirs(output_path, exist_ok=True)
    image_name = os.path.basename(image_path)
    save_path = os.path.join(output_path, f"gradcam_{image_name}")
    Image.fromarray(result).save(save_path)
    print(f"Saved Grad-CAM to {save_path}")

# Example usage
if __name__ == "__main__":
    image_path = r"D:\Deepfake\MesoXAI\processed_data\real_frames\01__kitchen_pan_0360.jpg"
    output_path = "D:\Deepfake\MesoXAI\XAI_outputs\gradcam"
    visualize_and_save_gradcam(image_path, output_path)
