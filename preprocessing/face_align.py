# /MesoXAI/preprocessing/face_align.py
import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)

def detect_and_crop_face(frame):
    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(img_rgb)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face = frame[y1:y2, x1:x2]
                if face.size != 0:
                    return face
        return None

    except Exception as e:
        print(f"[Face Detection Error] {str(e)}")
        return None