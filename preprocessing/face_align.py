# /MesoXAI/preprocessing/face_align.py

from facenet_pytorch import MTCNN
import torch
import cv2
import numpy as np

# Initialize MTCNN detector
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)


def detect_and_align_faces(frame):
    """
    Detect and align faces in a frame using MTCNN.
    Returns a list of aligned face crops.
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = []
    
    try:
        boxes, _ = mtcnn.detect(img_rgb)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]
                if face.size != 0:
                    faces.append(face)
    except Exception as e:
        print(f"Face detection error: {e}")

    return faces
