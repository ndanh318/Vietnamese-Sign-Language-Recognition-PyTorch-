import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights

from src.config import CLASSES


class Classifier:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 26)
        # self.model.to(device)
        # Load model
        if self.model_path is not None and os.path.isfile(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint["model_params"])
        else:
            print("No checkpoint provided")
            exit(0)

    def prediction(self, ori_image, draw=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        # Preprocessing
        image = cv2.resize(ori_image, (224, 224))
        image = np.transpose(image, (2, 0, 1)) / 255.
        image = image[None, :, :, :]
        image = torch.from_numpy(image).float()
        image = image.to(device)
        # Prediction
        with torch.no_grad():
            results = self.model(image)
            prediction = torch.argmax(results)

        if draw:
            cv2.putText(ori_image, CLASSES[prediction.item()], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0),
                        2)

        return list(results[0]), prediction
