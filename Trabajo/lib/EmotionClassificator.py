import numpy as np
import torch
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
import torch.nn as nn


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes, dropout_p=0.5, freeze_backbone=False):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        if freeze_backbone:
            for param in self.resnet.features[:-1].parameters():
                param.requires_grad = False
                
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.resnet(x)


class EmotionClassificator:
    """
    Clase para la clasificación de emociones usando un modelo pre-entrenado de resnet50
    """
    
    def __init__(self):
        EMOTION_MODEL_PATH = "./models/emotions_classificator.pth"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.emotion_classes = ["happy", "sad", "angry", "relaxed"]

        # Preprocesamiento de imagen
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)), 
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        # Cargar modelo
        self.model = ResNetClassifier(len(self.emotion_classes))
        self.model.load_state_dict(
            torch.load(EMOTION_MODEL_PATH, map_location=self.device)
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict_image_path(self, image_path, threshold=0.50):
        # Cargar imagen
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)

        # Predicción
        with torch.no_grad():
            output = torch.softmax(self.model(image), dim=1)
            confidence, idx = torch.max(output, dim=1)
            confidence = confidence.item()
            prediction = (
                self.emotion_classes[idx.item()]
                if confidence >= threshold
                else "Unknown"
            )
            return {"class": prediction, "confidence": f"{confidence * 100:.2f}%"}

    def predict_image(self, image, threshold=0.50):
        # Suponiendo que image es un numpy.ndarray cargado con OpenCV
        if isinstance(image, np.ndarray):
            # Convertir de numpy.ndarray a PIL.Image
            image = Image.fromarray(image)

        # Cargar imagen
        image = image.convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = torch.softmax(self.model(image), dim=1)
            confidence, idx = torch.max(output, dim=1)
            confidence = confidence.item()
            prediction = (
                self.emotion_classes[idx.item()]
                if confidence >= threshold
                else "Unknown"
            )
            return {"class": prediction, "confidence": f"{confidence * 100:.2f}%"}
