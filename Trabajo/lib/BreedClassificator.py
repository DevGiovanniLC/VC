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
            for param in self.resnet.layer1.parameters():
                param.requires_grad = False
            for param in self.resnet.layer2.parameters():
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

class BreedClassificator:
    """
    Clase para la detección de una raza de perro a partir de una imagen
    """
    def __init__(self):

        BREED_MODEL_PATH = "./models/breed_classificator.pth"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.breed_classes = [
    'chihuahua', 'japanese_spaniel', 'maltese_dog', 'pekinese', 'shih_tzu',
    'blenheim_spaniel', 'papillon', 'toy_terrier', 'rhodesian_ridgeback',
    'afghan_hound', 'basset', 'beagle', 'bloodhound', 'bluetick',
    'black_and_tan_coonhound', 'walker_hound', 'english_foxhound', 'redbone',
    'borzoi', 'irish_wolfhound', 'italian_greyhound', 'whippet', 'ibizan_hound',
    'norwegian_elkhound', 'otterhound', 'saluki', 'scottish_deerhound',
    'weimaraner', 'staffordshire_bullterrier', 'american_staffordshire_terrier',
    'bedlington_terrier', 'border_terrier', 'kerry_blue_terrier',
    'irish_terrier', 'norfolk_terrier', 'norwich_terrier', 'yorkshire_terrier',
    'wire_haired_fox_terrier', 'lakeland_terrier', 'sealyham_terrier',
    'airedale', 'cairn', 'australian_terrier', 'dandie_dinmont', 'boston_bull',
    'miniature_schnauzer', 'giant_schnauzer', 'standard_schnauzer',
    'scotch_terrier', 'tibetan_terrier', 'silky_terrier',
    'soft_coated_wheaten_terrier', 'west_highland_white_terrier', 'lhasa',
    'flat_coated_retriever', 'curly_coated_retriever', 'golden_retriever',
    'labrador_retriever', 'chesapeake_bay_retriever',
    'german_short_haired_pointer', 'vizsla', 'english_setter', 'irish_setter',
    'gordon_setter', 'brittany_spaniel', 'clumber', 'english_springer',
    'welsh_springer_spaniel', 'cocker_spaniel', 'sussex_spaniel',
    'irish_water_spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois',
    'briard', 'kelpie', 'komondor', 'old_english_sheepdog', 'shetland_sheepdog',
    'collie', 'border_collie', 'bouvier_des_flandres', 'rottweiler',
    'german_shepherd', 'doberman', 'miniature_pinscher',
    'greater_swiss_mountain_dog', 'bernese_mountain_dog', 'appenzeller',
    'entlebucher', 'boxer', 'bull_mastiff', 'tibetan_mastiff', 'french_bulldog',
    'great_dane', 'saint_bernard', 'eskimo_dog', 'malamute', 'siberian_husky',
    'affenpinscher', 'basenji', 'pug', 'leonberg', 'newfoundland',
    'great_pyrenees', 'samoyed', 'pomeranian', 'chow', 'keeshond',
    'brabancon_griffon', 'pembroke', 'cardigan', 'toy_poodle', 'miniature_poodle',
    'standard_poodle', 'mexican_hairless', 'dingo', 'dhole',
    'african_hunting_dog', 'akita_inu', 'bull_dog', 'dachshund',
    'golden_retreiver', 'poodle'
]

        # Preprocesamiento de imagen
        self.transform =transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

        self.model = ResNetClassifier(len(self.breed_classes))
        self.model.load_state_dict(
            torch.load(BREED_MODEL_PATH, map_location=self.device), strict=True
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
                self.breed_classes[idx.item()] if confidence >= threshold else "Unknown"
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
                self.breed_classes[idx.item()] if confidence >= threshold else "Unknown"
            )
            return {"class": prediction, "confidence": f"{confidence * 100:.2f}%"}
        

import torch

BREED_MODEL_PATH = "./models/breed_classificator.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar state_dict sin asignarlo al modelo
state_dict = torch.load(BREED_MODEL_PATH, map_location=device)

# Verificar qué claves contiene el state_dict
print("Claves en el state_dict:", list(state_dict.keys()))

# Verificar qué claves espera el modelo actual
model = ResNetClassifier(num_classes=125)  # Asegúrate de usar el número correcto de clases
model_dict = model.state_dict()
print("Claves en el modelo actual:", list(model_dict.keys()))



