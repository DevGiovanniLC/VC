import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

# Definir una semilla para reproducibilidad
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------------
# Carga y Preparación de los Datasets
# -----------------------------------

class StanfordDogsDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
        self.labels = self._parse_annotations()

    def _parse_annotations(self):
        labels = []
        for img_path in self.image_paths:
            filename = os.path.basename(img_path).split('.')[0]
            annotation_file = os.path.join(self.annotations_dir, f"{filename}.xml")
            if not os.path.exists(annotation_file):
                continue  # Saltar si no hay anotación
            tree = ET.parse(annotation_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                breed = obj.find('name').text
                labels.append(breed)
        return labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_kaggle_dog_breeds(kaggle_path, transform=None):
    return datasets.ImageFolder(root=kaggle_path, transform=transform)

def combine_datasets(stanford_dataset, kaggle_dataset):
    combined_images = stanford_dataset.image_paths + [img_path for img_path, _ in kaggle_dataset.imgs]
    combined_labels = stanford_dataset.labels + [kaggle_dataset.classes[label] for _, label in kaggle_dataset.imgs]
    return combined_images, combined_labels

def load_intelligence_size_csv(csv_path, detected_breeds):
    df = pd.read_csv(csv_path)
    filtered_df = df[df['Breed'].isin(detected_breeds)]
    return filtered_df

def is_dangerous(breed, dangerous_breeds):
    return breed in dangerous_breeds

def load_color_dataset(color_path, transform=None):
    return datasets.ImageFolder(root=color_path, transform=transform)

# ------------------------
# Preprocesamiento de Datos
# ------------------------

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def balance_dataset(images, labels, seed=SEED):
    df = pd.DataFrame({'image': images, 'label': labels})
    min_count = df['label'].value_counts().min()
    balanced_df = df.groupby('label').apply(lambda x: x.sample(min_count, random_state=seed)).reset_index(drop=True)
    return balanced_df['image'].tolist(), balanced_df['label'].tolist()

def split_dataset(images, labels, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=SEED):
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, train_size=train_ratio, random_state=seed, stratify=labels)
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=val_size, random_state=seed, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, heights_low, heights_high, weights_low, weights_high, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.heights_low = heights_low
        self.heights_high = heights_high
        self.weights_low = weights_low
        self.weights_high = weights_high
        self.transform = transform
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.encoded_labels = [self.label_to_idx[label] for label in labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.encoded_labels[idx]
        height_low = self.heights_low[idx]
        height_high = self.heights_high[idx]
        weight_low = self.weights_low[idx]
        weight_high = self.weights_high[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, height_low, height_high, weight_low, weight_high

def integrate_additional_data(df_intelligence, dangerous_breeds, color_dataset):
    # Asumiendo que color_dataset es un ImageFolder donde las clases son los colores
    color_mapping = {breed: color for breed, color in zip(color_dataset.classes, color_dataset.classes)}  # Simplificación
    integrated_data = []
    for idx, row in df_intelligence.iterrows():
        breed = row['Breed']
        height_low = row['height_low_inches']
        height_high = row['height_high_inches']
        weight_low = row['weight_low_lbs']
        weight_high = row['weight_high_lbs']
        is_potentially_dangerous = is_dangerous(breed, dangerous_breeds)
        color = color_mapping.get(breed, 'Unknown')  # Extraer color real según tu lógica
        integrated_data.append({
            'Breed': breed,
            'height_low_inches': height_low,
            'height_high_inches': height_high,
            'weight_low_lbs': weight_low,
            'weight_high_lbs': weight_high,
            'Dangerous': is_potentially_dangerous,
            'Color': color
        })
    return pd.DataFrame(integrated_data)

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader

# ---------------------------------
# Entrenamiento y Evaluación del Modelo
# ---------------------------------

class MultiTaskResNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiTaskResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Eliminamos la capa fully connected original

        # Tarea 1: Clasificación de razas
        self.classifier = nn.Linear(num_features, num_classes)

        # Tarea 2: Regresión de altura (low y high)
        self.height_regressor_low = nn.Linear(num_features, 1)
        self.height_regressor_high = nn.Linear(num_features, 1)

        # Tarea 3: Regresión de peso (low y high)
        self.weight_regressor_low = nn.Linear(num_features, 1)
        self.weight_regressor_high = nn.Linear(num_features, 1)

    def forward(self, x):
        features = self.resnet(x)
        class_output = self.classifier(features)
        height_low = self.height_regressor_low(features).squeeze(1)
        height_high = self.height_regressor_high(features).squeeze(1)
        weight_low = self.weight_regressor_low(features).squeeze(1)
        weight_high = self.weight_regressor_high(features).squeeze(1)
        return class_output, height_low, height_high, weight_low, weight_high

def train_model(model, train_loader, val_loader, epochs=10, device='cuda'):
    model = model.to(device)
    # Definir pérdidas para cada tarea
    criterion_class = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels, h_low, h_high, w_low, w_high in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            h_low = h_low.float().to(device)
            h_high = h_high.float().to(device)
            w_low = w_low.float().to(device)
            w_high = w_high.float().to(device)

            optimizer.zero_grad()
            outputs_class, outputs_h_low, outputs_h_high, outputs_w_low, outputs_w_high = model(images)
            loss_class = criterion_class(outputs_class, labels)
            loss_h_low = criterion_reg(outputs_h_low, h_low)
            loss_h_high = criterion_reg(outputs_h_high, h_high)
            loss_w_low = criterion_reg(outputs_w_low, w_low)
            loss_w_high = criterion_reg(outputs_w_high, w_high)
            loss = loss_class + loss_h_low + loss_h_high + loss_w_low + loss_w_high
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validación
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels, h_low, h_high, w_low, w_high in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                h_low = h_low.float().to(device)
                h_high = h_high.float().to(device)
                w_low = w_low.float().to(device)
                w_high = w_high.float().to(device)

                outputs_class, outputs_h_low, outputs_h_high, outputs_w_low, outputs_w_high = model(images)
                loss_class = criterion_class(outputs_class, labels)
                loss_h_low = criterion_reg(outputs_h_low, h_low)
                loss_h_high = criterion_reg(outputs_h_high, h_high)
                loss_w_low = criterion_reg(outputs_w_low, w_low)
                loss_w_high = criterion_reg(outputs_w_high, w_high)
                loss = loss_class + loss_h_low + loss_h_high + loss_w_low + loss_w_high
                val_running_loss += loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses

def evaluate_model(model, test_loader, device='cuda'):
    model = model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels, _, _, _, _ in test_loader:
            images = images.to(device)
            outputs_class, _, _, _, _ = model(images)
            preds = torch.argmax(outputs_class, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)

    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classifier.out_features if hasattr(model.classifier, 'out_features') else "Classes",
                yticklabels=model.classifier.out_features if hasattr(model.classifier, 'out_features') else "Classes")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    print(classification_report(all_labels, all_preds))
    return report

# -------------------
# Integración Final
# -------------------

def main():
    # Definir rutas (actualiza estas rutas según tu estructura de carpetas)
    stanford_images_path = '/ruta/al/stanford/images'
    stanford_annotations_path = '/ruta/al/stanford/annotations'
    kaggle_dog_breeds_path = '/ruta/al/kaggle/9_dog_breeds'
    intelligence_size_csv = '/ruta/al/csv/intelligence_size.csv'
    color_path = '/ruta/al/colores'
    dangerous_breeds = ['Breed1', 'Breed2', 'Breed3']  # Reemplaza con las razas peligrosas reales

    # Cargar transformaciones
    transform = get_transforms()

    # Cargar datasets
    stanford_dataset = StanfordDogsDataset(stanford_images_path, stanford_annotations_path, transform=transform)
    kaggle_dataset = load_kaggle_dog_breeds(kaggle_dog_breeds_path, transform=transform)

    # Combinar datasets
    combined_images, combined_labels = combine_datasets(stanford_dataset, kaggle_dataset)

    # Balancear dataset
    balanced_images, balanced_labels = balance_dataset(combined_images, combined_labels, seed=SEED)

    # Cargar y filtrar CSV
    df_intelligence = load_intelligence_size_csv(intelligence_size_csv, detected_breeds=balanced_labels)

    # Cargar dataset de colores
    color_dataset = load_color_dataset(color_path, transform=transform)

    # Integrar datos adicionales
    df_integrated = integrate_additional_data(df_intelligence, dangerous_breeds, color_dataset)

    # Manejar casos donde algunas razas puedan no estar en el CSV después del filtrado
    df_final = df_integrated[df_integrated['Breed'].isin(balanced_labels)]

    # Dividir dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        df_final['image'].tolist(),
        df_final['label'].tolist(),
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=SEED
    )

    # Extraer columnas de altura y peso
    heights_low_train = df_final.loc[X_train.index, 'height_low_inches'].tolist()
    heights_high_train = df_final.loc[X_train.index, 'height_high_inches'].tolist()
    weights_low_train = df_final.loc[X_train.index, 'weight_low_lbs'].tolist()
    weights_high_train = df_final.loc[X_train.index, 'weight_high_lbs'].tolist()

    heights_low_val = df_final.loc[X_val.index, 'height_low_inches'].tolist()
    heights_high_val = df_final.loc[X_val.index, 'height_high_inches'].tolist()
    weights_low_val = df_final.loc[X_val.index, 'weight_low_lbs'].tolist()
    weights_high_val = df_final.loc[X_val.index, 'weight_high_lbs'].tolist()

    heights_low_test = df_final.loc[X_test.index, 'height_low_inches'].tolist()
    heights_high_test = df_final.loc[X_test.index, 'height_high_inches'].tolist()
    weights_low_test = df_final.loc[X_test.index, 'weight_low_lbs'].tolist()
    weights_high_test = df_final.loc[X_test.index, 'weight_high_lbs'].tolist()

    # Crear datasets personalizados
    train_dataset = CustomImageDataset(X_train, y_train, heights_low_train, heights_high_train, weights_low_train, weights_high_train, transform=transform)
    val_dataset = CustomImageDataset(X_val, y_val, heights_low_val, heights_high_val, weights_low_val, weights_high_val, transform=transform)
    test_dataset = CustomImageDataset(X_test, y_test, heights_low_test, heights_high_test, weights_low_test, weights_high_test, transform=transform)

    # Crear DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32)

    # Inicializar y entrenar el modelo multitarea
    num_classes = len(train_dataset.label_to_idx)
    model = MultiTaskResNet(num_classes=num_classes)
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Evaluar el modelo
    report = evaluate_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Graficar pérdidas
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # Guardar el modelo
    torch.save(model.state_dict(), 'resnet50_dog_classifier_multitask.pth')

if __name__ == '__main__':
    main()
