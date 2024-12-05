import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------------
# Data Loading and Preparation
# -----------------------------------

class StanfordDogsDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(images_dir, '*', '*.jpg'))
        self.labels = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        self.data = []
        for img_path in self.image_paths:
            # Extract breed from the image path
            breed = os.path.basename(os.path.dirname(img_path))
            image_name = os.path.basename(img_path).split('.')[0]
            annotation_file = os.path.join(self.annotations_dir, breed, image_name)
            if not os.path.exists(annotation_file):
                continue  # Skip if annotation does not exist
            self.data.append((img_path, breed))
            self.labels.append(breed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def load_nine_breeds_dog_breeds(nine_breeds_path, transform=None):
    return datasets.ImageFolder(root=nine_breeds_path, transform=transform)

def create_annotation_file(image_path, breed_name, annotation_dir):
    """
    Creates an XML annotation file compatible with the Stanford Dogs format.
    """
    # Get image information
    img = Image.open(image_path)
    width, height = img.size
    depth = len(img.getbands())  # RGB = 3 channels

    # Create XML structure
    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder')
    folder.text = breed_name

    filename = ET.SubElement(annotation, 'filename')
    filename.text = os.path.basename(image_path)

    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = '9Breeds Dataset'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)

    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'

    obj = ET.SubElement(annotation, 'object')
    name = ET.SubElement(obj, 'name')
    name.text = breed_name

    pose = ET.SubElement(obj, 'pose')
    pose.text = 'Unspecified'
    truncated = ET.SubElement(obj, 'truncated')
    truncated.text = '0'
    difficult = ET.SubElement(obj, 'difficult')
    difficult.text = '0'

    bndbox = ET.SubElement(obj, 'bndbox')
    ET.SubElement(bndbox, 'xmin').text = '0'  # Placeholder coordinates
    ET.SubElement(bndbox, 'ymin').text = '0'
    ET.SubElement(bndbox, 'xmax').text = str(width)
    ET.SubElement(bndbox, 'ymax').text = str(height)

    # Save XML file
    breed_dir = os.path.join(annotation_dir, breed_name)
    os.makedirs(breed_dir, exist_ok=True)
    image_name = os.path.basename(image_path).split('.')[0]
    annotation_path = os.path.join(breed_dir, image_name)
    tree = ET.ElementTree(annotation)
    tree.write(annotation_path, encoding='utf-8', xml_declaration=True)

def load_intelligence_size_csv(csv_path, detected_breeds):
    df = pd.read_csv(csv_path)
    df['Breed'] = df['Breed'].str.lower().str.replace('-', '_').str.replace(' ', '_')
    detected_breeds = [breed.lower().replace('-', '_').replace(' ', '_') for breed in detected_breeds]
    filtered_df = df[df['Breed'].isin(detected_breeds)]
    return filtered_df

def is_dangerous(breed, dangerous_breeds):
    return breed.lower() in [b.lower() for b in dangerous_breeds]

# ------------------------
# Data Preprocessing
# ------------------------

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        # Remove ColorJitter to preserve color and emotion cues
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def split_dataset(df, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=SEED):
    train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=seed)
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(temp_df, train_size=val_size, random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

        # Create label mappings
        self.breed_set = sorted(set(self.dataframe['Breed'].dropna()))
        self.breed_to_idx = {breed: idx for idx, breed in enumerate(self.breed_set)}
        self.idx_to_breed = {idx: breed for breed, idx in self.breed_to_idx.items()}

        # For colors, collect all unique colors from lists
        all_colors = set()
        for colors in self.dataframe['Color'].dropna():
            if colors is not None:
                all_colors.update(colors)
        self.color_set = sorted(all_colors)
        self.color_to_idx = {color: idx for idx, color in enumerate(self.color_set)}
        self.idx_to_color = {idx: color for color, idx in self.color_to_idx.items()}

        self.emotion_set = sorted(set(self.dataframe['Emotion'].dropna()))
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotion_set)}
        self.idx_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_idx.items()}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image']
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Get labels if available, else set to None
        breed_label = self.breed_to_idx.get(row['Breed']) if pd.notna(row['Breed']) else None

        # Encode colors as multi-hot vector
        if pd.notna(row['Color']) and row['Color'] is not None:
            color_indices = [self.color_to_idx[color] for color in row['Color']]
            color_label = torch.zeros(len(self.color_set), dtype=torch.float32)
            color_label[color_indices] = 1.0
        else:
            color_label = None

        emotion_label = self.emotion_to_idx.get(row['Emotion']) if pd.notna(row['Emotion']) else None

        # Heights and weights
        height_low = row['height_low_inches'] if pd.notna(row['height_low_inches']) else None
        height_high = row['height_high_inches'] if pd.notna(row['height_high_inches']) else None
        weight_low = row['weight_low_lbs'] if pd.notna(row['weight_low_lbs']) else None
        weight_high = row['weight_high_lbs'] if pd.notna(row['weight_high_lbs']) else None

        sample = {
            'image': image,
            'breed_label': breed_label,
            'color_label': color_label,
            'emotion_label': emotion_label,
            'height_low': torch.tensor(height_low, dtype=torch.float32) if height_low is not None else None,
            'height_high': torch.tensor(height_high, dtype=torch.float32) if height_high is not None else None,
            'weight_low': torch.tensor(weight_low, dtype=torch.float32) if weight_low is not None else None,
            'weight_high': torch.tensor(weight_high, dtype=torch.float32) if weight_high is not None else None
        }
        return sample

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader

def collate_fn(batch):
    # Custom collate function to handle samples with missing labels
    images = []
    breed_labels = []
    color_labels = []
    emotion_labels = []
    height_lows = []
    height_highs = []
    weight_lows = []
    weight_highs = []

    for sample in batch:
        images.append(sample['image'])
        breed_labels.append(sample['breed_label'])
        color_labels.append(sample['color_label'])
        emotion_labels.append(sample['emotion_label'])
        height_lows.append(sample['height_low'])
        height_highs.append(sample['height_high'])
        weight_lows.append(sample['weight_low'])
        weight_highs.append(sample['weight_high'])

    images = torch.stack(images)

    # For color labels, stack the multi-hot vectors
    if any(lbl is not None for lbl in color_labels):
        color_labels_tensor = torch.stack([lbl if lbl is not None else torch.zeros(len(color_labels[0])) for lbl in color_labels])
    else:
        color_labels_tensor = None

    return {
        'images': images,
        'breed_labels': torch.tensor([lbl if lbl is not None else -1 for lbl in breed_labels], dtype=torch.long),
        'color_labels': color_labels_tensor,
        'emotion_labels': torch.tensor([lbl if lbl is not None else -1 for lbl in emotion_labels], dtype=torch.long),
        'height_lows': torch.stack([hl if hl is not None else torch.tensor(0.0) for hl in height_lows]),
        'height_highs': torch.stack([hh if hh is not None else torch.tensor(0.0) for hh in height_highs]),
        'weight_lows': torch.stack([wl if wl is not None else torch.tensor(0.0) for wl in weight_lows]),
        'weight_highs': torch.stack([wh if wh is not None else torch.tensor(0.0) for wh in weight_highs]),
        'breed_mask': torch.tensor([lbl is not None for lbl in breed_labels], dtype=torch.bool),
        'color_mask': torch.tensor([lbl is not None for lbl in color_labels], dtype=torch.bool),
        'emotion_mask': torch.tensor([lbl is not None for lbl in emotion_labels], dtype=torch.bool),
        'height_mask': torch.tensor([hl is not None for hl in height_lows], dtype=torch.bool),
        'weight_mask': torch.tensor([wl is not None for wl in weight_lows], dtype=torch.bool),
    }

# ---------------------------
# Model Definition and Training
# ---------------------------

class MultiTaskResNet(nn.Module):
    def __init__(self, num_breeds, num_colors, num_emotions):
        super(MultiTaskResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the original fully connected layer

        # Task-specific heads
        self.breed_classifier = nn.Linear(num_features, num_breeds)
        self.color_classifier = nn.Linear(num_features, num_colors)  # Output logits for each color class
        self.emotion_classifier = nn.Linear(num_features, num_emotions)
        self.height_regressor_low = nn.Linear(num_features, 1)
        self.height_regressor_high = nn.Linear(num_features, 1)
        self.weight_regressor_low = nn.Linear(num_features, 1)
        self.weight_regressor_high = nn.Linear(num_features, 1)

    def forward(self, x):
        features = self.resnet(x)
        outputs = {
            'breed': self.breed_classifier(features),
            'color': self.color_classifier(features),  # No activation here; we'll use BCEWithLogitsLoss
            'emotion': self.emotion_classifier(features),
            'height_low': self.height_regressor_low(features).squeeze(1),
            'height_high': self.height_regressor_high(features).squeeze(1),
            'weight_low': self.weight_regressor_low(features).squeeze(1),
            'weight_high': self.weight_regressor_high(features).squeeze(1),
        }
        return outputs

def train_model(model, train_loader, val_loader, datasets_info, epochs=10, device='cuda'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_class = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_multilabel = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch['images'].to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss = 0.0
            # Breed classification loss
            if batch['breed_mask'].any():
                breed_labels = batch['breed_labels'].to(device)
                breed_outputs = outputs['breed']
                loss_breed = criterion_class(breed_outputs, breed_labels)
                loss += loss_breed

            # Color classification loss (multi-label)
            if batch['color_mask'].any():
                color_labels = batch['color_labels'][batch['color_mask']].to(device)
                color_outputs = outputs['color'][batch['color_mask']]
                loss_color = criterion_multilabel(color_outputs, color_labels)
                loss += loss_color

            # Emotion classification loss
            if batch['emotion_mask'].any():
                emotion_labels = batch['emotion_labels'][batch['emotion_mask']].to(device)
                emotion_outputs = outputs['emotion'][batch['emotion_mask']]
                loss_emotion = criterion_class(emotion_outputs, emotion_labels)
                loss += loss_emotion

            # Height regression loss
            if batch['height_mask'].any():
                height_mask = batch['height_mask'].to(device)
                height_low = batch['height_lows'][height_mask].to(device)
                height_high = batch['height_highs'][height_mask].to(device)
                height_low_outputs = outputs['height_low'][height_mask]
                height_high_outputs = outputs['height_high'][height_mask]
                loss_height_low = criterion_reg(height_low_outputs, height_low)
                loss_height_high = criterion_reg(height_high_outputs, height_high)
                loss += loss_height_low + loss_height_high

            # Weight regression loss
            if batch['weight_mask'].any():
                weight_mask = batch['weight_mask'].to(device)
                weight_low = batch['weight_lows'][weight_mask].to(device)
                weight_high = batch['weight_highs'][weight_mask].to(device)
                weight_low_outputs = outputs['weight_low'][weight_mask]
                weight_high_outputs = outputs['weight_high'][weight_mask]
                loss_weight_low = criterion_reg(weight_low_outputs, weight_low)
                loss_weight_high = criterion_reg(weight_high_outputs, weight_high)
                loss += loss_weight_low + loss_weight_high

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                outputs = model(images)

                loss = 0.0
                # Breed classification loss
                if batch['breed_mask'].any():
                    breed_labels = batch['breed_labels'].to(device)
                    breed_outputs = outputs['breed']
                    loss_breed = criterion_class(breed_outputs, breed_labels)
                    loss += loss_breed

                # Color classification loss (multi-label)
                if batch['color_mask'].any():
                    color_labels = batch['color_labels'][batch['color_mask']].to(device)
                    color_outputs = outputs['color'][batch['color_mask']]
                    loss_color = criterion_multilabel(color_outputs, color_labels)
                    loss += loss_color

                # Emotion classification loss
                if batch['emotion_mask'].any():
                    emotion_labels = batch['emotion_labels'][batch['emotion_mask']].to(device)
                    emotion_outputs = outputs['emotion'][batch['emotion_mask']]
                    loss_emotion = criterion_class(emotion_outputs, emotion_labels)
                    loss += loss_emotion

                # Height regression loss
                if batch['height_mask'].any():
                    height_mask = batch['height_mask'].to(device)
                    height_low = batch['height_lows'][height_mask].to(device)
                    height_high = batch['height_highs'][height_mask].to(device)
                    height_low_outputs = outputs['height_low'][height_mask]
                    height_high_outputs = outputs['height_high'][height_mask]
                    loss_height_low = criterion_reg(height_low_outputs, height_low)
                    loss_height_high = criterion_reg(height_high_outputs, height_high)
                    loss += loss_height_low + loss_height_high

                # Weight regression loss
                if batch['weight_mask'].any():
                    weight_mask = batch['weight_mask'].to(device)
                    weight_low = batch['weight_lows'][weight_mask].to(device)
                    weight_high = batch['weight_highs'][weight_mask].to(device)
                    weight_low_outputs = outputs['weight_low'][weight_mask]
                    weight_high_outputs = outputs['weight_high'][weight_mask]
                    loss_weight_low = criterion_reg(weight_low_outputs, weight_low)
                    loss_weight_high = criterion_reg(weight_high_outputs, weight_high)
                    loss += loss_weight_low + loss_weight_high

                val_running_loss += loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses

def evaluate_model(model, test_loader, datasets_info, device='cuda'):
    model.to(device)
    model.eval()
    # Initialize accumulators for predictions and labels
    breed_preds, breed_labels = [], []
    color_preds_list, color_labels_list = [], []
    emotion_preds, emotion_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['images'].to(device)
            outputs = model(images)

            # Breed classification evaluation
            if batch['breed_mask'].any():
                breed_output = outputs['breed']
                preds = torch.argmax(breed_output, dim=1)
                mask = batch['breed_mask']
                breed_preds.extend(preds[mask].cpu().numpy())
                breed_labels.extend(batch['breed_labels'][mask].cpu().numpy())

            # Color classification evaluation (multi-label)
            if batch['color_mask'].any():
                color_output = outputs['color'][batch['color_mask']]
                color_probs = torch.sigmoid(color_output)
                # Apply threshold (e.g., 0.5) to get predicted labels
                preds = (color_probs > 0.5).cpu().numpy().astype(int)
                color_preds_list.append(preds)
                color_labels_list.append(batch['color_labels'][batch['color_mask']].cpu().numpy())

            # Emotion classification evaluation
            if batch['emotion_mask'].any():
                emotion_output = outputs['emotion']
                preds = torch.argmax(emotion_output, dim=1)
                mask = batch['emotion_mask']
                emotion_preds.extend(preds[mask].cpu().numpy())
                emotion_labels.extend(batch['emotion_labels'][mask].cpu().numpy())

    # Generate reports for each task
    if breed_labels:
        breed_class_names = datasets_info['breed_class_names']
        print("Breed Classification Report:")
        print(classification_report(breed_labels, breed_preds, target_names=breed_class_names))

    if color_labels_list:
        color_class_names = datasets_info['color_class_names']
        # Concatenate all batches
        color_labels = np.vstack(color_labels_list)
        color_preds = np.vstack(color_preds_list)
        print("Color Multi-label Classification Report:")
        from sklearn.metrics import classification_report
        print(classification_report(color_labels, color_preds, target_names=color_class_names, zero_division=0))

    if emotion_labels:
        emotion_class_names = datasets_info['emotion_class_names']
        print("Emotion Classification Report:")
        print(classification_report(emotion_labels, emotion_preds, target_names=emotion_class_names))

# -------------------
# Final Integration
# -------------------

def build_image_color_mapping(color_dataset_path):
    image_color_mapping = {}
    for color in os.listdir(color_dataset_path):
        color_dir = os.path.join(color_dataset_path, color)
        if os.path.isdir(color_dir):
            for img_file in os.listdir(color_dir):
                img_path = os.path.join(color_dir, img_file)
                img_path = os.path.normpath(img_path)  # Normalize the path
                if img_path in image_color_mapping:
                    image_color_mapping[img_path].add(color)
                else:
                    image_color_mapping[img_path] = {color}
    return image_color_mapping

def main():
    # Define paths (update these paths according to your folder structure)
    stanford_images_path = './datasets/stanford_breeds/Images'
    stanford_annotations_path = './datasets/stanford_breeds/Annotation'
    intelligence_size_csv = './datasets/AKC_Breed_Info.csv'
    color_path = './datasets/dog_colors'
    emotion_path = './datasets/dogs_emotions'
    dangerous_breeds = ['Pit Bull', 'Rottweiler', 'German Shepherd', 'Doberman Pinscher', 'Chow Chow',
                        'Presa Canario', 'Akita', 'Alaskan Malamute', 'Siberian Husky', 'Wolf Hybrid']

    # Create annotations for the 9 breeds dataset
    nine_breeds_dog_breeds_path = './datasets/9Breeds'
    for breed in os.listdir(nine_breeds_dog_breeds_path):
        breed_dir = os.path.join(nine_breeds_dog_breeds_path, breed)
        if os.path.isdir(breed_dir):
            for img_file in os.listdir(breed_dir):
                if img_file.endswith('.jpg'):
                    image_path = os.path.join(breed_dir, img_file)
                    create_annotation_file(image_path, breed, stanford_annotations_path)

    # Load transformations
    transform = get_transforms()

    # Load datasets
    stanford_dataset = StanfordDogsDataset(stanford_images_path, stanford_annotations_path, transform=transform)
    nine_breeds_dataset = load_nine_breeds_dog_breeds(nine_breeds_dog_breeds_path, transform=transform)
    emotion_dataset = datasets.ImageFolder(root=emotion_path)

    # Prepare dataframes
    # Images with breed labels
    breed_images = [img_path for img_path, _ in stanford_dataset.data] + [path for path, _ in nine_breeds_dataset.samples]
    breed_labels = stanford_dataset.labels + [nine_breeds_dataset.classes[label] for _, label in nine_breeds_dataset.samples]
    breed_images = [os.path.normpath(path) for path in breed_images]  # Normalize paths
    df_breed = pd.DataFrame({'image': breed_images, 'Breed': breed_labels})

    # Images with emotion labels
    emotion_images = [path for path, _ in emotion_dataset.samples]
    emotion_labels = [emotion_dataset.classes[label] for _, label in emotion_dataset.samples]
    emotion_images = [os.path.normpath(path) for path in emotion_images]  # Normalize paths
    df_emotion = pd.DataFrame({'image': emotion_images, 'Emotion': emotion_labels})

    # Combine dataframes
    df_combined = pd.concat([df_breed, df_emotion], ignore_index=True, sort=False)
    df_combined['height_low_inches'] = None
    df_combined['height_high_inches'] = None
    df_combined['weight_low_lbs'] = None
    df_combined['weight_high_lbs'] = None

    # Build the image-to-colors mapping
    color_mapping = build_image_color_mapping(color_path)

    # Assign colors to images in df_combined
    color_labels = []
    for img_path in df_combined['image']:
        img_path = os.path.normpath(img_path)
        colors = color_mapping.get(img_path, set())
        if colors:
            color_labels.append(list(colors))
        else:
            color_labels.append(None)  # No color labels available
    df_combined['Color'] = color_labels

    # Load intelligence and size data
    detected_breeds = df_breed['Breed'].unique()
    df_intelligence = load_intelligence_size_csv(intelligence_size_csv, detected_breeds=detected_breeds)

    # Merge intelligence data
    df_combined = pd.merge(df_combined, df_intelligence, on='Breed', how='left')

    # Handle dangerous breeds
    df_combined['Dangerous'] = df_combined['Breed'].apply(lambda x: is_dangerous(x, dangerous_breeds) if pd.notna(x) else False)

    # Split dataset
    train_df, val_df, test_df = split_dataset(df_combined, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=SEED)

    # Create datasets
    train_dataset = CustomImageDataset(train_df, transform=transform)
    val_dataset = CustomImageDataset(val_df, transform=transform)
    test_dataset = CustomImageDataset(test_df, transform=transform)

    # Gather dataset info
    datasets_info = {
        'breed_class_names': [train_dataset.idx_to_breed[idx] for idx in range(len(train_dataset.breed_set))],
        'color_class_names': [train_dataset.idx_to_color[idx] for idx in range(len(train_dataset.color_set))],
        'emotion_class_names': [train_dataset.idx_to_emotion[idx] for idx in range(len(train_dataset.emotion_set))],
    }

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32)

    # Initialize and train the multitask model
    num_breeds = len(train_dataset.breed_set)
    num_colors = len(train_dataset.color_set)
    num_emotions = len(train_dataset.emotion_set)
    model = MultiTaskResNet(num_breeds=num_breeds, num_colors=num_colors, num_emotions=num_emotions)
    train_losses, val_losses = train_model(model, train_loader, val_loader, datasets_info, epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Evaluate the model
    evaluate_model(model, test_loader, datasets_info, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Plot losses
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # Save the model
    torch.save(model.state_dict(), 'resnet50_dog_classifier_multitask.pth')

if __name__ == '__main__':
    main()
