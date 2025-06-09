import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from PIL import Image
import random
from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm
import pandas as pd
from torchvision import transforms
import os

from torchvision import transforms

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class TripletNet(nn.Module):
    def __init__(self, base_model_name='vit_base_patch16_224', embedding_dim=128):
        super(TripletNet, self).__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=True, num_classes=0)
        in_features = self.base_model.num_features
        self.embedding = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        features = self.base_model(x)
        embeddings = self.embedding(features)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)  # L2 нормализация
        return embeddings

def calculate_triplet_metrics(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for anchor, positive, negative in tqdm(test_loader, desc="Testing Triplets"):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            pos_dist = torch.norm(anchor_emb - positive_emb, p=2, dim=1)
            neg_dist = torch.norm(anchor_emb - negative_emb, p=2, dim=1)
            
            preds = (pos_dist < neg_dist).cpu().numpy()
            labels = np.ones_like(preds)  # все триплеты корректны
            
            all_labels.extend(labels)
            all_preds.extend(preds)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }


def calculate_image_metrics(model, test_loader, device, df_test):
    """Вычисляем Recall@k для отдельных изображений из тестового датасета"""
    model.eval()
    embeddings = []
    true_labels = []
    
    # Словарь path -> индивидуальный id (у тебя в df_test должен быть такой столбец)
    path_to_label = {row['path']: row['individual_id'] for _, row in df_test.iterrows()}
    
    with torch.no_grad():
        idx = 0
        for anchor, _, _ in tqdm(test_loader, desc="Processing Images for Recall@k"):
            anchor = anchor.to(device)
            emb = model(anchor)
            embeddings.append(emb.cpu().numpy())
            
            batch_size = anchor.size(0)
            batch_paths = df_test.iloc[idx:idx+batch_size]['path'].values
            batch_labels = [path_to_label[p] for p in batch_paths]
            true_labels.extend(batch_labels)
            idx += batch_size
    
    embeddings = np.concatenate(embeddings)
    true_labels = np.array(true_labels)
    
    dist_matrix = pairwise_distances(embeddings)
    
    k = 5
    recall_count = 0
    
    for i in range(len(dist_matrix)):
        nearest_indices = np.argsort(dist_matrix[i])[1:k+1]
        nearest_labels = true_labels[nearest_indices]
        if true_labels[i] in nearest_labels:
            recall_count += 1
    
    recall_at_k = recall_count / len(dist_matrix)
    
    return {
        'recall_at_5': recall_at_k
    }

class TripletDataset_test(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

        self.class_to_indices = {}
        for idx, row in self.df.iterrows():
            class_id = row['class_id']
            if class_id not in self.class_to_indices:
                self.class_to_indices[class_id] = []
            self.class_to_indices[class_id].append(idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        anchor_row = self.df.iloc[index]
        anchor_image_path = anchor_row['path']
        anchor_class = anchor_row['class_id']

        # Загружаем якорь
        anchor_img = Image.open(anchor_image_path).convert('RGB')
        if self.transform:
            anchor_img = self.transform(anchor_img)

        # Позитивный пример из того же класса, кроме текущего
        positive_idx = index
        while positive_idx == index:
            positive_idx = random.choice(self.class_to_indices[anchor_class])
        positive_row = self.df.iloc[positive_idx]
        positive_img = Image.open(positive_row['path']).convert('RGB')
        if self.transform:
            positive_img = self.transform(positive_img)

        # Негативный пример из другого класса
        negative_class = anchor_class
        while negative_class == anchor_class:
            negative_class = random.choice(list(self.class_to_indices.keys()))
        negative_idx = random.choice(self.class_to_indices[negative_class])
        negative_row = self.df.iloc[negative_idx]
        negative_img = Image.open(negative_row['path']).convert('RGB')
        if self.transform:
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img


def test_triplet_model(model, test_loader, device, margin=1.0):
    model.eval()
    criterion = nn.TripletMarginLoss(margin=margin, p=2)

    total_loss = 0.0
    total_samples = 0

    total_positive_distance = 0.0
    total_negative_distance = 0.0
    total_correct = 0

    with torch.no_grad():
        for anchor, positive, negative in test_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            embed_anchor = model(anchor)
            embed_positive = model(positive)
            embed_negative = model(negative)

            loss = criterion(embed_anchor, embed_positive, embed_negative)
            total_loss += loss.item() * anchor.size(0)
            total_samples += anchor.size(0)

            # Расстояния Евклида
            pos_dist = torch.norm(embed_anchor - embed_positive, p=2, dim=1)
            neg_dist = torch.norm(embed_anchor - embed_negative, p=2, dim=1)

            total_positive_distance += pos_dist.sum().item()
            total_negative_distance += neg_dist.sum().item()

            # Accuracy: считаем, сколько триплетов проходят условие triplet loss margin
            correct = (neg_dist - pos_dist > margin).sum().item()
            total_correct += correct

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_pos_dist = total_positive_distance / total_samples if total_samples > 0 else 0
    avg_neg_dist = total_negative_distance / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    avg_margin = avg_neg_dist - avg_pos_dist

    return {
        'triplet_loss': avg_loss,
        'triplet_accuracy': accuracy,
        'avg_positive_distance': avg_pos_dist,
        'avg_negative_distance': avg_neg_dist,
        'avg_triplet_margin': avg_margin
    }


# ======= Остальной код для загрузки данных, тренировок и тестов =======

# Загрузки данных, трансформации, DataLoader — оставляем как есть

# Пример вызова теста:

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


df_test = pd.read_csv("C:/klasss/archive/newts/model_final/labels_test.csv")

test_dataset = TripletDataset_test(df_test, transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TripletNet().to(device)
model.load_state_dict(torch.load("C:/klasss/archive/newts/model_final/vit_model.pth", map_location=device))
model.eval()

# Вызов твоей функции теста, которую ты написал (триплетные метрики с loss)
test_metrics = test_triplet_model(model, test_loader, device, margin=1.0)

# Новые метрики (accuracy, precision, recall для триплетов)
triplet_classification_metrics = calculate_triplet_metrics(model, test_loader, device)

# Recall@5 для поиска по эмбеддингам
image_retrieval_metrics = calculate_image_metrics(model, test_loader, device, df_test)

print("\n=== Triplet Model Test Results ===")
print(f"Triplet Loss: {test_metrics['triplet_loss']:.4f}")
print(f"Triplet Accuracy (from loss fn): {test_metrics['triplet_accuracy']:.4f}")
print(f"Avg Positive Distance: {test_metrics['avg_positive_distance']:.4f}")
print(f"Avg Negative Distance: {test_metrics['avg_negative_distance']:.4f}")
print(f"Avg Triplet Margin: {test_metrics['avg_triplet_margin']:.4f}")

print("\n=== Additional Triplet Classification Metrics ===")
print(f"Accuracy: {triplet_classification_metrics['accuracy']:.4f}")
print(f"Precision: {triplet_classification_metrics['precision']:.4f}")
print(f"Recall: {triplet_classification_metrics['recall']:.4f}")

print("\n=== Image Retrieval Metrics ===")
print(f"Recall@5: {image_retrieval_metrics['recall_at_5']:.4f}")









'''
model = TripletNet().to(device)
model.load_state_dict(torch.load("C:/klasss/archive/newts/model_final/vit_model.pth", map_location=device))
model.eval()

correct_cls_total = 0
total_samples = 0

predictions = []

with torch.no_grad():
    for images, class_id, individual_id in test_loader:
        images = images.to(device)
        labels = class_id.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        # Подсчет точности по классам
        correct_cls_total += (preds == labels).sum().item()
        total_samples += labels.size(0)

        # Сохраняем предсказания и person_id для анализа
        for pred, true_label, pid in zip(preds.cpu().numpy(), class_id.cpu().numpy(), individual_id):
            predictions.append({
                'person_id': pid,
                'predicted_class': pred,
                'true_class': true_label
            })

accuracy = correct_cls_total / total_samples * 100
print(f'Accuracy по классам: {accuracy:.2f}%')
'''