import torch
import torch.nn as nn
from torch import optim
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import random
import timm
import os



# Определение модели
class TripletNet(nn.Module):
    def __init__(self, base_model_name='vit_base_patch16_224', embedding_dim=128):
        super(TripletNet, self).__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=True)
        in_features = self.base_model.head.in_features
        self.base_model.head = nn.Identity()
        self.embedding = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        features = self.base_model(x)
        embeddings = self.embedding(features)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings



# Dataset и DataLoader

class TripletDataset(Dataset):
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

    def __getitem__(self, idx):
        anchor_row = self.df.iloc[idx]  
        anchor_image = Image.open(anchor_row['path']).convert('RGB')

        # Выбираем позитивный пример
        positive_indices = [i for i in self.class_to_indices[anchor_row['class_id']] if i != idx]
        positive_idx = random.choice(positive_indices)
        positive_image = Image.open(self.df.iloc[positive_idx]['path']).convert('RGB')

        # Выбираем негативный пример
        negative_classes = [c for c in self.class_to_indices.keys() if
                            c != anchor_row['class_id']]
        negative_class = random.choice(negative_classes)
        negative_idx = random.choice(self.class_to_indices[negative_class])
        negative_image = Image.open(self.df.iloc[negative_idx]['path']).convert('RGB')

        if self.transform:
            anchor_image = self.transform(anchor_image)  # Убрана лишняя точка
            positive_image = self.transform(positive_image)  # Убрана лишняя точка
            negative_image = self.transform(negative_image)  # Убрана лишняя точка

        return anchor_image, positive_image, negative_image, anchor_row['class_id']


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, row['class_id']



# Функция тестирования

def evaluate_model(model, dataloader, device, epoch=None, mode='test'):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluation {mode} epoch {epoch}" if epoch else f"Evaluation {mode}"):
            if mode == 'test':
                images, class_ids = batch
                images = images.to(device)
                embed = model(images)
            else:
                anchor_imgs, _, _, class_ids = batch
                images = anchor_imgs.to(device)
                embed = model(images)

            embeddings.append(embed.cpu().numpy())
            labels.extend(class_ids.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    # Вычисление метрик
    dist_matrix = pairwise_distances(embeddings, metric='euclidean')

    avg_pos_dist = []
    avg_neg_dist = []
    precision_at_1 = 0
    precision_at_5 = 0
    total = 0

    for i in range(len(labels)):
        pos_mask = labels == labels[i]
        pos_mask[i] = False
        if np.any(pos_mask):
            avg_pos_dist.append(np.mean(dist_matrix[i, pos_mask]))

        neg_mask = labels != labels[i]
        if np.any(neg_mask):
            avg_neg_dist.append(np.mean(dist_matrix[i, neg_mask]))

        if i < len(labels) - 1:
            sorted_indices = np.argsort(dist_matrix[i, :])
            sorted_indices = sorted_indices[sorted_indices != i]

            # Precision@1
            nearest = sorted_indices[0]
            precision_at_1 += (labels[nearest] == labels[i])

            # Precision@5
            top5 = sorted_indices[:5]
            precision_at_5 += np.sum(labels[top5] == labels[i]) / 5

            total += 1

    metrics = {
        'avg_pos_distance': np.mean(avg_pos_dist),
        'avg_neg_distance': np.mean(avg_neg_dist),
        'pos_neg_ratio': np.mean(avg_pos_dist) / np.mean(avg_neg_dist),
        'precision@1': precision_at_1 / total,
        'precision@5': precision_at_5 / total,
        'num_samples': len(labels)
    }

    return metrics


# Основной цикл обучения
def train_and_evaluate():
    # Конфигурация
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_epochs = 10
    lr = 1e-4

    # Трансформации
    basic_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Загрузка данных
    df_train = pd.read_csv("C:/Users/User/Documents/Triton_dataset_new/crop_dataset/labels_train.csv")
    df_val = pd.read_csv("C:/Users/User/Documents/Triton_dataset_new/crop_dataset/labels_val.csv")
    df_test = pd.read_csv("C:/Users/User/Documents/Triton_dataset_new/crop_dataset/labels_test.csv")

    train_dataset = TripletDataset(df_train, transform=basic_transforms)
    val_dataset = TripletDataset(df_val, transform=basic_transforms)
    test_dataset = TestDataset(df_test, transform=basic_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Инициализация модели
    model = TripletNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    # История метрик
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': [],
        'test_metrics': None
    }

    # Основной цикл обучения
    for epoch in range(num_epochs):
        # Обучение
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            anchor_imgs, pos_imgs, neg_imgs, _ = batch
            anchor_imgs = anchor_imgs.to(device)
            pos_imgs = pos_imgs.to(device)
            neg_imgs = neg_imgs.to(device)

            optimizer.zero_grad()
            embed_anchor = model(anchor_imgs)
            embed_positive = model(pos_imgs)
            embed_negative = model(neg_imgs)

            loss = criterion(embed_anchor, embed_positive, embed_negative)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                anchor_imgs, pos_imgs, neg_imgs, _ = batch
                anchor_imgs = anchor_imgs.to(device)
                pos_imgs = pos_imgs.to(device)
                neg_imgs = neg_imgs.to(device)

                embed_anchor = model(anchor_imgs)
                embed_positive = model(pos_imgs)
                embed_negative = model(neg_imgs)

                loss = criterion(embed_anchor, embed_positive, embed_negative)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        # Оценка на валидации
        val_metrics = evaluate_model(model, val_loader, device, epoch + 1, 'val')
        history['val_metrics'].append(val_metrics)

        # Вывод результатов эпохи
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print("Validation Metrics:")
        print(f"  Avg Pos Distance: {val_metrics['avg_pos_distance']:.4f}")
        print(f"  Avg Neg Distance: {val_metrics['avg_neg_distance']:.4f}")
        print(f"  Pos/Neg Ratio: {val_metrics['pos_neg_ratio']:.4f}")
        print(f"  Precision@1: {val_metrics['precision@1']:.4f}")
        print(f"  Precision@5: {val_metrics['precision@5']:.4f}")

        # Сохранение лучшей модели
        if val_metrics['pos_neg_ratio'] == min([m['pos_neg_ratio'] for m in history['val_metrics']]):
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved new best model")

    # Финальное тестирование на лучшей модели
    print("\nStarting final testing...")
    model.load_state_dict(torch.load('best_model.pth'))
    test_metrics = evaluate_model(model, test_loader, device, None, 'test')
    history['test_metrics'] = test_metrics

    # Вывод итоговых результатов
    print("\n=== Final Test Results ===")
    print(f"Test Samples: {test_metrics['num_samples']}")
    print(f"Avg Pos Distance: {test_metrics['avg_pos_distance']:.4f}")
    print(f"Avg Neg Distance: {test_metrics['avg_neg_distance']:.4f}")
    print(f"Pos/Neg Ratio: {test_metrics['pos_neg_ratio']:.4f}")
    print(f"Precision@1: {test_metrics['precision@1']:.4f}")
    print(f"Precision@5: {test_metrics['precision@5']:.4f}")

    # Визуализация истории обучения
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([m['pos_neg_ratio'] for m in history['val_metrics']], label='Pos/Neg Ratio')
    plt.plot([m['precision@1'] for m in history['val_metrics']], label='Precision@1')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

    return history


# Запуск обучения и тестирования
if __name__ == "__main__":
    history = train_and_evaluate()