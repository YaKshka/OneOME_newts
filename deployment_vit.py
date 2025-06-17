import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os
from tqdm import tqdm
import shutil
import torch.nn as nn
import timm
import pickle
from torchvision import transforms


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


def load_model(model_path, device):
    model = TripletNet(base_model_name='vit_base_patch16_224', embedding_dim=128)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_embedding(image_path, model, transform, device):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(image)
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Ошибка обработки изображения {image_path}: {str(e)}")
        return None


def compute_distances(embeddings1, embedding2):
    similarities = cosine_similarity(embeddings1, embedding2.reshape(1, -1))
    distances = 1 - similarities.flatten()
    return distances


def save_embeddings(embeddings, paths, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'paths': paths}, f)


def load_embeddings(save_path):
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['paths']


def find_similar_images(model_path, database_dir, query_image_path, output_dir, transform, device):
    # Путь для сохранения эмбеддингов
    embeddings_save_path = os.path.join(database_dir, 'database_embeddings.pkl')

    # Проверяем, есть ли сохраненные эмбеддинги
    if os.path.exists(embeddings_save_path):
        print("Загрузка сохраненных эмбеддингов...")
        database_embeddings, database_image_paths = load_embeddings(embeddings_save_path)
    else:
        print("Вычисление эмбеддингов базы...")
        model = load_model(model_path, device)

        database_image_paths = []
        for root, _, files in os.walk(database_dir):
            # Пропускаем папку с результатами
            if os.path.normpath(root).startswith(os.path.normpath(output_dir)):
                continue

            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    if os.path.exists(full_path):
                        database_image_paths.append(full_path)

        print(f"Найдено {len(database_image_paths)} изображений в базе")

        database_embeddings = []
        valid_paths = []
        for path in tqdm(database_image_paths, desc="Обработка базы"):
            emb = get_embedding(path, model, transform, device)
            if emb is not None:
                database_embeddings.append(emb)
                valid_paths.append(path)

        database_embeddings = np.array(database_embeddings)
        save_embeddings(database_embeddings, valid_paths, embeddings_save_path)
        print(f"Эмбеддинги сохранены в {embeddings_save_path}")
        database_image_paths = valid_paths

    # Обработка запросного изображения
    model = load_model(model_path, device)
    query_embedding = get_embedding(query_image_path, model, transform, device)
    if query_embedding is None:
        print("Не удалось обработать запросное изображение")
        return

    # Поиск похожих изображений
    distances = compute_distances(np.array(database_embeddings), query_embedding)
    top5_idx = np.argsort(distances)[:5]

    print("\n=== Топ-5 результатов ===")
    os.makedirs(output_dir, exist_ok=True)

    for i, idx in enumerate(top5_idx, 1):
        src_path = database_image_paths[idx]
        dst_filename = f"top{i}_{os.path.basename(src_path)}"
        dst_path = os.path.join(output_dir, dst_filename)

        try:
            shutil.copy(src_path, dst_path)
            class_name = os.path.basename(os.path.dirname(os.path.dirname(src_path)))
            individual = os.path.basename(os.path.dirname(src_path))
            print(f"{i}. Класс: {class_name} | Особь: {individual} | Путь: {src_path}")
        except Exception as e:
            print(f"Ошибка копирования файла {src_path}: {str(e)}")

    print(f"\nРезультаты сохранены в: {output_dir}")


if __name__ == "__main__":
    # Конфигурация путей
    MODEL_PATH = "C:/Users/User/PycharmProjects/Tritons/best_model.pth" #путь до параметров модели (best_model.pth)
    DATABASE_DIR = "C:/Users/User/Documents/Triton_dataset_new/crop_dataset" #путь до базы (уже обрезанные после yolo)
    QUERY_IMAGE = "C:/Users/User/Documents/original_17-4-jan_17_ribbet.JPG" #Изображение, которое ищем
    OUTPUT_DIR = "C:/Users/User/Documents/Triton_results"  # Папка с результатом
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Трансформы для изображений
    TRANSFORMS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Запуск поиска
    find_similar_images(
        model_path=MODEL_PATH,
        database_dir=DATABASE_DIR,
        query_image_path=QUERY_IMAGE,
        output_dir=OUTPUT_DIR,
        transform=TRANSFORMS,
        device=DEVICE
    )