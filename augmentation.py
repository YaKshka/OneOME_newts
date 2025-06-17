import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

# Настройки аугментации
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
])


def read_image_correctly(path):
    """Чтение изображения с учетом кириллических путей в Windows"""
    try:
        # Читаем файл как массив байтов
        with open(path, 'rb') as f:
            bytes = np.frombuffer(f.read(), dtype=np.uint8)

        # Декодируем изображение
        image = cv2.imdecode(bytes, cv2.IMREAD_COLOR)

        if image is None:
            print(f"Не удалось декодировать изображение: {path}")
            return None

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Ошибка при чтении файла {path}: {str(e)}")
        return None


def save_image_correctly(image, path):
    """Сохранение изображения с учетом кириллических путей"""
    try:
        # Конвертируем обратно в BGR для сохранения
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Кодируем и сохраняем
        _, encoded = cv2.imencode(os.path.splitext(path)[1], image)
        with open(path, 'wb') as f:
            encoded.tofile(f)
        return True
    except Exception as e:
        print(f"Ошибка при сохранении файла {path}: {str(e)}")
        return False


def augment_images(input_dir, output_dir, num_augmentations=1):
    """Основная функция аугментации"""
    os.makedirs(output_dir, exist_ok=True)

    # Получаем список всех поддиректорий (видов)
    species_list = [d for d in os.listdir(input_dir)
                    if os.path.isdir(os.path.join(input_dir, d))]

    for species in tqdm(species_list, desc="Обработка видов"):
        species_path = os.path.join(input_dir, species)
        output_species_path = os.path.join(output_dir, species)
        os.makedirs(output_species_path, exist_ok=True)

        # Обрабатываем все файлы в директории вида
        for img_name in os.listdir(species_path):
            img_path = os.path.join(species_path, img_name)

            if not os.path.isfile(img_path):
                continue

            # Читаем изображение
            image = read_image_correctly(img_path)
            if image is None:
                continue

            # Сохраняем оригинал
            original_output_path = os.path.join(output_species_path, f"original_{img_name}")
            if not save_image_correctly(image, original_output_path):
                continue

            # Создаем аугментированные версии
            for i in range(num_augmentations):
                augmented = transform(image=image)
                aug_output_path = os.path.join(output_species_path, f"aug_{i}_{img_name}")
                save_image_correctly(augmented["image"], aug_output_path)


if __name__ == "__main__":
    # Укажите ваши пути здесь
    input_directory = r"C:\Users\User\Documents\tritons_dataset\Тритон ID\Ribbed_newt"
    output_directory = r"C:\Users\User\Documents\tritons_dataset\augmented_rebbed"

    print("Начало аугментации...")
    augment_images(input_directory, output_directory, num_augmentations=3)
    print("Аугментация завершена!")