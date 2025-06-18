import torch
from torchvision import transforms
import os
from ultralytics import YOLO
import deployment_yolo
import deployment_vit
import asyncio

async def photo_processing():
    
    input_image_path = "/home/yakshka/prod2/get_photo/image.jpg"
    output_dir = "/home/yakshka/prod2/get_photo"
    
    print("Начало обработки изображения...")
    
    try:
        await deployment_yolo.process_single_image(input_image_path, output_dir)
        print("Обработка завершена!")
        
        MODEL_PATH = "/home/yakshka/prod2/model/best_model.pth" #путь до параметров модели (best_model.pth)
        DATABASE_DIR = "/home/yakshka/prod2/model/crop_dataset" #путь до базы (уже обрезанные после yolo)
        QUERY_IMAGE = "/home/yakshka/prod2/get_photo/image_cropped.jpg"  #Изображение, которое ищем
        OUTPUT_DIR = "/home/yakshka/prod2/send_photo"  # Папка с результатом
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        TRANSFORMS = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        await deployment_vit.find_similar_images(
            model_path=MODEL_PATH,
            database_dir=DATABASE_DIR,
            query_image_path=QUERY_IMAGE,
            output_dir=OUTPUT_DIR,
            transform=TRANSFORMS,
            device=DEVICE
        )
        
        return True

    except Exception as e:
        print(f"Ошибка при обработке: {str(e)}")
        return False


if __name__ == "__main__":
    photo_processing()