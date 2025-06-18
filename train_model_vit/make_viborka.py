import os
import shutil
import random

# Параметры
source_dir = r'C:/Users/User/Documents/tritons_dataset/Augmented_triton_dataset/augmented_ribbed'  # замените на путь к папке с особями
dest_dir = r'C:/Users/User/Documents/tritons_dataset'   # куда создавать train, val, test

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# Создаем папки train, val, test
for folder in ['train', 'val', 'test']:
    os.makedirs(os.path.join(dest_dir, folder), exist_ok=True)

# Собираем список всех особей (имена папок)
persons = [name for name in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, name))]
random.shuffle(persons)

# Распределяем по группам
num_persons = len(persons)
train_end = int(num_persons * train_ratio)
val_end = train_end + int(num_persons * val_ratio)

train_persons = persons[:train_end]
val_persons = persons[train_end:val_end]
test_persons = persons[val_end:]

def copy_persons(person_list, target_folder):
    for особь in person_list:
        src_особь_path = os.path.join(source_dir, особь)
        dst_особь_path = os.path.join(dest_dir, target_folder, особь)
        os.makedirs(dst_особь_path, exist_ok=True)

        for фото in os.listdir(src_особь_path):
            src_file = os.path.join(src_особь_path, фото)
            dst_file = os.path.join(dst_особь_path, фото)
            shutil.copy2(src_file, dst_file)

# Копируем файлы с сохранением структуры
copy_persons(train_persons, 'train')
copy_persons(val_persons, 'val')
copy_persons(test_persons, 'test')

print("Распределение завершено.")