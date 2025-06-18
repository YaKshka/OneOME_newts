import os
import cv2
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
from scipy.interpolate import splprep, splev
from scipy.ndimage import map_coordinates, gaussian_filter1d
from tqdm import tqdm
import re

def extract_smooth_centerline(mask, sigma=3):
    """
    Вычисляет центральную линию по медиальной оси (более устойчивый к шуму способ).
    """
    # Получаем расстояние до границы маски
    dist = cv2.distanceTransform(mask.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)

    # Бинаризуем расстояние — пиксели с max значениями ближе к центру
    dist_bin = (dist > dist.max() * 0.9).astype(np.uint8)

    # Находим контуры этой области
    contours, _ = cv2.findContours(dist_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        raise ValueError("Центральная линия не найдена")

    # Выбираем самый длинный контур — это и есть центральная ось
    centerline = max(contours, key=len).squeeze()  # shape: (N, 2)

    # Сортируем его по координатам X или Y, в зависимости от ориентации ящерицы
    dxy = np.ptp(centerline, axis=0)
    if dxy[0] > dxy[1]:
        centerline = centerline[np.argsort(centerline[:, 0])]
    else:
        centerline = centerline[np.argsort(centerline[:, 1])]

    # Сглаживаем координаты
    centerline = centerline.astype(np.float32)
    centerline[:, 0] = gaussian_filter1d(centerline[:, 0], sigma=sigma)
    centerline[:, 1] = gaussian_filter1d(centerline[:, 1], sigma=sigma)

    return centerline


def unwrap_belly_trimmed_ends(image, mask, pt1=None, pt2=None, save_path="unwrapped.jpg", debug_path="debug_centerline.jpg",
                               final_size=250, smoothness=400):
        # Используем keypoints как pt1 и pt2
    centerline = extract_smooth_centerline(mask)

    if len(centerline) < 2:
        raise ValueError("Центральная линия слишком короткая")
    
    dist_map = np.zeros_like(mask, dtype=np.uint8)
    for x, y in centerline.astype(int):
        dist_map[y, x] = 1


    if pt1 is not None and pt2 is not None:
        start = (int(pt1[1]), int(pt1[0]))  # (y, x)
        end = (int(pt2[1]), int(pt2[0]))    # (y, x)
    else:
        start = tuple(centerline[0].astype(int))
        end = tuple(centerline[-1].astype(int))


    path, _ = route_through_array(1 - dist_map, start, end, fully_connected=True)
    centerline = np.array([[x, y] for y, x in path], dtype=np.float32)


    x = centerline[:, 0]
    y = centerline[:, 1]

    # Прямое интерполирование по длине
    x_interp = np.interp(np.linspace(0, len(x) - 1, final_size), np.arange(len(x)), x)
    y_interp = np.interp(np.linspace(0, len(y) - 1, final_size), np.arange(len(y)), y)

    # Сглаживание вручную
    x_smooth = gaussian_filter1d(x_interp, sigma=5)
    y_smooth = gaussian_filter1d(y_interp, sigma=5)
    centerline_smooth = np.column_stack((x_smooth, y_smooth))
    if len(centerline_smooth) < 2:
        raise ValueError(" После усечения центрлайн слишком короткий. Возможно, маска плохая.")


    dx = gaussian_filter1d(np.gradient(centerline_smooth[:, 0]), sigma=3)
    dy = gaussian_filter1d(np.gradient(centerline_smooth[:, 1]), sigma=3)
    lengths = np.hypot(dx, dy)
    normals = np.column_stack((-dy / lengths, dx / lengths))


    h_img, w_img = image.shape[:2]
    lines = []
    max_strip_width = 0

    # Найдём допустимые индексы по маске (внутри маски)
    valid_indices = []
    for i in range(len(centerline_smooth)):
        cx, cy = map(int, centerline_smooth[i])
        if 0 <= cx < w_img and 0 <= cy < h_img and mask[cy, cx] != 0:
            valid_indices.append(i)

    # Если есть допустимые точки — обрезаем линию
    if valid_indices:
        start = valid_indices[0]
        end = valid_indices[-1] + 1  # +1, чтобы включить последнюю
        centerline_smooth = centerline_smooth[start:end]
        normals = normals[start:end]
        final_size = len(centerline_smooth)
    else:
        # Нет ни одной точки внутри маски — можно прервать
        print("Вся центральная линия вне маски!")
        return

    for i in range(final_size):
        cx, cy = centerline_smooth[i]
        nx, ny = normals[i]

        length_neg, length_pos = 0, 0
        for step in range(1, max(h_img, w_img)):
            px, py = int(cx - nx * step), int(cy - ny * step)
            if not (0 <= px < w_img and 0 <= py < h_img) or mask[py, px] == 0:
                break
            length_neg += 1
        for step in range(1, max(h_img, w_img)):
            px, py = int(cx + nx * step), int(cy + ny * step)
            if not (0 <= px < w_img and 0 <= py < h_img) or mask[py, px] == 0:
                break
            length_pos += 1

        strip_width = length_neg + length_pos
        max_strip_width = max(max_strip_width, strip_width)

        line = []
        for j in range(-length_neg, length_pos):
            px = cx + nx * j
            py = cy + ny * j
            if 0 <= int(py) < h_img and 0 <= int(px) < w_img:
                coords_sample = np.array([[py], [px]])
                pixel = np.stack([
                    map_coordinates(image[:, :, c], coords_sample, order=1, mode='reflect')[0]
                    for c in range(3)
                ], axis=-1)
                line.append(pixel)
        lines.append(np.array(line, dtype=np.uint8))

    unwrapped = np.zeros((final_size, max_strip_width, 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        if line.shape[0] < 2:
            continue
        resized_line = cv2.resize(line[None, :, :], (max_strip_width, 1), interpolation=cv2.INTER_LINEAR)
        unwrapped[i] = resized_line[0]

    if unwrapped.size == 0:
        raise ValueError(" Unwrapped изображение пустое. Невозможно выполнить resize.")

    final = cv2.resize(unwrapped, (final_size, final_size), interpolation=cv2.INTER_LINEAR)
    final_resized = cv2.resize(final, (224, 224), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(save_path, final_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


# Получение маски из сегментации
def get_segmentation_mask(image, img_path):
    h_img, w_img = image.shape[:2]
    seg_results = seg_model(img_path)
    masks = seg_results[0].masks
    if masks is None:
        return None
    mask_tensor = masks.data[0].cpu().numpy()
    mask_resized = cv2.resize(mask_tensor, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
    mask = (mask_resized * 255).astype(np.uint8)
    return mask


#  Получение ключевых точек
def get_keypoints_from_model(img_path):
    results = keypoint_model(img_path)
    for result in results:
        if result.keypoints is not None and len(result.keypoints) > 0:
            keypoints_raw = result.keypoints.data.cpu().numpy()[0]
            keypoints = [kp for kp in keypoints_raw if kp[2] > 0.3]
            if len(keypoints) < 2:
                keypoints = [kp for kp in keypoints_raw if kp[2] > 0.05]
            if len(keypoints) >= 2:
                pt1 = tuple(map(int, keypoints[0][:2]))
                pt2 = tuple(map(int, keypoints[1][:2]))
                return pt1, pt2
    return None, None


input_image_path = "/home/yakshka/prod2/get_photo/image.jpg"  #папка с изображением, которое ищем
output_dir = "/home/yakshka/prod2/get_photo"  #путь сохранения обрезанных фотографий (то же самое в модели vit)
os.makedirs(output_dir, exist_ok=True)

seg_model_path = "/home/yakshka/prod2/model/best_seg.pt"  #параметры модели сегментации
keypoint_model_path = "/home/yakshka/prod2/model/best_keypoints.pt" #параметры модели ключевых точек

# Загрузка моделей
seg_model = YOLO(seg_model_path)
keypoint_model = YOLO(keypoint_model_path)


def process_single_image(img_path, output_dir, species_name="unknown", individual_name="unknown"):
    """Обрабатывает одно изображение"""
    try:
        # Проверка расширения файла
        if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
            print(f"Файл {img_path} не является изображением (jpg/jpeg/png)")
            return False

        # Загрузка изображения
        image = cv2.imread(img_path)
        if image is None:
            print(f"Не удалось загрузить изображение: {img_path}")
            return False

        # Получение маски сегментации
        mask = get_segmentation_mask(image, img_path)
        if mask is None:
            print(f"Не удалось получить маску для {img_path}")
            return False

        # Получение ключевых точек
        pt1, pt2 = get_keypoints_from_model(img_path)
        if None in (pt1, pt2):
            print(f"Не удалось определить ключевые точки для {img_path}")
            return False

        # Подготовка путей для сохранения
        file_name = 'image_cropped.jpg'
        save_path = os.path.join(output_dir, file_name)
        debug_path = os.path.join(output_dir, "debug", f"debug_{file_name}")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)

        # Обработка и сохранение изображения
        unwrap_belly_trimmed_ends(image, mask, pt1, pt2, save_path, debug_path)
        print(f"Изображение успешно обработано и сохранено в: {save_path}")
        return True

    except Exception as e:
        print(f"Ошибка при обработке {img_path}: {str(e)}")
        return False

# Обработка одного изображения
if __name__ == "__main__":
    
    print("Начало обработки изображения...")
    process_single_image(input_image_path, output_dir)
    print("Обработка завершена!")