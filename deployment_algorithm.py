import os
import cv2
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
from scipy.interpolate import splprep, splev
from scipy.ndimage import map_coordinates, gaussian_filter1d
from tqdm import tqdm

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
        raise ValueError("❌ После усечения центрлайн слишком короткий. Возможно, маска плохая.")


    dx = gaussian_filter1d(np.gradient(centerline_smooth[:, 0]), sigma=3)
    dy = gaussian_filter1d(np.gradient(centerline_smooth[:, 1]), sigma=3)
    lengths = np.hypot(dx, dy)
    normals = np.column_stack((-dy / lengths, dx / lengths))


    h_img, w_img = image.shape[:2]
    lines = []
    max_strip_width = 0

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
        raise ValueError("❌ Unwrapped изображение пустое. Невозможно выполнить resize.")

    final = cv2.resize(unwrapped, (final_size, final_size), interpolation=cv2.INTER_LINEAR)
    final_resized = cv2.resize(final, (224, 224), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(save_path, final_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


    '''
    # Debug визуализация
    debug_img = image.copy()

    # Наложение маски: делаем маску зелёной (или любой другой цвет)
    mask_overlay = np.zeros_like(debug_img)
    mask_overlay[mask > 0] = (0, 180, 0)  # Зелёная маска
    debug_img = cv2.addWeighted(debug_img, 1.0, mask_overlay, 0.4, 0)

    # Отрисовка центральной линии
    for pt in centerline_smooth:
        cv2.circle(debug_img, tuple(np.round(pt).astype(int)), 1, (0, 255, 255), -1)  # Жёлтые точки

    # Отрисовка нормалей
    for i in range(0, len(centerline_smooth), 10):
        cx, cy = centerline_smooth[i]
        nx, ny = normals[i]
        p1 = (int(cx - nx * 20), int(cy - ny * 20))
        p2 = (int(cx + nx * 20), int(cy + ny * 20))
        cv2.line(debug_img, p1, p2, (255, 0, 255), 1)  # Фиолетовые линии

    # Сохраняем debug
    cv2.imwrite(debug_path, debug_img)
    '''

    print("✅ Готово: развёртка сохранена.")

'''
if __name__ == "__main__":
    # Пути к файлам и моделям
    img_path = "C:/klasss/archive/Тритон ID/Тритон Карелина/03.12.2024/55/IMG_9665.JPG"
    seg_model_path = "C:/klasss/archive/newts/final_segmentation/yolo11s_seg_final/weights/best.pt"
    keypoint_model_path = "C:/klasss/archive/OneOME_newts/lizard_belly_keypoints23/weights/best.pt"
    save_unwrap_path = "C:/klasss/archive/OneOME_newts/debug/unwrapped_output.jpg"
    save_debug_path = "C:/klasss/archive/OneOME_newts/debug/debug_centerline.jpg"

    # Загружаем изображение
    image = cv2.imread(img_path)
    h_img, w_img = image.shape[:2]

    # Загружаем модели
    seg_model = YOLO(seg_model_path)
    keypoint_model = YOLO(keypoint_model_path)

    # Предсказание сегментации
    seg_results = seg_model(img_path)
    masks = seg_results[0].masks
    if masks is None:
        raise ValueError("❌ Сегментация не найдена.")
    mask_tensor = masks.data[0].cpu().numpy()
    mask_resized = cv2.resize(mask_tensor, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
    mask = (mask_resized * 255).astype(np.uint8)

    # Предсказание ключевых точек
    results = keypoint_model(img_path)

    pt1, pt2 = None, None  # значения по умолчанию

    for result in results:
        if result.keypoints is not None and len(result.keypoints) > 0:
            keypoints_raw = result.keypoints.data.cpu().numpy()[0]  # (num_keypoints, 3)

            # Сначала фильтруем с conf > 0.3
            keypoints = [kp for kp in keypoints_raw if kp[2] > 0.3]

            # Если не нашли, пробуем с conf > 0.05
            if len(keypoints) < 2:
                keypoints = [kp for kp in keypoints_raw if kp[2] > 0.05]

            # Если нашли 2 точки — сохраняем
            if len(keypoints) >= 2:
                pt1 = tuple(map(int, keypoints[0][:2]))  # начало
                pt2 = tuple(map(int, keypoints[1][:2]))  # конец
                print("✅ Используем ключевые точки для построения центральной линии.")
                break

    # Теперь выбор между режимами:
    if pt1 is not None and pt2 is not None:
        unwrap_belly_trimmed_ends(image, mask, pt1, pt2, save_unwrap_path, save_debug_path)
    else:
        print("⚠️ Ключевые точки не найдены. Переход к старому методу с усечением 15% по краям.")
        unwrap_belly_trimmed_ends(image, mask, None, None, save_unwrap_path, save_debug_path)
'''
# ⚙️ Параметры
input_dir = r"C:/klasss/archive/newts/model_final/images"
output_dir = r"C:/klasss/archive/newts/model_final/finish"
os.makedirs(output_dir, exist_ok=True)

seg_model_path = r"C:/klasss/archive/newts/final_segmentation/yolo11s_seg_final/weights/best.pt"
keypoint_model_path = r"C:/klasss/archive/OneOME_newts/lizard_belly_keypoints23/weights/best.pt"

# 📦 Загрузка моделей
seg_model = YOLO(seg_model_path)
keypoint_model = YOLO(keypoint_model_path)


# 📌 Получение маски из сегментации
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


# 📌 Получение ключевых точек
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


# 🚀 Основной цикл
for file_name in tqdm(os.listdir(input_dir)):
    if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    try:
        img_path = os.path.join(input_dir, file_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ Поврежденное изображение: {file_name}")
            continue

        mask = get_segmentation_mask(image, img_path)
        if mask is None:
            print(f"⚠️ Маска не найдена для {file_name}")
            continue

        pt1, pt2 = get_keypoints_from_model(img_path)

        save_path = os.path.join(output_dir, file_name)

        try:
            unwrap_belly_trimmed_ends(image, mask, pt1, pt2, save_path)
        except Exception as unwrap_error:
            print(f"⚠️ Ошибка unwrap для {file_name}: {unwrap_error}")
            continue

    except Exception as e:
        print(f"⚠️ Ошибка при обработке {file_name}: {e}")
        continue




