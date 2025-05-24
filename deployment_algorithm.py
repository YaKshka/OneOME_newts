from ultralytics import YOLO
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, medial_axis
from scipy.ndimage import map_coordinates
from scipy.interpolate import splprep, splev
from ultralytics import YOLO
from skimage.graph import route_through_array
import networkx as nx

def unwrap_belly_trimmed_ends(image, mask, save_path="unwrapped.png", debug_path="debug_centerline.jpg",
                               final_size=250, strip_half_width=100, smoothness=5, trim_ratio=0.15):
    from scipy.ndimage import gaussian_filter1d

    skeleton = skeletonize(mask > 0).astype(np.uint8)
    coords = np.column_stack(np.nonzero(skeleton))
    if len(coords) < 2:
        raise ValueError("Скелет слишком короткий")

    pt1, pt2 = coords[0], coords[-1]
    dist_map = np.zeros_like(mask, dtype=np.uint8)
    dist_map[skeleton > 0] = 1
    path, _ = route_through_array(1 - dist_map, pt1, pt2, fully_connected=True)
    centerline = np.array([[x, y] for y, x in path], dtype=np.float32)

    # 2. Сглаживаем
    tck, u = splprep(centerline.T, s=smoothness)
    u_fine = np.linspace(0, 1, final_size)
    x_smooth, y_smooth = splev(u_fine, tck)
    centerline_smooth = np.column_stack((x_smooth, y_smooth))

    # 3. Отбрасываем крайние 10%
    margin = int(0.1 * final_size)
    centerline_smooth = centerline_smooth[margin:-margin]
    final_size = len(centerline_smooth)

    # 4. Нормали
    dx, dy = np.gradient(centerline_smooth[:, 0]), np.gradient(centerline_smooth[:, 1])
    lengths = np.hypot(dx, dy)
    normals = np.column_stack((-dy / lengths, dx / lengths))

    # 5. Сэмплирование по полной ширине маски
    h_img, w_img = image.shape[:2]
    lines = []
    max_strip_width = 0

    for i in range(final_size):
        cx, cy = centerline_smooth[i]
        nx, ny = normals[i]

        # Расширяем нормаль в обе стороны до выхода за пределы маски
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

        # Обновляем максимальную ширину
        strip_width = length_neg + length_pos
        max_strip_width = max(max_strip_width, strip_width)

        # Сбор пикселей вдоль нормали
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

    # 6. Интерполяция до max_strip_width
    unwrapped = np.zeros((final_size, max_strip_width, 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        if line.shape[0] < 2:
            continue
        resized_line = cv2.resize(line[None, :, :], (max_strip_width, 1), interpolation=cv2.INTER_LINEAR)
        unwrapped[i] = resized_line[0]


    # 7. Финальное масштабирование
    final = cv2.resize(unwrapped, (final_size, final_size), interpolation=cv2.INTER_LINEAR)
    
    final_resized = cv2.resize(final, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Убедитесь, что путь заканчивается на .jpg
    save_path = save_path.replace(".png", ".jpg") if save_path.endswith(".png") else save_path
    cv2.imwrite(save_path, final_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


    # 8. Debug визуализация
    debug_img = image.copy()
    for pt in centerline_smooth:
        cv2.circle(debug_img, tuple(np.round(pt).astype(int)), 1, (0, 255, 255), -1)  # Жёлтая линия
    for i in range(0, len(centerline_smooth), 10):  # Отрисовка нормалей
        cx, cy = centerline_smooth[i]
        nx, ny = normals[i]
        p1 = (int(cx - nx * 20), int(cy - ny * 20))
        p2 = (int(cx + nx * 20), int(cy + ny * 20))
        cv2.line(debug_img, p1, p2, (255, 0, 255), 1)
    cv2.imwrite(debug_path, debug_img)

    print("✅ Готово: развёртка сохранена")


# Пути
image_path = "C:/klasss/archive/Тритон_ID/Тритон_Карелина/03.12.2024/28/IMG_9461.JPG"
model_path = "C:/klasss/archive/OneOME_newts/yolo11_seg_training/weights/best.pt"
output_path = "C:/klasss/archive/OneOME_newts/unwrapped_output.jpg"
debug_path = "C:/klasss/archive/OneOME_newts/debug_centerline.jpg"

# Загрузка изображения и модели
image = cv2.imread(image_path)
h_img, w_img = image.shape[:2]

model = YOLO(model_path)
results = model(image_path)
masks = results[0].masks

if masks is None:
    raise ValueError("❌ Модель не вернула сегментацию.")

# Преобразование маски
mask_tensor = masks.data[0].cpu().numpy()
mask_resized = cv2.resize(mask_tensor, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
mask = (mask_resized * 255).astype(np.uint8)

# Вызов финальной функции
unwrap_belly_trimmed_ends(image, mask, output_path, debug_path)