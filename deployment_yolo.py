import os
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.ndimage import map_coordinates, gaussian_filter1d

# ===============================
# КОНФИГ ПУТЕЙ
# ===============================

ROOT_DIR = r"C:/klasss/archive/OneOME_newts-main"

# Папки с данными
INPUT_ROOT = os.path.join(ROOT_DIR, "train")           # исходные фото по подпапкам
CROP_DATASET_DIR = os.path.join(ROOT_DIR, "crop_dataset")
SEND_PHOTO_DIR = os.path.join(ROOT_DIR, "send_photo")  # конечный вывод (unwrap)

# Модели
SEG_MODEL_PATH = os.path.join(ROOT_DIR, "yolo11s_seg_last/weights/best.pt")
KEYPOINT_MODEL_PATH = os.path.join(
    ROOT_DIR,
    "keypoints_01_12_2025/pose_newts_exp1/weights/best.pt"
)

os.makedirs(CROP_DATASET_DIR, exist_ok=True)
os.makedirs(SEND_PHOTO_DIR, exist_ok=True)

# Загрузка моделей
seg_model = YOLO(SEG_MODEL_PATH)
keypoint_model = YOLO(KEYPOINT_MODEL_PATH)


# ===============================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ===============================

def project_point_to_centerline(point, centerline):
    """Возвращает координаты ближайшей точки центрлайна и её индекс."""
    px, py = point
    diffs = centerline - np.array([px, py], dtype=np.float32)
    dists2 = np.sum(diffs ** 2, axis=1)
    idx = int(np.argmin(dists2))
    return centerline[idx].copy(), idx


def extract_smooth_centerline(mask, step=2, sigma_x=3):
    """
    Строим медиальную линию по маске:
    для каждого y берём середину между левым и правым краем маски.
    """
    h, w = mask.shape
    ys = np.arange(0, h, step)
    pts = []
    for y in ys:
        xs = np.where(mask[y] > 0)[0]
        if len(xs) == 0:
            continue
        x_center = 0.5 * (xs.min() + xs.max())
        pts.append([x_center, y])

    if len(pts) < 2:
        raise ValueError("Центральная линия не найдена")

    centerline = np.array(pts, dtype=np.float32)
    # сглаживаем X, Y почти и так монотонен
    centerline[:, 0] = gaussian_filter1d(centerline[:, 0], sigma=sigma_x)
    return centerline




def unwrap_belly_trimmed_ends(image, mask, centerline,
                              pt1=None, pt2=None,
                              save_path="unwrapped.jpg",
                              final_size=250):
    # центрлайн уже сглажен
    if len(centerline) < 2:
        raise ValueError("Центральная линия слишком короткая")

    # --- 1. Обрезаем центрлайн по ключевым точкам, жёстко по Y ---
    if pt1 is not None and pt2 is not None:
        y_min = min(pt1[1], pt2[1])
        y_max = max(pt1[1], pt2[1])

        # убираем по 5% диапазона сверху и снизу, чтобы не брать самые шумные края
        margin = int(0.05 * (y_max - y_min))
        y_min += margin
        y_max -= margin

        mask_y = (centerline[:, 1] >= y_min) & (centerline[:, 1] <= y_max)
        centerline = centerline[mask_y]

    if len(centerline) < 2:
        raise ValueError("После обрезки по kpts центрлайн слишком короткий")

    # --- 2. Ресэмплинг по длине до final_size ---
    x = centerline[:, 0]
    y = centerline[:, 1]
    t = np.linspace(0, 1, len(centerline))
    t_new = np.linspace(0, 1, final_size)

    x_interp = np.interp(t_new, t, x)
    y_interp = np.interp(t_new, t, y)

    x_smooth = gaussian_filter1d(x_interp, sigma=7)
    y_smooth = gaussian_filter1d(y_interp, sigma=3)
    centerline_smooth = np.column_stack((x_smooth, y_smooth)).astype(np.float32)

    # === 2.1 Максимально выпрямляем концы центрлайна ===
    n = len(centerline_smooth)
    if n >= 10:
        k_tail = max(3, int(0.05 * n))  # хвосты ~5% длины

        # опорные точки сразу перед хвостами
        x_top, y_top = centerline_smooth[k_tail, 0], centerline_smooth[k_tail, 1]
        x_bot, y_bot = centerline_smooth[n - k_tail - 1, 0], centerline_smooth[n - k_tail - 1, 1]

        def x_on_midline(yv):
            if y_bot == y_top:
                return x_top
            t_rel = (yv - y_top) / (y_bot - y_top)
            return x_top + t_rel * (x_bot - x_top)

        # верхний хвост
        for i in range(k_tail):
            y_i = centerline_smooth[i, 1]
            centerline_smooth[i, 0] = x_on_midline(y_i)

        # нижний хвост
        for i in range(n - k_tail, n):
            y_i = centerline_smooth[i, 1]
            centerline_smooth[i, 0] = x_on_midline(y_i)
    # === конец выпрямления концов ===

    # --- 3. Нормали к центрлайну ---
    dx = gaussian_filter1d(np.gradient(centerline_smooth[:, 0]), sigma=3)
    dy = gaussian_filter1d(np.gradient(centerline_smooth[:, 1]), sigma=3)
    lengths = np.hypot(dx, dy) + 1e-6
    normals = np.column_stack((-dy / lengths, dx / lengths))

    h_img, w_img = image.shape[:2]
    lines = []
    max_strip_width = 0

    # --- 4. Для каждой точки центрлайна идём по нормали до границ маски ---
    for i in range(final_size):
        cx, cy = centerline_smooth[i]
        nx, ny = normals[i]

        length_neg, length_pos = 0, 0

        # в одну сторону
        for step in range(1, max(h_img, w_img)):
            px, py = int(cx - nx * step), int(cy - ny * step)
            if not (0 <= px < w_img and 0 <= py < h_img) or mask[py, px] == 0:
                break
            length_neg += 1

        # в другую
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
                coords_sample = np.array([[py], [px]], dtype=np.float32)
                pixel = np.stack(
                    [map_coordinates(image[:, :, c], coords_sample,
                                     order=1, mode='reflect')[0]
                     for c in range(3)],
                    axis=-1
                )
                line.append(pixel)
        lines.append(np.array(line, dtype=np.uint8))

    if not lines:
        raise ValueError("Не удалось построить развёртку: нет валидных линий")

    unwrapped = np.zeros((final_size, max_strip_width, 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        if line.shape[0] < 2:
            continue
        resized_line = cv2.resize(
            line[None, :, :],
            (max_strip_width, 1),
            interpolation=cv2.INTER_LINEAR
        )
        unwrapped[i] = resized_line[0]

    final = cv2.resize(unwrapped, (final_size, final_size),
                       interpolation=cv2.INTER_LINEAR)
    final_resized = cv2.resize(final, (224, 224),
                               interpolation=cv2.INTER_LINEAR)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, final_resized,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])




def get_segmentation_mask(image, img_path):
    h_img, w_img = image.shape[:2]
    seg_results = seg_model(img_path)
    masks = seg_results[0].masks
    if masks is None:
        return None
    mask_tensor = masks.data[0].cpu().numpy()
    mask_resized = cv2.resize(mask_tensor, (w_img, h_img),
                              interpolation=cv2.INTER_LINEAR)
    mask = (mask_resized * 255).astype(np.uint8)
    return mask


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


def save_debug_image(image, mask, centerline, pt1, pt2, save_path):
    """Создаёт debug-картинку с маской, ключевыми точками и центральной линией."""
    debug = image.copy()

    # Маска зелёным
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_colored = np.zeros_like(mask_rgb)
    mask_colored[:, :, 1] = mask
    debug = cv2.addWeighted(debug, 0.7, mask_colored, 0.3, 0)

    # Центральная линия (красная)
    if centerline is not None and len(centerline) > 1:
        for i in range(len(centerline) - 1):
            x1, y1 = map(int, centerline[i])
            x2, y2 = map(int, centerline[i + 1])
            cv2.line(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Ключевые точки
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    if pt1 is not None:
        cv2.circle(debug, pt1, 5, (255, 0, 0), -1)
        cv2.putText(
            debug, "kpt0",
            (pt1[0] + 5, pt1[1] - 5),
            font, font_scale,
            (255, 255, 255),
            thickness, cv2.LINE_AA,
        )

    if pt2 is not None:
        cv2.circle(debug, pt2, 5, (255, 0, 0), -1)
        cv2.putText(
            debug, "kpt1",
            (pt2[0] + 5, pt2[1] - 5),
            font, font_scale,
            (255, 255, 255),
            thickness, cv2.LINE_AA,
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True
                )
    cv2.imwrite(save_path, debug, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


# ===============================
# ОБРАБОТКА ПАПОК
# ===============================
def process_folders(input_root, output_root):
    for folder_name in os.listdir(input_root):
        folder_path = os.path.join(input_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        output_folder = os.path.join(output_root, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        for file_name in os.listdir(folder_path):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            input_path = os.path.join(folder_path, file_name)
            output_path = os.path.join(output_folder, file_name)

            try:
                image = cv2.imread(input_path)
                if image is None:
                    print(f"❌ Поврежденное изображение: {input_path}")
                    continue

                mask = get_segmentation_mask(image, input_path)
                if mask is None:
                    print(f"❌ Маска не найдена: {input_path}")
                    continue

                pt1, pt2 = get_keypoints_from_model(input_path)
                if None in (pt1, pt2):
                    print(f"❌ Ключевые точки не найдены: {input_path}")
                    continue

                centerline = extract_smooth_centerline(mask)

                # только финальная развёртка, без debug-изображений
                unwrap_belly_trimmed_ends(
                    image=image,
                    mask=mask,
                    centerline=centerline,
                    pt1=pt1,
                    pt2=pt2,
                    save_path=output_path,
                )

                print(f"✔️ Успешно обработано: {input_path}")

            except Exception as e:
                print(f"🔥 Ошибка при обработке {input_path}: {e}")



if __name__ == "__main__":
    print("🚀 Начало обработки...")
    process_folders(INPUT_ROOT, SEND_PHOTO_DIR)
    print("🏁 Обработка завершена!")
    print(f"📁 Все результаты находятся в: {SEND_PHOTO_DIR}")


'''
from ultralytics import YOLO

def train():
    model = YOLO("yolo11n-pose.pt")
    model.train(
        data="C:/klasss/archive/OneOME_newts-main/keypoints_01_12_2025/data.yaml",
        epochs=100,
        imgsz=640,
        degrees=5.0,
        translate=0.05,
        project="C:/klasss/archive/OneOME_newts-main/keypoints_01_12_2025",
        name="pose_newts_exp1",
        scale=0.10,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.4,
    )

if __name__ == "__main__":
    train()'''


