"""
Кейс №6 — Машинное зрение: выделение объектов на изображении

Задача: автоматически выделить контуры и маски объектов
на двумерном изображении при помощи оконно-матричных
преобразований (фильтрации).

Подход — фильтрация (лекция):
  Два класса методов:
    1. Фильтры — универсальная обработка изображения
       для выделения структур (контуров).
    2. Метод опорных фигур — обучение модели видеть
       конкретные объекты (нейронные сети).

  Мы реализуем первый подход.

Оконно-матричное преобразование (лекция):
  Для каждого пикселя (x, y) исходного изображения:

    P'_{xy} = Σ_m  C_m · P^m_{xy}

  где C_m — коэффициенты ядра (матрицы фильтра),
      P^m_{xy} — значения пикселей в окрестности (x, y),
      m — индекс ячейки внутри оконной матрицы.

  Результат зависит от коэффициентов C:
    - Размытие (blur): C ~ e^{-r²}  (гауссово ядро)
    - Повышение резкости (sharpen): C с отрицательными
      коэффициентами по краям и большим в центре
    - Выделение контуров: C как конечная разность
      (производная по нескольким направлениям)

Ключевая идея (лекция):
  Контуры объектов — это места, где производная
  изображения велика. Фон меняется мало, а на границе
  объекта происходит резкое изменение яркости.
  Поэтому фильтр-производная подсвечивает контуры.

Цепочка фильтров:
  1. Размытие (убираем шум).
  2. Выделение контуров (производная).
  3. Бинаризация (порог).
  4. Заполнение масок (кластеризация).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
from PIL import Image
import json

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10


# ============================================================
# 1. Оконно-матричное преобразование (свёртка)
# ============================================================

def convolve2d(image, kernel):
    """
    Оконно-матричное преобразование (лекция):

        P'_{xy} = Σ_m  C_m · P^m_{xy}

    Для каждого пикселя берём окрестность, умножаем
    поэлементно на ядро (матрицу коэффициентов C),
    суммируем — получаем новое значение пикселя.

    image:  2D массив (высота, ширина)
    kernel: 2D массив (размер ядра, например 3×3)
    """
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Дополнение нулями на границах
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    result = np.zeros_like(image, dtype=np.float64)
    for i in range(h):
        for j in range(w):
            window = padded[i:i + kh, j:j + kw]
            result[i, j] = np.sum(window * kernel)

    return result


def convolve2d_fast(image, kernel):
    """
    Быстрая свёртка через numpy (сдвиги массива).
    Та же формула P'_{xy} = Σ_m C_m · P^m_{xy}.
    """
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    result = np.zeros_like(image, dtype=np.float64)
    for di in range(kh):
        for dj in range(kw):
            result += kernel[di, dj] * padded[di:di + h, dj:dj + w]

    return result


# ============================================================
# 2. Библиотека фильтров (ядра свёртки)
# ============================================================

def kernel_gaussian_blur(size=5, sigma=1.0):
    """
    Размытие по Гауссу (лекция):
    Коэффициенты C ~ e^{-r²/(2σ²)},
    где r — расстояние от центра ядра.

    Убирает шум, сглаживает мелкие детали.
    """
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / kernel.sum()


def kernel_box_blur(size=3):
    """
    Простое усреднение (box blur).
    Все коэффициенты одинаковы: C_m = 1/N.
    """
    return np.ones((size, size)) / (size * size)


def kernel_sharpen():
    """
    Повышение резкости — Sharpen (лекция, фото с доски):
    Центральный коэффициент 9, все остальные −1.

    [[-1, -1, -1],
     [-1,  9, -1],
     [-1, -1, -1]]
    """
    return np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float64)


def kernel_edge_detect_1():
    """
    Выделение контуров — вариант 1 (лекция, фото с доски):
    Производная по 4 направлениям, центр 0.

    [[-1, -1,  1],
     [-1,  0,  1],
     [-1,  1,  1]]
    """
    return np.array([
        [-1, -1,  1],
        [-1,  0,  1],
        [-1,  1,  1]
    ], dtype=np.float64)


def kernel_edge_detect_laplacian():
    """
    Лапласиан — выделение контуров по всем направлениям.
    Сумма вторых производных по x и y.

    [[ 0, -1,  0],
     [-1,  4, -1],
     [ 0, -1,  0]]
    """
    return np.array([
        [0, -1,  0],
        [-1,  4, -1],
        [0, -1,  0]
    ], dtype=np.float64)


def kernel_edge_detect_full():
    """
    Лапласиан с диагоналями — 8 направлений.

    [[-1, -1, -1],
     [-1,  8, -1],
     [-1, -1, -1]]
    """
    return np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float64)


def kernel_gradient_x():
    """Производная по x (конечная разность по горизонтали)."""
    return np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)


def kernel_gradient_y():
    """Производная по y (конечная разность по вертикали)."""
    return np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)


# ============================================================
# 3. Обработка изображений
# ============================================================

def to_grayscale(image):
    """Перевод в оттенки серого (если RGB)."""
    if len(image.shape) == 3:
        return np.mean(image[:, :, :3], axis=2)
    return image.astype(np.float64)


def normalize_image(image):
    """Нормировка изображения в диапазон [0, 1]."""
    vmin, vmax = image.min(), image.max()
    if vmax - vmin < 1e-12:
        return np.zeros_like(image)
    return (image - vmin) / (vmax - vmin)


def binarize(image, threshold=0.5):
    """Бинаризация: пиксель > порога → 1, иначе → 0."""
    return (image > threshold).astype(np.float64)


def gradient_magnitude(image):
    """
    Модуль градиента: sqrt(Gx² + Gy²).
    Комбинирует конечные разности по x и y.
    """
    gx = convolve2d_fast(image, kernel_gradient_x())
    gy = convolve2d_fast(image, kernel_gradient_y())
    return np.sqrt(gx**2 + gy**2)


def flood_fill_mask(binary, min_area=50):
    """
    Заполнение масок объектов из бинарного изображения контуров.
    Простая кластеризация связных компонент.
    """
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    mask = np.zeros_like(binary, dtype=np.float64)
    label = 0

    for i in range(h):
        for j in range(w):
            if binary[i, j] > 0.5 and not visited[i, j]:
                # BFS для поиска связной компоненты
                queue = [(i, j)]
                component = []
                visited[i, j] = True

                while queue:
                    ci, cj = queue.pop(0)
                    component.append((ci, cj))

                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            if binary[ni, nj] > 0.5 and not visited[ni, nj]:
                                visited[ni, nj] = True
                                queue.append((ni, nj))

                if len(component) >= min_area:
                    label += 1
                    for ci, cj in component:
                        mask[ci, cj] = label

    return mask, label


def morphological_dilate(binary, radius=2):
    """Морфологическое расширение для замыкания контуров."""
    size = 2 * radius + 1
    kernel = np.ones((size, size))
    result = convolve2d_fast(binary, kernel)
    return (result > 0.5).astype(np.float64)


def morphological_erode(binary, radius=1):
    """Морфологическое сужение для удаления шума."""
    size = 2 * radius + 1
    kernel = np.ones((size, size)) / (size * size)
    result = convolve2d_fast(binary, kernel)
    return (result > 0.99).astype(np.float64)


# ============================================================
# 4. Генерация тестовых изображений
# ============================================================

def generate_test_simple(size=200):
    """Простой тест: красная фигура на белом фоне."""
    img = np.ones((size, size, 3))
    # Круг
    cy, cx = size // 3, size // 3
    Y, X = np.ogrid[:size, :size]
    circle = (X - cx)**2 + (Y - cy)**2 < (size // 6)**2
    img[circle] = [0.9, 0.1, 0.1]
    # Прямоугольник
    img[size//2:size//2+size//4, size//2:size//2+size//3] = [0.1, 0.1, 0.8]
    # Треугольник
    for i in range(size // 5):
        left = size // 6 + i
        right = size // 6 + size // 3 - i
        row = size * 2 // 3 + i
        if row < size and left < right:
            img[row, left:right] = [0.1, 0.7, 0.1]
    return img


def generate_test_medium(size=200):
    """Средняя сложность: несколько объектов, неоднородный фон."""
    np.random.seed(42)
    # Градиентный фон
    Y, X = np.ogrid[:size, :size]
    bg = 0.7 + 0.2 * X / size + 0.05 * np.random.randn(size, size)
    img = np.stack([bg, bg * 0.95, bg * 1.05], axis=2)
    img = np.clip(img, 0, 1)
    # Тёмный эллипс
    ellipse = ((X - size * 0.3)**2 / (size * 0.15)**2 +
               (Y - size * 0.4)**2 / (size * 0.1)**2) < 1
    img[ellipse] = [0.2, 0.15, 0.1]
    # Светлый круг
    circle = (X - size * 0.7)**2 + (Y - size * 0.6)**2 < (size * 0.12)**2
    img[circle] = [1.0, 0.95, 0.8]
    # Маленький квадрат
    img[size//8:size//8+size//10, size*3//4:size*3//4+size//10] = [0.1, 0.5, 0.9]
    return np.clip(img, 0, 1)


def generate_test_hard(size=200):
    """Сложный тест: низкий контраст, шум."""
    np.random.seed(123)
    bg = 0.5 + 0.15 * np.random.randn(size, size)
    img = np.stack([bg, bg, bg], axis=2)
    Y, X = np.ogrid[:size, :size]
    # Объект с низким контрастом
    obj1 = ((X - size * 0.35)**2 + (Y - size * 0.35)**2) < (size * 0.15)**2
    img[obj1] = img[obj1] + 0.15
    # Ещё один
    obj2 = ((X - size * 0.65)**2 + (Y - size * 0.7)**2) < (size * 0.1)**2
    img[obj2] = img[obj2] - 0.12
    return np.clip(img, 0, 1)


# ============================================================
# 5. Пайплайн выделения объектов
# ============================================================

def detect_objects(image_rgb, blur_sigma=1.5, blur_size=5,
                   edge_method='gradient', threshold=0.15,
                   dilate_radius=2, min_area=30, verbose=True):
    """
    Полный пайплайн выделения объектов (лекция):

    1. Перевод в оттенки серого.
    2. Размытие по Гауссу (убираем шум).
    3. Выделение контуров (фильтр-производная).
    4. Нормировка + бинаризация (порог).
    5. Морфологическое расширение (замыкание контуров).
    6. Заполнение масок (кластеризация связных компонент).

    Возвращает: словарь со всеми промежуточными результатами.
    """
    results = {}

    # 1. Grayscale
    gray = to_grayscale(image_rgb)
    results['grayscale'] = gray
    if verbose:
        print(f"    1. Grayscale: {gray.shape}, range [{gray.min():.3f}, {gray.max():.3f}]")

    # 2. Размытие
    k_blur = kernel_gaussian_blur(blur_size, blur_sigma)
    blurred = convolve2d_fast(gray, k_blur)
    results['blurred'] = blurred
    results['blur_kernel'] = k_blur
    if verbose:
        print(f"    2. Размытие: σ={blur_sigma}, ядро {blur_size}×{blur_size}")

    # 3. Выделение контуров
    if edge_method == 'gradient':
        edges = gradient_magnitude(blurred)
        method_name = 'Градиент (√(Gx²+Gy²))'
    elif edge_method == 'laplacian':
        edges = np.abs(convolve2d_fast(blurred, kernel_edge_detect_laplacian()))
        method_name = 'Лапласиан (4 направления)'
    elif edge_method == 'laplacian_full':
        edges = np.abs(convolve2d_fast(blurred, kernel_edge_detect_full()))
        method_name = 'Лапласиан (8 направлений)'
    elif edge_method == 'custom':
        edges = np.abs(convolve2d_fast(blurred, kernel_edge_detect_1()))
        method_name = 'Пользовательский (с доски)'
    else:
        edges = gradient_magnitude(blurred)
        method_name = 'Градиент'

    edges_norm = normalize_image(edges)
    results['edges'] = edges_norm
    results['edge_method'] = method_name
    if verbose:
        print(f"    3. Контуры: метод={method_name}")

    # 4. Бинаризация
    binary = binarize(edges_norm, threshold)
    results['binary'] = binary
    if verbose:
        n_edge = np.sum(binary > 0.5)
        print(f"    4. Бинаризация: порог={threshold}, пикселей контура={n_edge}")

    # 5. Морфология
    dilated = morphological_dilate(binary, dilate_radius)
    results['dilated'] = dilated
    if verbose:
        print(f"    5. Расширение: радиус={dilate_radius}")

    # 6. Кластеризация
    mask, n_objects = flood_fill_mask(dilated, min_area)
    results['mask'] = mask
    results['n_objects'] = n_objects
    if verbose:
        print(f"    6. Найдено объектов: {n_objects}")

    return results


# ============================================================
# 6. Визуализация
# ============================================================

def plot_pipeline(image_rgb, results, title=''):
    """Визуализация всех этапов пайплайна."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # 1. Исходное
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('1. Исходное изображение', fontweight='bold')
    axes[0, 0].axis('off')

    # 2. Grayscale
    axes[0, 1].imshow(results['grayscale'], cmap='gray')
    axes[0, 1].set_title('2. Оттенки серого', fontweight='bold')
    axes[0, 1].axis('off')

    # 3. Размытие
    axes[0, 2].imshow(results['blurred'], cmap='gray')
    axes[0, 2].set_title('3. Размытие (Гаусс)', fontweight='bold')
    axes[0, 2].axis('off')

    # 4. Контуры
    axes[0, 3].imshow(results['edges'], cmap='hot')
    axes[0, 3].set_title(f'4. Контуры ({results["edge_method"]})', fontweight='bold', fontsize=9)
    axes[0, 3].axis('off')

    # 5. Бинаризация
    axes[1, 0].imshow(results['binary'], cmap='gray')
    axes[1, 0].set_title('5. Бинаризация', fontweight='bold')
    axes[1, 0].axis('off')

    # 6. Морфология
    axes[1, 1].imshow(results['dilated'], cmap='gray')
    axes[1, 1].set_title('6. Морфологическое расширение', fontweight='bold')
    axes[1, 1].axis('off')

    # 7. Маска объектов
    axes[1, 2].imshow(results['mask'], cmap='nipy_spectral')
    axes[1, 2].set_title(f'7. Маска ({results["n_objects"]} объектов)', fontweight='bold')
    axes[1, 2].axis('off')

    # 8. Наложение маски на исходное
    overlay = image_rgb.copy().astype(np.float64)
    if overlay.max() > 1:
        overlay = overlay / 255.0
    mask_bool = results['mask'] > 0
    # Подсветка объектов
    overlay_vis = overlay.copy()
    overlay_vis[~mask_bool] *= 0.3  # затемняем фон
    contour = results['binary'] > 0.5
    overlay_vis[contour] = [1.0, 0.2, 0.2]  # красные контуры
    axes[1, 3].imshow(np.clip(overlay_vis, 0, 1))
    axes[1, 3].set_title('8. Результат (объекты + контуры)', fontweight='bold')
    axes[1, 3].axis('off')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


def plot_kernels():
    """Визуализация ядер свёртки."""
    kernels = {
        'Размытие (Гаусс σ=1)': kernel_gaussian_blur(5, 1.0),
        'Размытие (Box 3×3)': kernel_box_blur(3),
        'Резкость (Sharpen)': kernel_sharpen(),
        'Контуры (с доски)': kernel_edge_detect_1(),
        'Лапласиан (4 напр.)': kernel_edge_detect_laplacian(),
        'Лапласиан (8 напр.)': kernel_edge_detect_full(),
        '∂f/∂x (по горизонтали)': kernel_gradient_x(),
        '∂f/∂y (по вертикали)': kernel_gradient_y(),
    }

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, (name, k) in zip(axes.flatten(), kernels.items()):
        im = ax.imshow(k, cmap='RdBu_r', interpolation='nearest')
        ax.set_title(name, fontweight='bold', fontsize=10)
        # Подписи значений
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                val = k[i, j]
                txt = f'{val:.2f}' if abs(val) < 10 else f'{val:.0f}'
                color = 'white' if abs(val) > (k.max() - k.min()) * 0.4 else 'black'
                ax.text(j, i, txt, ha='center', va='center', fontsize=7, color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle('Ядра свёртки (оконные матрицы) — коэффициенты C_m',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_filter_effects(image_rgb):
    """Сравнение эффектов разных фильтров."""
    gray = to_grayscale(image_rgb)

    filters = {
        'Исходное': gray,
        'Размытие (Гаусс)': convolve2d_fast(gray, kernel_gaussian_blur(5, 1.5)),
        'Резкость (Sharpen)': convolve2d_fast(gray, kernel_sharpen()),
        'Контуры (с доски)': np.abs(convolve2d_fast(gray, kernel_edge_detect_1())),
        'Лапласиан': np.abs(convolve2d_fast(gray, kernel_edge_detect_laplacian())),
        'Лапласиан (полный)': np.abs(convolve2d_fast(gray, kernel_edge_detect_full())),
        '∂f/∂x (по горизонтали)': np.abs(convolve2d_fast(gray, kernel_gradient_x())),
        'Градиент |∇f|': gradient_magnitude(gray),
    }

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, (name, img) in zip(axes.flatten(), filters.items()):
        ax.imshow(normalize_image(img), cmap='gray')
        ax.set_title(name, fontweight='bold')
        ax.axis('off')

    fig.suptitle('Сравнение эффектов различных фильтров', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_threshold_comparison(image_rgb, edge_method='gradient'):
    """Сравнение разных порогов бинаризации."""
    gray = to_grayscale(image_rgb)
    blurred = convolve2d_fast(gray, kernel_gaussian_blur(5, 1.5))

    if edge_method == 'gradient':
        edges = normalize_image(gradient_magnitude(blurred))
    else:
        edges = normalize_image(np.abs(convolve2d_fast(blurred, kernel_edge_detect_full())))

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, thr in zip(axes.flatten(), thresholds):
        binary = binarize(edges, thr)
        ax.imshow(binary, cmap='gray')
        n_px = np.sum(binary > 0.5)
        ax.set_title(f'Порог = {thr:.2f} ({n_px} пикс.)', fontweight='bold')
        ax.axis('off')

    fig.suptitle('Сравнение порогов бинаризации', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_edge_methods_comparison(image_rgb):
    """Сравнение методов выделения контуров."""
    methods = ['gradient', 'laplacian', 'laplacian_full', 'custom']
    method_names = ['Градиент |∇f|', 'Лапласиан (4)', 'Лапласиан (8)', 'С доски']

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for idx, (method, name) in enumerate(zip(methods, method_names)):
        res = detect_objects(image_rgb, edge_method=method, verbose=False)
        axes[0, idx].imshow(res['edges'], cmap='hot')
        axes[0, idx].set_title(f'Контуры: {name}', fontweight='bold', fontsize=10)
        axes[0, idx].axis('off')

        axes[1, idx].imshow(res['mask'], cmap='nipy_spectral')
        axes[1, idx].set_title(f'Маска: {name} ({res["n_objects"]} объектов)',
                               fontweight='bold', fontsize=10)
        axes[1, idx].axis('off')

    fig.suptitle('Сравнение методов выделения контуров', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def main():
    print("=" * 80)
    print("КЕЙС №6 — МАШИННОЕ ЗРЕНИЕ")
    print("Выделение объектов на изображении методом фильтрации")
    print("=" * 80)

    os.makedirs('plots', exist_ok=True)

    # ================================================================
    # ЧАСТЬ 1: Генерация тестовых изображений
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 1: ТЕСТОВЫЕ ИЗОБРАЖЕНИЯ")
    print("=" * 80)

    test_images = {
        'Простой (контрастный фон)': generate_test_simple(200),
        'Средний (градиентный фон)': generate_test_medium(200),
        'Сложный (шум, низкий контраст)': generate_test_hard(200),
    }

    # Загрузка реального изображения, если есть
    uploads = '/mnt/user-data/uploads'
    real_image = None
    if os.path.exists(uploads):
        for f in os.listdir(uploads):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                if 'photo' not in f.lower() and 'rbm' not in f.lower():
                    try:
                        img = np.array(Image.open(os.path.join(uploads, f)).resize((300, 300))) / 255.0
                        real_image = img
                        test_images['Реальное фото'] = img
                        print(f"  Загружено реальное изображение: {f}")
                        break
                    except:
                        pass

    for name in test_images:
        img = test_images[name]
        print(f"  {name}: {img.shape}, range [{img.min():.3f}, {img.max():.3f}]")

    # ================================================================
    # ЧАСТЬ 2: Визуализация ядер свёртки
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 2: ЯДРА СВЁРТКИ (ОКОННЫЕ МАТРИЦЫ)")
    print("=" * 80)

    fig_kernels = plot_kernels()
    fig_kernels.savefig('plots/cv_kernels.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: plots/cv_kernels.png")
    plt.close(fig_kernels)

    # ================================================================
    # ЧАСТЬ 3: Сравнение эффектов фильтров
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 3: СРАВНЕНИЕ ЭФФЕКТОВ ФИЛЬТРОВ")
    print("=" * 80)

    fig_effects = plot_filter_effects(test_images['Простой (контрастный фон)'])
    fig_effects.savefig('plots/cv_filter_effects.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: plots/cv_filter_effects.png")
    plt.close(fig_effects)

    # ================================================================
    # ЧАСТЬ 4: Пайплайн на разных изображениях
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 4: ПАЙПЛАЙН ВЫДЕЛЕНИЯ ОБЪЕКТОВ")
    print("=" * 80)

    all_results = {}

    # Индивидуальные параметры для каждого уровня сложности
    params_per_level = {
        'Простой (контрастный фон)': dict(
            blur_sigma=1.0, blur_size=3, edge_method='gradient',
            threshold=0.25, dilate_radius=1, min_area=50),
        'Средний (градиентный фон)': dict(
            blur_sigma=2.0, blur_size=5, edge_method='gradient',
            threshold=0.20, dilate_radius=2, min_area=40),
        'Сложный (шум, низкий контраст)': dict(
            blur_sigma=3.0, blur_size=7, edge_method='gradient',
            threshold=0.25, dilate_radius=3, min_area=80),
    }
    default_params = dict(
        blur_sigma=1.5, blur_size=5, edge_method='gradient',
        threshold=0.20, dilate_radius=2, min_area=40)

    for name, img in test_images.items():
        print(f"\n  --- {name} ---")
        p = params_per_level.get(name, default_params)
        results = detect_objects(img, **p)
        all_results[name] = results

        fig = plot_pipeline(img, results, title=f'Пайплайн: {name}')
        fname = f'plots/cv_pipeline_{name.split("(")[0].strip().lower().replace(" ", "_")}.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"    Сохранено: {fname}")
        plt.close(fig)

    # ================================================================
    # ЧАСТЬ 5: Сравнение методов выделения контуров
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 5: СРАВНЕНИЕ МЕТОДОВ ВЫДЕЛЕНИЯ КОНТУРОВ")
    print("=" * 80)

    fig_methods = plot_edge_methods_comparison(test_images['Простой (контрастный фон)'])
    fig_methods.savefig('plots/cv_edge_methods.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: plots/cv_edge_methods.png")
    plt.close(fig_methods)

    # ================================================================
    # ЧАСТЬ 6: Сравнение порогов бинаризации
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 6: СРАВНЕНИЕ ПОРОГОВ БИНАРИЗАЦИИ")
    print("=" * 80)

    fig_thr = plot_threshold_comparison(test_images['Средний (градиентный фон)'])
    fig_thr.savefig('plots/cv_thresholds.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: plots/cv_thresholds.png")
    plt.close(fig_thr)

    # ================================================================
    # ЧАСТЬ 7: Эксперимент с параметрами размытия
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 7: ВЛИЯНИЕ ПАРАМЕТРОВ РАЗМЫТИЯ")
    print("=" * 80)

    sigmas = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    img_test = test_images['Средний (градиентный фон)']
    gray_test = to_grayscale(img_test)

    for ax, sigma in zip(axes.flatten(), sigmas):
        blurred = convolve2d_fast(gray_test, kernel_gaussian_blur(
            max(3, int(sigma * 4) | 1), sigma))
        edges = normalize_image(gradient_magnitude(blurred))
        ax.imshow(edges, cmap='hot')
        ax.set_title(f'σ = {sigma}', fontweight='bold')
        ax.axis('off')
        print(f"  σ={sigma:.1f}: max_edge={edges.max():.3f}")

    fig.suptitle('Влияние σ размытия на выделение контуров', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig('plots/cv_blur_sigma.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: plots/cv_blur_sigma.png")
    plt.close(fig)

    # --- Сохранение результатов ---
    results_summary = {
        'параметры_по_уровням': {},
        'результаты': {}
    }
    for name, res in all_results.items():
        p = params_per_level.get(name, default_params)
        results_summary['параметры_по_уровням'][name] = {
            k: v for k, v in p.items()
        }
        results_summary['результаты'][name] = {
            'n_objects': int(res['n_objects']),
            'edge_method': res['edge_method']
        }

    with open('plots/results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print("\n  Сохранено: plots/results.json")

    print("\n" + "=" * 80)
    print("ГОТОВО!")
    print("=" * 80)


if __name__ == "__main__":
    main()
