"""
Кейс №6 — Машинное зрение: Оконно-матричное преобразование
Выделение контуров объектов и построение бинарных масок (фильтрация).

Основные этапы (по лекции):
  1. Оконно-матричное преобразование (Convolution) — замена пикселя
     на взвешенную сумму его соседей (матрица/ядро).
  2. Размытие (Blur) — усреднение для подавления шума.
  3. Выделение контуров (Edge Detection) — фильтры на основе
     конечных разностей (аналог производной), например, оператор Лапласа.
  4. Построение маски — пороговое преобразование (Thresholding),
     где контур/объект = 1, а фон = 0.
  5. Тестирование на палитре от простого к сложному.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10


# ============================================================
# 1. Библиотека фильтров (Ядра свертки / Матрицы)
# ============================================================

class Kernels:
    """Набор базовых матриц для оконного преобразования."""

    @staticmethod
    def gaussian_blur_3x3():
        """Размытие по Гауссу (приближенное)."""
        kernel = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=float)
        return kernel / np.sum(kernel)

    @staticmethod
    def box_blur_5x5():
        """Обычное усреднение (сильное размытие)."""
        kernel = np.ones((5, 5), dtype=float)
        return kernel / np.sum(kernel)

    @staticmethod
    def laplacian():
        """Выделение контуров (производная по всем направлениям)."""
        return np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], dtype=float)

    @staticmethod
    def sobel_x():
        """Производная по оси X (вертикальные контуры)."""
        return np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=float)

    @staticmethod
    def sobel_y():
        """Производная по оси Y (горизонтальные контуры)."""
        return np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=float)


# ============================================================
# 2. Оконно-матричное преобразование (Свертка)
# ============================================================

def apply_filter(image, kernel):
    """
    Применяет оконно-матричное преобразование к изображению.
    Математика (с доски):
      Новое_значение(x, y) = Σ Σ ( Image(x+i, y+j) * Kernel(i, j) )
    """
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    # Добавляем "паддинг" (отступы), чтобы матрица не выходила за границы
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    out_img = np.zeros_like(image, dtype=float)

    # Проход окном по каждому пикселю исходного изображения
    # В реальных задачах используется векторизация, но здесь реализовано
    # близко к классическому циклическому пониманию для наглядности
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Вырезаем окно
            window = padded_img[i: i + k_h, j: j + k_w]
            # Умножение матриц поэлементно и сумма
            out_img[i, j] = np.sum(window * kernel)

    return out_img


def build_mask(edge_image, threshold=0.2):
    """
    Превращает карту контуров в бинарную маску (1 - контур, 0 - фон).
    Ожидается, что edge_image нормализован от 0 до 1.
    """
    mask = np.zeros_like(edge_image)
    mask[edge_image > threshold] = 1.0
    return mask


def normalize(image):
    """Нормирует значения изображения в диапазон [0, 1]."""
    dmin, dmax = np.min(image), np.max(image)
    if dmax - dmin < 1e-12:
        return image
    return (image - dmin) / (dmax - dmin)


# ============================================================
# 3. Генерация тестовых данных ("Палитра примеров")
# ============================================================

def generate_test_images(size=100):
    """
    Генерирует словарь с тестовыми изображениями (от простого к сложному),
    как просил преподаватель.
    """
    images = {}

    # 1. Простая фигура на простом фоне (Четкий контраст)
    img_simple = np.zeros((size, size))
    cv_x, cv_y = size // 2, size // 2
    y, x = np.ogrid[-cv_y:size - cv_y, -cv_x:size - cv_x]
    mask = x ** 2 + y ** 2 <= (size // 4) ** 2
    img_simple[mask] = 1.0
    images['Простой контраст'] = img_simple

    # 2. Сложная фигура на простом фоне
    img_complex = np.zeros((size, size))
    # Рисуем несколько пересекающихся кругов и квадратов
    img_complex[20:50, 20:50] = 1.0
    img_complex[40:80, 60:80] = 0.8
    mask2 = (x - 20) ** 2 + (y + 10) ** 2 <= 20 ** 2
    img_complex[mask2] = 0.6
    images['Сложный объект'] = img_complex

    # 3. Зашумленный фон (Имитация плохого качества / текстуры)
    noise = np.random.normal(0, 0.2, (size, size))
    img_noisy_bg = np.clip(img_simple + noise, 0, 1)
    images['Зашумленный фон'] = img_noisy_bg

    # 4. Градиентный фон с малоконтрастным объектом
    img_grad = np.tile(np.linspace(0, 0.8, size), (size, 1))
    img_low_contrast = img_grad.copy()
    img_low_contrast[40:60, 40:60] = img_grad[40:60, 40:60] + 0.15  # Очень слабый перепад
    images['Слабый контраст'] = np.clip(img_low_contrast, 0, 1)

    return images


# ============================================================
# 4. Основной пайплайн выделения контуров
# ============================================================

def process_image(img):
    """
    Комплексная обработка одного изображения:
    1. Легкое размытие (чтобы убрать шум).
    2. Выделение контуров (Лапласиан).
    3. Нормализация.
    4. Построение бинарной маски.
    """
    # 1. Сглаживание шума (размытие по Гауссу)
    blurred = apply_filter(img, Kernels.gaussian_blur_3x3())

    # 2. Поиск производных (контуров)
    edges_raw = apply_filter(blurred, Kernels.laplacian())

    # Берем модуль, так как перепад может быть как в +, так и в -
    edges_abs = np.abs(edges_raw)

    # 3. Нормализация контуров
    edges_norm = normalize(edges_abs)

    # 4. Построение маски по порогу (подбирается эвристически)
    # Для зашумленных данных порог должен быть выше
    threshold = 0.25
    mask = build_mask(edges_norm, threshold)

    return blurred, edges_norm, mask


# ============================================================
# 5. Визуализация
# ============================================================

def plot_results(images_dict):
    """Строит сетку графиков для всей палитры примеров."""
    n = len(images_dict)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))

    for i, (title, img) in enumerate(images_dict.items()):
        blurred, edges, mask = process_image(img)

        # 1. Исходник
        ax = axes[i, 0]
        im0 = ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'1. Исходное: {title}', fontweight='bold')
        ax.axis('off')

        # 2. Размытие
        ax = axes[i, 1]
        im1 = ax.imshow(blurred, cmap='gray')
        ax.set_title('2. Размытие (усреднение)', fontweight='bold')
        ax.axis('off')

        # 3. Контуры (Производная)
        ax = axes[i, 2]
        im2 = ax.imshow(edges, cmap='hot')
        ax.set_title('3. Градиенты (Матрица контуров)', fontweight='bold')
        ax.axis('off')

        # 4. Итоговая маска
        ax = axes[i, 3]
        im3 = ax.imshow(mask, cmap='gray')
        ax.set_title('4. Бинарная маска (1 - контур, 0 - фон)', fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    return fig


def plot_kernels_demo(img):
    """Демонстрация различных матриц (ядер) на одном изображении."""
    kernels = {
        'Оригинал': None,
        'Размытие (Гаусс 3x3)': Kernels.gaussian_blur_3x3(),
        'Размытие (Box 5x5)': Kernels.box_blur_5x5(),
        'Лапласиан (Все контуры)': Kernels.laplacian(),
        'Собель X (Вертикальные)': Kernels.sobel_x(),
        'Собель Y (Горизонтальные)': Kernels.sobel_y()
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (name, kernel) in enumerate(kernels.items()):
        ax = axes[idx]
        if kernel is None:
            ax.imshow(img, cmap='gray')
        else:
            res = apply_filter(img, kernel)
            # Для контурных фильтров показываем модуль, для размытия - оригинал
            if 'Размытие' not in name:
                res = normalize(np.abs(res))
            ax.imshow(res, cmap='gray' if 'Размытие' in name else 'inferno')

        ax.set_title(name, fontweight='bold', fontsize=12)
        ax.axis('off')

    plt.suptitle('Сравнение различных оконных матриц (Ядер свертки)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================
# 6. Главная функция
# ============================================================

def main():
    print("=" * 80)
    print("КЕЙС №6: ВЫДЕЛЕНИЕ КОНТУРОВ И ОКОННО-МАТРИЧНОЕ ПРЕОБРАЗОВАНИЕ")
    print("=" * 80)

    os.makedirs("plots", exist_ok=True)
    np.random.seed(42)

    print("\nГенерация палитры тестовых изображений...")
    test_images = generate_test_images(size=120)
    print(f"Сгенерировано объектов: {len(test_images)}")
    for k in test_images.keys():
        print(f"  - {k}")

    print("\nПрименение оконно-матричных преобразований и расчет производных...")

    # 1. Построение основного отчета (от простого к сложному)
    fig_main = plot_results(test_images)
    fig_main.savefig('plots/case6_edge_detection.png', dpi=150, bbox_inches='tight')
    print("Сохранено: plots/case6_edge_detection.png")
    plt.close(fig_main)

    # 2. Построение отчета со сравнением разных матриц на сложном объекте
    complex_img = test_images['Сложный объект']
    fig_demo = plot_kernels_demo(complex_img)
    fig_demo.savefig('plots/case6_kernels_demo.png', dpi=150, bbox_inches='tight')
    print("Сохранено: plots/case6_kernels_demo.png")
    plt.close(fig_demo)

    print("\n" + "=" * 80)
    print("ГОТОВО! Результаты сохранены в папке 'plots'.")
    print("=" * 80)


if __name__ == "__main__":
    main()
