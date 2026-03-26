"""
РГР: Разложение произвольной замкнутой двумерной кривой
на совокупность простых кривых (парабол) методом регрессии.

Подход v2 — итеративная сегментация + прямой фит:
1. Равномерно разбиваем контур на N сегментов.
2. Для каждого сегмента фитим параболу (МНК в локальной СК).
3. Перераспределяем точки: каждая точка контура «уходит» к ближайшей параболе.
4. Заново фитим параболы по обновлённым сегментам.
5. Повторяем (как k-means, но для кривых).
6. Финальная полировка — scipy.optimize.

y(x) = Ax² + Bx + C  или  x(y) = A'y² + B'y + C'
(обобщённая параметрическая форма с произвольным поворотом)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. Генерация тестовых контуров
# ============================================================

def generate_star(n_points=600, n_tips=5, r_inner=0.4, r_outer=1.0):
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    r = r_inner + (r_outer - r_inner) * (0.5 + 0.5 * np.cos(n_tips * t))
    return r * np.cos(t), r * np.sin(t)


def generate_blob(n_points=600, seed=42):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    r = 1.0
    for k in range(2, 8):
        r = r + rng.uniform(0.05, 0.2) * np.cos(k * t + rng.uniform(0, 2 * np.pi))
    return r * np.cos(t), r * np.sin(t)


def generate_heart(n_points=600):
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return x / 17, y / 17


def generate_cat_silhouette(n_points=600):
    """Контур 'кошки' — более сложная форма с ушами."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    # Базовый овал (тело)
    r = 0.6 + 0.15 * np.cos(2 * t)
    # Уши — острые пики сверху
    ear_l = 0.35 * np.exp(-((t - 2.3) ** 2) / 0.02)
    ear_r = 0.35 * np.exp(-((t - 1.0) ** 2) / 0.02)
    # Хвост — выступ справа-снизу
    tail = 0.25 * np.exp(-((t - 5.0) ** 2) / 0.05)
    r = r + ear_l + ear_r + tail
    return r * np.cos(t), r * np.sin(t)


def generate_random_complex_contour(n_points=600, max_freq=12, seed=None):
    """
    Генерирует случайный сложный замкнутый контур без самопересечений.
    Использует сумму тригонометрических гармоник в полярных координатах.

    max_freq: максимальная частота гармоник (влияет на "изрезанность").
    seed: для воспроизводимости (если нужно).
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # Базовый радиус
    r = np.ones_like(t) * 1.5

    # Накладываем случайные гармоники
    for k in range(1, max_freq + 1):
        # Амплитуда затухает с ростом частоты, чтобы контур не превратился в "шум"
        # Меняя делитель (например, k**0.8 или k**1.2), можно регулировать гладкость
        amplitude = rng.uniform(0.1, 0.5) / (k ** 0.8)
        phase = rng.uniform(0, 2 * np.pi)

        # Случайно выбираем синус или косинус
        if rng.choice([True, False]):
            r += amplitude * np.cos(k * t + phase)
        else:
            r += amplitude * np.sin(k * t + phase)

    # Защита от прохождения через центр (и образования "петель")
    # Радиус всегда должен быть строго положительным
    r = np.clip(r, 0.2, None)

    x = r * np.cos(t)
    y = r * np.sin(t)

    # Центрируем контур по массе
    x -= np.mean(x)
    y -= np.mean(y)

    return x, y


# ============================================================
# 2. Параболическая кривая (обобщённая)
# ============================================================

class ParabolaCurve:
    """
    Парабола в 2D с произвольной ориентацией.
    В локальной системе координат: v = A*u² + B*u + C
    Глобальные координаты получаются поворотом на angle и сдвигом на (cx, cy).
    
    Параметры: [A, B, C, cx, cy, angle, u_min, u_max]
    """
    N_PARAMS = 8

    def __init__(self, params=None):
        if params is None:
            params = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0])
        self.params = np.array(params, dtype=float)

    @property
    def A(self): return self.params[0]
    @property
    def B(self): return self.params[1]
    @property
    def C(self): return self.params[2]
    @property
    def cx(self): return self.params[3]
    @property
    def cy(self): return self.params[4]
    @property
    def angle(self): return self.params[5]
    @property
    def u_min(self): return self.params[6]
    @property
    def u_max(self): return self.params[7]

    def eval_points(self, n=300):
        """Вернуть (x, y) точки вдоль параболы."""
        u = np.linspace(self.u_min, self.u_max, n)
        v = self.A * u**2 + self.B * u + self.C
        ca, sa = np.cos(self.angle), np.sin(self.angle)
        x = self.cx + ca * u - sa * v
        y = self.cy + sa * u + ca * v
        return x, y

    def min_dist_to(self, px, py, n=300):
        """Мин. расстояние от каждой (px[i], py[i]) до ближайшей точки параболы."""
        cx, cy = self.eval_points(n)
        # Векторизованно
        pts_curve = np.column_stack([cx, cy])  # (n, 2)
        pts_query = np.column_stack([px, py])  # (m, 2)
        D = cdist(pts_query, pts_curve)  # (m, n)
        return np.min(D, axis=1)

    @staticmethod
    def fit_to_segment(sx, sy):
        """
        Напрямую фитим параболу к набору точек (sx, sy).
        1. PCA для определения главного направления.
        2. Проецируем на локальную ось.
        3. Фитим полином 2-й степени.
        """
        cx, cy = np.mean(sx), np.mean(sy)
        pts = np.column_stack([sx - cx, sy - cy])

        if len(pts) < 3:
            return ParabolaCurve(np.array([0, 0, 0, cx, cy, 0, -0.5, 0.5]))

        # PCA
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        main = eigvecs[:, 1]  # наибольшая дисперсия
        angle = np.arctan2(main[1], main[0])

        ca, sa = np.cos(angle), np.sin(angle)
        u = ca * (sx - cx) + sa * (sy - cy)
        v = -sa * (sx - cx) + ca * (sy - cy)

        # МНК: v = A*u² + B*u + C
        coeffs = np.polyfit(u, v, 2)
        A, B, C = coeffs

        u_min, u_max = np.min(u), np.max(u)
        # Немного расширим
        margin = 0.05 * (u_max - u_min + 1e-8)
        u_min -= margin
        u_max += margin

        return ParabolaCurve(np.array([A, B, C, cx, cy, angle, u_min, u_max]))


# ============================================================
# 3. Итеративная сегментация (основной алгоритм)
# ============================================================

def assign_points_to_parabolas(contour_x, contour_y, parabolas, n_eval=400):
    """Каждую точку контура приписываем к ближайшей параболе."""
    n_pts = len(contour_x)
    n_par = len(parabolas)

    dist_matrix = np.zeros((n_pts, n_par))
    for j, p in enumerate(parabolas):
        dist_matrix[:, j] = p.min_dist_to(contour_x, contour_y, n_eval)

    assignment = np.argmin(dist_matrix, axis=1)
    min_dists = np.min(dist_matrix, axis=1)
    return assignment, min_dists, dist_matrix


def iterative_fit(contour_x, contour_y, n_parabolas=12, n_iter=30, verbose=True):
    """
    K-means-подобная итеративная процедура:
    1) Начальная равномерная сегментация контура.
    2) Фит парабол к сегментам.
    3) Перераспределение точек к ближайшим параболам.
    4) Повтор до сходимости.
    """
    n = len(contour_x)
    seg_size = n // n_parabolas

    # ---- Начальная сегментация (по порядку вдоль контура) ----
    assignment = np.zeros(n, dtype=int)
    for i in range(n_parabolas):
        start = i * seg_size
        end = (i + 1) * seg_size if i < n_parabolas - 1 else n
        assignment[start:end] = i

    prev_loss = np.inf
    parabolas = [None] * n_parabolas
    history = []

    for iteration in range(n_iter):
        # ---- Фит парабол по текущим сегментам ----
        for j in range(n_parabolas):
            mask = assignment == j
            if np.sum(mask) < 4:
                # Слишком мало точек — берём ближайшие к центру этой параболы
                if parabolas[j] is not None:
                    continue
                # Или просто берём случайный кусок
                idx = np.random.choice(n, size=max(10, seg_size // 2), replace=False)
                mask = np.zeros(n, dtype=bool)
                mask[idx] = True

            sx = contour_x[mask]
            sy = contour_y[mask]
            parabolas[j] = ParabolaCurve.fit_to_segment(sx, sy)

        # ---- Перераспределение точек ----
        assignment, min_dists, _ = assign_points_to_parabolas(
            contour_x, contour_y, parabolas, n_eval=400
        )

        loss = np.mean(min_dists ** 2)
        history.append(loss)

        if verbose and (iteration < 5 or iteration % 5 == 0 or iteration == n_iter - 1):
            mean_d = np.mean(min_dists)
            max_d = np.max(min_dists)
            pct = np.mean(min_dists < 0.03) * 100
            print(f"  Итерация {iteration + 1:3d}: MSE={loss:.6f}, "
                  f"mean_dist={mean_d:.4f}, max_dist={max_d:.4f}, <0.03: {pct:.1f}%")

        # Проверка сходимости
        if abs(prev_loss - loss) < 1e-9 and iteration > 5:
            if verbose:
                print(f"  Сходимость достигнута на итерации {iteration + 1}")
            break
        prev_loss = loss

    return parabolas, assignment, history


# ============================================================
# 4. Финальная полировка (fine-tuning) через least_squares
# ============================================================

def fine_tune(contour_x, contour_y, parabolas, max_nfev=3000, verbose=True):
    """
    Полировка: оптимизируем каждую параболу отдельно по «своим» точкам.
    Гораздо быстрее, чем совместная оптимизация.
    """
    if verbose:
        print("  Финальная полировка (по-парабольно)...")

    # Определяем принадлежность точек
    assignment, _, _ = assign_points_to_parabolas(contour_x, contour_y, parabolas, 300)

    optimized = []
    for j, p in enumerate(parabolas):
        mask = assignment == j
        if np.sum(mask) < 4:
            optimized.append(p)
            continue

        sx, sy = contour_x[mask], contour_y[mask]

        def residuals_single(params):
            pc = ParabolaCurve(params)
            return pc.min_dist_to(sx, sy, 200)

        result = least_squares(residuals_single, p.params, method='trf',
                               max_nfev=max(200, max_nfev // len(parabolas)),
                               ftol=1e-10, xtol=1e-10, verbose=0)
        optimized.append(ParabolaCurve(result.x))

    if verbose:
        dists = np.full(len(contour_x), np.inf)
        for p in optimized:
            d = p.min_dist_to(contour_x, contour_y, 300)
            dists = np.minimum(dists, d)
        print(f"  После полировки: mean_dist={np.mean(dists):.5f}, "
              f"max_dist={np.max(dists):.5f}, "
              f"<0.03: {np.mean(dists < 0.03) * 100:.1f}%")

    return optimized, None


# ============================================================
# 5. Полный пайплайн
# ============================================================

def decompose_contour(contour_x, contour_y, n_parabolas=12,
                      n_iter=40, fine_tune_nfev=5000, verbose=True):
    """Полный пайплайн разложения контура на параболы."""
    if verbose:
        print(f"\n  Этап 1: Итеративная сегментация ({n_iter} итераций, {n_parabolas} парабол)")
    parabolas, assignment, history = iterative_fit(
        contour_x, contour_y, n_parabolas, n_iter, verbose
    )

    if verbose:
        print(f"\n  Этап 2: Fine-tuning")
    parabolas, opt_result = fine_tune(
        contour_x, contour_y, parabolas, fine_tune_nfev, verbose
    )

    return parabolas, history


# ============================================================
# 6. Метрики
# ============================================================

def compute_metrics(contour_x, contour_y, parabolas):
    n = len(contour_x)
    dists = np.full(n, np.inf)
    for p in parabolas:
        d = p.min_dist_to(contour_x, contour_y, 500)
        dists = np.minimum(dists, d)

    return {
        'mean': np.mean(dists),
        'max': np.max(dists),
        'median': np.median(dists),
        'std': np.std(dists),
        'p90': np.percentile(dists, 90),
        'p95': np.percentile(dists, 95),
        'pct_003': np.mean(dists < 0.03) * 100,
        'pct_005': np.mean(dists < 0.05) * 100,
        'pct_01': np.mean(dists < 0.1) * 100,
    }


def print_metrics(m, title=""):
    print(f"\n{'=' * 55}")
    print(f"  Метрики: {title}")
    print(f"{'=' * 55}")
    print(f"  Средняя ошибка:        {m['mean']:.5f}")
    print(f"  Медиана:               {m['median']:.5f}")
    print(f"  Макс. ошибка:          {m['max']:.5f}")
    print(f"  90-перцентиль:         {m['p90']:.5f}")
    print(f"  95-перцентиль:         {m['p95']:.5f}")
    print(f"  Точек с d<0.03:        {m['pct_003']:.1f}%")
    print(f"  Точек с d<0.05:        {m['pct_005']:.1f}%")
    print(f"  Точек с d<0.10:        {m['pct_01']:.1f}%")
    print(f"{'=' * 55}")


# ============================================================
# 7. Визуализация
# ============================================================

def plot_decomposition(contour_x, contour_y, parabolas, title="", save_path=None):
    """Три панели: контур, параболы, ошибка."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    cx_closed = np.append(contour_x, contour_x[0])
    cy_closed = np.append(contour_y, contour_y[0])

    colors = plt.cm.tab20(np.linspace(0, 1, len(parabolas)))

    # --- 1. Исходный контур ---
    ax = axes[0]
    ax.plot(cx_closed, cy_closed, 'k-', linewidth=2.5)
    ax.set_title('Исходный контур', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # --- 2. Параболы ---
    ax = axes[1]
    ax.plot(cx_closed, cy_closed, 'k-', linewidth=1, alpha=0.25)
    for i, p in enumerate(parabolas):
        px, py = p.eval_points(500)
        ax.plot(px, py, '-', color=colors[i], linewidth=3.0,
                label=f'P{i + 1}', solid_capstyle='round')
    ax.set_title(f'Разложение на {len(parabolas)} парабол', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    if len(parabolas) <= 15:
        ax.legend(fontsize=7, ncol=3, loc='best')

    # --- 3. Карта ошибки ---
    ax = axes[2]
    dists = np.full(len(contour_x), np.inf)
    for p in parabolas:
        d = p.min_dist_to(contour_x, contour_y, 500)
        dists = np.minimum(dists, d)

    vmax = np.percentile(dists, 95)
    sc = ax.scatter(contour_x, contour_y, c=dists, cmap='RdYlGn_r',
                    s=8, vmin=0, vmax=max(vmax, 0.01), edgecolors='none')
    for i, p in enumerate(parabolas):
        px, py = p.eval_points(300)
        ax.plot(px, py, '-', color=colors[i], linewidth=1.5, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Расстояние до ближ. параболы')
    ax.set_title('Карта ошибки', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=17, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_convergence_history(history, title="", save_path=None):
    """График сходимости итеративного процесса."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(history) + 1), history, 'b-o', markersize=4)
    ax.set_xlabel('Итерация', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title(f'Сходимость: {title}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================
# 8. Запуск экспериментов
# ============================================================

def run_experiment(name, cx, cy, n_parabolas, n_iter=40, fine_nfev=5000):
    print(f"\n{'#' * 60}")
    print(f"  {name}")
    print(f"  Точек: {len(cx)}, Парабол: {n_parabolas}")
    print(f"{'#' * 60}")

    parabolas, history = decompose_contour(
        cx, cy, n_parabolas, n_iter, fine_nfev, verbose=True
    )

    metrics = compute_metrics(cx, cy, parabolas)
    print_metrics(metrics, name)

    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
    fig1 = plot_decomposition(cx, cy, parabolas, title=name,
                              save_path=f'{safe_name}_result.png')
    fig2 = plot_convergence_history(history, title=name,
                                    save_path=f'{safe_name}_conv.png')
    plt.close('all')

    return parabolas, metrics


if __name__ == '__main__':
    print("=" * 60)
    print("  РГР: Разложение замкнутой кривой на параболы")
    print("  Метод: итеративная сегментация + регрессия")
    print("=" * 60)

    all_metrics = {}

    # --- Тест 1: Звезда ---
    cx, cy = generate_star(600, n_tips=5)
    _, m = run_experiment("Звезда 5 лучей", cx, cy, n_parabolas=15, n_iter=40)
    all_metrics['Звезда'] = m

    # --- Тест 2: Блоб ---
    cx, cy = generate_blob(600)
    _, m = run_experiment("Амёба blob", cx, cy, n_parabolas=12, n_iter=40)
    all_metrics['Амёба'] = m

    # --- Тест 3: Сердце ---
    cx, cy = generate_heart(600)
    _, m = run_experiment("Сердце", cx, cy, n_parabolas=12, n_iter=40)
    all_metrics['Сердце'] = m

    # --- Тест 4: Кошка ---
    cx, cy = generate_cat_silhouette(600)
    _, m = run_experiment("Кошка силуэт", cx, cy, n_parabolas=14, n_iter=40)
    all_metrics['Кошка'] = m

    # --- Тест 5: Случайный сложный контур ---
    cx, cy = generate_random_complex_contour(600, max_freq=10)
    # n_parabolas берем побольше, так как фигура сложная
    _, m = run_experiment("Случайный контур", cx, cy, n_parabolas=18, n_iter=50)
    all_metrics['Случайный'] = m

    # === СВОДНАЯ ТАБЛИЦА ===
    print(f"\n\n{'=' * 75}")
    print("  СВОДНАЯ ТАБЛИЦА")
    print(f"{'=' * 75}")
    print(f"{'Контур':<12} {'Mean':>8} {'Median':>8} {'Max':>8} "
          f"{'P90':>8} {'<0.03':>7} {'<0.05':>7} {'<0.10':>7}")
    print("-" * 75)
    for name, m in all_metrics.items():
        print(f"{name:<12} {m['mean']:>8.5f} {m['median']:>8.5f} {m['max']:>8.5f} "
              f"{m['p90']:>8.5f} {m['pct_003']:>6.1f}% {m['pct_005']:>6.1f}% {m['pct_01']:>6.1f}%")
    print(f"{'=' * 75}")

    # Копируем в outputs
    import shutil, glob
    for f in glob.glob('*_result.png') + glob.glob('*_conv.png'):
        shutil.copy(f, 'outputs/')

    print("\nГотово! Файлы сохранены.")