"""
РГР: Разложение произвольной замкнутой двумерной кривой
на совокупность простых кривых (парабол) методом регрессии.

Подход: Итеративная сегментация + собственный градиентный спуск (Adam)
Полностью независимая реализация без использования SciPy.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# 0. Самописные математические функции (вместо SciPy)
# ============================================================

def custom_cdist(pts_query, pts_curve):
    """
    Вычисляет матрицу Евклидовых расстояний между двумя наборами точек.
    Вместо scipy.spatial.distance.cdist.
    """
    diff = pts_query[:, np.newaxis, :] - pts_curve[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def custom_polyfit_2d(u, v):
    """
    Решает задачу регрессии (МНК) аналитически через матричные операции.
    Вместо np.polyfit.
    Ищем коэффициенты W = [A, B, C] для уравнения v = A*u^2 + B*u + C.
    """
    X = np.column_stack([u ** 2, u, np.ones_like(u)])
    # W = (X^T * X)^(-1) * X^T * v
    W = np.linalg.solve(X.T @ X, X.T @ v)
    return W


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
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    r = 0.6 + 0.15 * np.cos(2 * t)
    ear_l = 0.35 * np.exp(-((t - 2.3) ** 2) / 0.02)
    ear_r = 0.35 * np.exp(-((t - 1.0) ** 2) / 0.02)
    tail = 0.25 * np.exp(-((t - 5.0) ** 2) / 0.05)
    r = r + ear_l + ear_r + tail
    return r * np.cos(t), r * np.sin(t)


def generate_random_complex_contour(n_points=600, max_freq=10, seed=None):
    """Случайный сложный контур на основе усеченного ряда Фурье."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    r = np.ones_like(t) * 1.5
    for k in range(1, max_freq + 1):
        amplitude = rng.uniform(0.1, 0.4) / (k ** 0.8)
        phase = rng.uniform(0, 2 * np.pi)
        if rng.choice([True, False]):
            r += amplitude * np.cos(k * t + phase)
        else:
            r += amplitude * np.sin(k * t + phase)
    r = np.clip(r, 0.2, None)
    x, y = r * np.cos(t), r * np.sin(t)
    x -= np.mean(x)
    y -= np.mean(y)
    return x, y


# ============================================================
# 2. Параболическая кривая (обобщённая)
# ============================================================

class ParabolaCurve:
    def __init__(self, params=None):
        if params is None:
            params = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0])
        self.params = np.array(params, dtype=float)

    @property
    def A(self):
        return self.params[0]

    @property
    def B(self):
        return self.params[1]

    @property
    def C(self):
        return self.params[2]

    @property
    def cx(self):
        return self.params[3]

    @property
    def cy(self):
        return self.params[4]

    @property
    def angle(self):
        return self.params[5]

    @property
    def u_min(self):
        return self.params[6]

    @property
    def u_max(self):
        return self.params[7]

    def eval_points(self, n=200):
        u = np.linspace(self.u_min, self.u_max, n)
        v = self.A * u ** 2 + self.B * u + self.C
        ca, sa = np.cos(self.angle), np.sin(self.angle)
        x = self.cx + ca * u - sa * v
        y = self.cy + sa * u + ca * v
        return x, y

    def min_dist_to(self, px, py, n=200):
        cx, cy = self.eval_points(n)
        pts_curve = np.column_stack([cx, cy])
        pts_query = np.column_stack([px, py])
        # Используем нашу функцию расстояния
        D = custom_cdist(pts_query, pts_curve)
        return np.min(D, axis=1)

    def custom_optimize(self, px, py, epochs=100, lr=0.01):
        """
        Собственный нелинейный оптимизатор (Adam) для минимизации
        ортогонального геометрического расстояния (ODR).
        """
        params = self.params.copy()
        delta = 1e-5
        m = np.zeros(6)
        v = np.zeros(6)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8

        pts_query = np.column_stack([px, py])

        for t in range(1, epochs + 1):
            pc_current = ParabolaCurve(params)
            pts_curve = np.column_stack(pc_current.eval_points(150))
            dists = np.min(custom_cdist(pts_query, pts_curve), axis=1)
            current_loss = np.sum(dists ** 2)

            grad = np.zeros(6)
            for i in range(6):  # Оптимизируем A, B, C, cx, cy, angle
                p_bump = params.copy()
                p_bump[i] += delta
                pc_bump = ParabolaCurve(p_bump)

                pts_curve_bump = np.column_stack(pc_bump.eval_points(150))
                dists_bump = np.min(custom_cdist(pts_query, pts_curve_bump), axis=1)
                loss_bump = np.sum(dists_bump ** 2)

                grad[i] = (loss_bump - current_loss) / delta

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            params[:6] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        return ParabolaCurve(params)

    @staticmethod
    def fit_to_segment(sx, sy):
        cx, cy = np.mean(sx), np.mean(sy)
        pts = np.column_stack([sx - cx, sy - cy])

        if len(pts) < 3:
            return ParabolaCurve(np.array([0, 0, 0, cx, cy, 0, -0.5, 0.5]))

        # PCA: вычисляем матрицу ковариации и находим направление наибольшей дисперсии
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        main = eigvecs[:, 1]
        angle = np.arctan2(main[1], main[0])

        ca, sa = np.cos(angle), np.sin(angle)
        u = ca * (sx - cx) + sa * (sy - cy)
        v = -sa * (sx - cx) + ca * (sy - cy)

        # Вызываем наш аналитический МНК
        A, B, C = custom_polyfit_2d(u, v)

        u_min, u_max = np.min(u), np.max(u)
        margin = 0.05 * (u_max - u_min + 1e-8)

        return ParabolaCurve(np.array([A, B, C, cx, cy, angle, u_min - margin, u_max + margin]))


# ============================================================
# 3. Итеративная сегментация (K-Means для кривых)
# ============================================================

def assign_points_to_parabolas(contour_x, contour_y, parabolas, n_eval=300):
    n_pts = len(contour_x)
    n_par = len(parabolas)

    dist_matrix = np.zeros((n_pts, n_par))
    for j, p in enumerate(parabolas):
        dist_matrix[:, j] = p.min_dist_to(contour_x, contour_y, n_eval)

    assignment = np.argmin(dist_matrix, axis=1)
    min_dists = np.min(dist_matrix, axis=1)
    return assignment, min_dists, dist_matrix


def iterative_fit(contour_x, contour_y, n_parabolas=12, n_iter=30, verbose=True):
    n = len(contour_x)
    seg_size = n // n_parabolas

    assignment = np.zeros(n, dtype=int)
    for i in range(n_parabolas):
        start = i * seg_size
        end = (i + 1) * seg_size if i < n_parabolas - 1 else n
        assignment[start:end] = i

    prev_loss = np.inf
    parabolas = [None] * n_parabolas
    history = []

    for iteration in range(n_iter):
        for j in range(n_parabolas):
            mask = assignment == j
            if np.sum(mask) < 4:
                if parabolas[j] is not None: continue
                idx = np.random.choice(n, size=max(10, seg_size // 2), replace=False)
                mask = np.zeros(n, dtype=bool)
                mask[idx] = True

            sx = contour_x[mask]
            sy = contour_y[mask]
            parabolas[j] = ParabolaCurve.fit_to_segment(sx, sy)

        assignment, min_dists, _ = assign_points_to_parabolas(
            contour_x, contour_y, parabolas, n_eval=300
        )

        loss = np.mean(min_dists ** 2)
        history.append(loss)

        if verbose and (iteration < 5 or iteration % 5 == 0 or iteration == n_iter - 1):
            pct = np.mean(min_dists < 0.03) * 100
            print(f"  Итерация {iteration + 1:3d}: MSE={loss:.6f}, "
                  f"mean_d={np.mean(min_dists):.4f}, <0.03: {pct:.1f}%")

        if abs(prev_loss - loss) < 1e-9 and iteration > 5:
            if verbose: print(f"  Сходимость достигнута на итерации {iteration + 1}")
            break
        prev_loss = loss

    return parabolas, assignment, history


# ============================================================
# 4. Финальная полировка (Кастомный Adam Градиентный спуск)
# ============================================================

def fine_tune(contour_x, contour_y, parabolas, epochs=100, verbose=True):
    if verbose:
        print("  Финальная полировка (собственный Adam оптимайзер)...")

    assignment, _, _ = assign_points_to_parabolas(contour_x, contour_y, parabolas, 300)

    optimized = []
    for j, p in enumerate(parabolas):
        mask = assignment == j
        if np.sum(mask) < 4:
            optimized.append(p)
            continue

        sx, sy = contour_x[mask], contour_y[mask]

        # Вызов нашего самописного оптимизатора ортогонального расстояния
        opt_p = p.custom_optimize(sx, sy, epochs=epochs, lr=0.02)
        optimized.append(opt_p)

    if verbose:
        dists = np.full(len(contour_x), np.inf)
        for p in optimized:
            dists = np.minimum(dists, p.min_dist_to(contour_x, contour_y, 300))
        print(f"  После полировки: mean_dist={np.mean(dists):.5f}, "
              f"<0.03: {np.mean(dists < 0.03) * 100:.1f}%")

    return optimized


# ============================================================
# 5. Пайплайн, Метрики и Визуализация
# ============================================================

def decompose_contour(cx, cy, n_parabolas=12, n_iter=40, epochs=100, verbose=True):
    if verbose: print(f"\n  Этап 1: Итеративная сегментация ({n_iter} итераций)")
    parabolas, _, history = iterative_fit(cx, cy, n_parabolas, n_iter, verbose)

    if verbose: print(f"\n  Этап 2: Fine-tuning (Градиентный спуск ODR)")
    parabolas = fine_tune(cx, cy, parabolas, epochs, verbose)
    return parabolas, history


def compute_metrics(cx, cy, parabolas):
    dists = np.full(len(cx), np.inf)
    for p in parabolas:
        dists = np.minimum(dists, p.min_dist_to(cx, cy, 300))
    return {
        'mean': np.mean(dists), 'max': np.max(dists), 'median': np.median(dists),
        'p90': np.percentile(dists, 90), 'pct_003': np.mean(dists < 0.03) * 100
    }


def plot_decomposition(contour_x, contour_y, parabolas, title="", save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    cx_cl, cy_cl = np.append(contour_x, contour_x[0]), np.append(contour_y, contour_y[0])
    colors = plt.cm.tab20(np.linspace(0, 1, len(parabolas)))

    axes[0].plot(cx_cl, cy_cl, 'k-', lw=2.5)
    axes[0].set_title('Исходный контур', fontweight='bold')

    axes[1].plot(cx_cl, cy_cl, 'k-', lw=1, alpha=0.25)
    for i, p in enumerate(parabolas):
        px, py = p.eval_points(300)
        axes[1].plot(px, py, '-', color=colors[i], lw=3.0)
    axes[1].set_title(f'Разложение ({len(parabolas)} сегментов)', fontweight='bold')

    dists = np.full(len(contour_x), np.inf)
    for p in parabolas:
        dists = np.minimum(dists, p.min_dist_to(contour_x, contour_y, 300))
    sc = axes[2].scatter(contour_x, contour_y, c=dists, cmap='RdYlGn_r', s=8, vmin=0)
    plt.colorbar(sc, ax=axes[2], label='Ошибка дистанции')
    axes[2].set_title('Карта ошибки', fontweight='bold')

    for ax in axes: ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=16, fontweight='bold');
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=100, bbox_inches='tight')
    return fig


def run_experiment(name, cx, cy, n_parabolas, n_iter=40, epochs=100):
    print(f"\n{'=' * 60}\n  {name} | Точек: {len(cx)} | Парабол: {n_parabolas}\n{'=' * 60}")
    parabolas, _ = decompose_contour(cx, cy, n_parabolas, n_iter, epochs, verbose=True)
    m = compute_metrics(cx, cy, parabolas)
    plot_decomposition(cx, cy, parabolas, title=name, save_path=f"{name.replace(' ', '_')}.png")
    plt.close('all')
    return m


# ============================================================
# 6. Запуск (MAIN)
# ============================================================
if __name__ == '__main__':
    all_metrics = {}

    cx, cy = generate_star(500, n_tips=5)
    all_metrics['Звезда'] = run_experiment("Звезда", cx, cy, n_parabolas=15, n_iter=100)

    cx, cy = generate_blob(500)
    all_metrics['Амёба'] = run_experiment("Амёба", cx, cy, n_parabolas=12, n_iter=100)

    cx, cy = generate_cat_silhouette(500)
    all_metrics['Кошка'] = run_experiment("Кошка", cx, cy, n_parabolas=14, n_iter=100)

    cx, cy = generate_random_complex_contour(500, max_freq=12)
    all_metrics['Случайный'] = run_experiment("Случайный контур", cx, cy, n_parabolas=16, n_iter=100)

    print(f"\n\n{'=' * 65}\n  СВОДНАЯ ТАБЛИЦА (Self-Implemented Math)\n{'-' * 65}")
    print(f"{'Контур':<15} {'Mean':>8} {'Median':>8} {'Max':>8} {'<0.03':>8}")
    for name, m in all_metrics.items():
        print(f"{name:<15} {m['mean']:>8.5f} {m['median']:>8.5f} {m['max']:>8.5f} {m['pct_003']:>7.1f}%")
