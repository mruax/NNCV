import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Целевая функция
# ============================================================

def f(x, y):
    """z = 2*sin(3*x*y) + 4*cos(x + 4*y)"""
    return 2 * np.sin(3 * x * y) + 4 * np.cos(x + 4 * y)


# ============================================================
# 2. Три способа вычисления градиента
# ============================================================

def grad_analytical(x, y):
    """
    Способ 1 — аналитический градиент.
    df/dx = 6y*cos(3xy) - 4*sin(x+4y)
    df/dy = 6x*cos(3xy) - 16*sin(x+4y)
    """
    dfdx = 6 * y * np.cos(3 * x * y) - 4 * np.sin(x + 4 * y)
    dfdy = 6 * x * np.cos(3 * x * y) - 16 * np.sin(x + 4 * y)
    return dfdx, dfdy


def grad_naive(x, y, dx=0.001):
    """
    Способ 2 — наивный численный градиент (центральные конечные разности).
    dx фиксирован.
    """
    dfdx = (f(x + dx, y) - f(x - dx, y)) / (2 * dx)
    dfdy = (f(x, y + dx) - f(x, y - dx)) / (2 * dx)
    return dfdx, dfdy


def grad_stochastic(x, y, dx_min=0.0005, dx_max=0.003):
    """
    Способ 3 — стохастический численный градиент.
    На каждой итерации delta берется случайно из [dx_min, dx_max].
    """
    dx = np.random.uniform(dx_min, dx_max)
    dy = np.random.uniform(dx_min, dx_max)
    dfdx = (f(x + dx, y) - f(x - dx, y)) / (2 * dx)
    dfdy = (f(x, y + dy) - f(x, y - dy)) / (2 * dy)
    return dfdx, dfdy


# ============================================================
# 3. Градиентный спуск
# ============================================================

def gradient_descent(grad_fn, x0, y0, lr=0.01, max_iter=500, tol=1e-7):
    """
    Градиентный спуск (поиск минимума).

    Параметры:
        grad_fn   — функция (x, y) -> (dfdx, dfdy)
        x0, y0    — начальная точка
        lr        — learning rate (lambda)
        max_iter  — макс. число итераций
        tol       — порог по длине градиента для остановки

    Возвращает:
        trajectory — массив (N, 2) координат пути
    """
    path = [(x0, y0)]
    x, y = x0, y0
    for _ in range(max_iter):
        gx, gy = grad_fn(x, y)
        grad_norm = np.sqrt(gx**2 + gy**2)
        if grad_norm < tol:
            break
        x = x - lr * gx
        y = y - lr * gy
        path.append((x, y))
    return np.array(path)


# ============================================================
# 4. Визуализация
# ============================================================

COLORS = ["#00FF00", "#FF00FF", "#00FFFF", "#FFFF00", "#FF1493", "#FF4500"]


def plot_contour(ax, title, grad_fn, starts, lr, grid_x, grid_y, Z):
    """Контурная карта + траектории спуска из нескольких начальных точек."""
    levels = np.linspace(Z.min(), Z.max(), 60)
    ax.contourf(grid_x, grid_y, Z, levels=levels, cmap="RdBu_r", alpha=0.85)
    ax.contour(grid_x, grid_y, Z, levels=20, colors="black", linewidths=0.3, alpha=0.4)

    for i, (sx, sy) in enumerate(starts):
        path = gradient_descent(grad_fn, sx, sy, lr=lr)
        c = COLORS[i % len(COLORS)]
        ax.plot(path[:, 0], path[:, 1], "-", color=c, linewidth=1.5, alpha=0.95)
        ax.plot(path[0, 0], path[0, 1], "o", color=c, markersize=8,
                markeredgecolor="black", markeredgewidth=1.2,
                label=f"start ({sx:.1f}, {sy:.1f})")
        ax.plot(path[-1, 0], path[-1, 1], "s", color=c, markersize=9,
                markeredgecolor="black", markeredgewidth=1.2)

    ax.set_title(title, fontsize=11, fontweight="bold", color="black")
    ax.set_xlabel("x", color="black"); ax.set_ylabel("y", color="black")
    ax.tick_params(colors="black")
    ax.legend(fontsize=6, loc="upper right", framealpha=0.9, facecolor="white", edgecolor="black")
    ax.set_xlim(grid_x.min(), grid_x.max())
    ax.set_ylim(grid_y.min(), grid_y.max())
    ax.set_facecolor("white")


def print_table(methods, starts, lr):
    """Таблица результатов в консоль."""
    print(f"\n{'Метод':<40} {'Старт':<16} {'Финиш':<24} {'f(fin)':<12} {'Шагов'}")
    print("-" * 100)
    for title, gfn in methods:
        for sx, sy in starts:
            path = gradient_descent(gfn, sx, sy, lr=lr)
            xf, yf = path[-1]
            fval = f(xf, yf)
            print(f"{title:<40} ({sx:+5.1f},{sy:+5.1f})   "
                  f"({xf:+7.4f},{yf:+7.4f})   {fval:+10.6f}         {len(path)}")


# ============================================================
# 5. Главная функция
# ============================================================

def main():
    x_range = (-3, 3)
    y_range = (-3, 3)
    resolution = 500
    dx_naive = 0.001
    dx_min, dx_max = 0.0005, 0.003

    lambdas = [0.001, 0.005, 0.01, 0.03, 0.05]

    starts = [
        ( 0.6,  1.7),
        (-2.8, -1.5),
        ( 0.5, -1.4),
        (-1.7,  1.3),
        ( 2.6, -0.8),
        (-0.4,  2.2),
    ]

    # Сетки
    xs = np.linspace(*x_range, resolution)
    ys = np.linspace(*y_range, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    Z = f(grid_x, grid_y)

    grad_fn_analytic   = grad_analytical
    grad_fn_naive      = lambda x, y: grad_naive(x, y, dx=dx_naive)
    grad_fn_stochastic = lambda x, y: grad_stochastic(x, y, dx_min=dx_min, dx_max=dx_max)

    os.makedirs("plots", exist_ok=True)

    # ==================== Перебор λ ====================
    for lr in lambdas:
        lr_str = f"{lr:.4f}".rstrip("0").rstrip(".")

        print(f"\n{'='*60}")
        print(f"  λ = {lr}")
        print(f"{'='*60}")

        methods = [
            ("1. Аналитический",                            grad_fn_analytic),
            (f"2. Наивный (Δ={dx_naive})",                  grad_fn_naive),
            (f"3. Стохастический (Δ∈[{dx_min},{dx_max}])",  grad_fn_stochastic),
        ]

        # --- Рис. 1: три способа рядом ---
        fig1, axes1 = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
        fig1.patch.set_facecolor("white")
        fig1.suptitle(
            f"λ = {lr}",
            fontsize=14, fontweight="bold", color="black"
        )
        for ax, (title, gfn) in zip(axes1, methods):
            plot_contour(ax, title, gfn, starts, lr, grid_x, grid_y, Z)

        p = os.path.join("plots", f"lambda_{lr_str}.png")
        fig1.savefig(p, dpi=150, facecolor="white"); plt.close(fig1)
        print(f"  Сохранено: {p}")

        # --- Таблица ---
        print_table(methods, starts, lr)

    # ==================== Сводный рисунок ====================
    fig_all, axes_all = plt.subplots(1, len(lambdas),
                                     figsize=(5 * len(lambdas), 7),
                                     constrained_layout=True)
    fig_all.patch.set_facecolor("white")
    fig_all.suptitle(
        r"Сравнение λ (аналитический градиент)   |   "
        r"$z = 2\sin(3xy)+4\cos(x+4y)$",
        fontsize=14, fontweight="bold", color="black"
    )
    for ax, lr in zip(axes_all, lambdas):
        plot_contour(ax, f"λ = {lr}", grad_fn_analytic, starts, lr, grid_x, grid_y, Z)

    p = os.path.join("plots", "summary_all_lambdas.png")
    fig_all.savefig(p, dpi=150, facecolor="white"); plt.close(fig_all)
    print(f"\nСводный (λ): {p}")

    # ==================== Сравнение различных Δx (наивный метод) ====================
    dx_values = [0.5, 0.1, 0.01, 0.001, 0.00001]
    lr_for_dx = 0.03  # Фиксированная λ для сравнения Δx

    fig_dx, axes_dx = plt.subplots(1, len(dx_values),
                                   figsize=(5 * len(dx_values), 7),
                                   constrained_layout=True)
    fig_dx.patch.set_facecolor("white")
    fig_dx.suptitle(
        r"Сравнение Δx (наивный численный градиент)   |   "
        r"$z = 2\sin(3xy)+4\cos(x+4y)$"
        f"   |   λ = {lr_for_dx}",
        fontsize=14, fontweight="bold", color="black"
    )

    for ax, dx_val in zip(axes_dx, dx_values):
        grad_fn_dx = lambda x, y, dv=dx_val: grad_naive(x, y, dx=dv)
        plot_contour(ax, f"Δx = {dx_val}", grad_fn_dx, starts, lr_for_dx, grid_x, grid_y, Z)

    p_dx = os.path.join("plots", "summary_all_dx.png")
    fig_dx.savefig(p_dx, dpi=150, facecolor="white"); plt.close(fig_dx)
    print(f"Сводный (Δx): {p_dx}")

    print("\nГотово!")


if __name__ == "__main__":
    main()