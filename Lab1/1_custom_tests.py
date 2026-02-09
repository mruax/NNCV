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
# 2. Вычисление градиента (Наивный метод с параметром dx)
# ============================================================

def grad_naive(x, y, dx):
    """
    Наивный численный градиент (центральные конечные разности).
    dx передается как параметр для экспериментов.
    """
    dfdx = (f(x + dx, y) - f(x - dx, y)) / (2 * dx)
    dfdy = (f(x, y + dx) - f(x, y - dx)) / (2 * dx)
    return dfdx, dfdy


# ============================================================
# 3. Градиентный спуск
# ============================================================

def gradient_descent(grad_fn, x0, y0, lr=0.01, max_iter=10000, tol=1e-7):
    """
    Классический градиентный спуск.
    """
    path = [(x0, y0)]
    x, y = x0, y0
    for _ in range(max_iter):
        gx, gy = grad_fn(x, y)
        grad_norm = np.sqrt(gx ** 2 + gy ** 2)

        # Условие остановки, если градиент слишком мал
        if grad_norm < tol:
            break

        x = x - lr * gx
        y = y - lr * gy
        path.append((x, y))
    return np.array(path)


# ============================================================
# 4. Визуализация
# ============================================================

COLORS = ["#00FF00", "#FF00FF", "#00FFFF", "#FFFF00", "#FF8800", "#FFFFFF"]


def plot_contour(ax, title, grad_fn, starts, lr, grid_x, grid_y, Z):
    """Рисует контурную карту и траектории спуска."""
    levels = np.linspace(Z.min(), Z.max(), 60)
    ax.contourf(grid_x, grid_y, Z, levels=levels, cmap="RdYlBu_r", alpha=0.85)
    ax.contour(grid_x, grid_y, Z, levels=20, colors="k", linewidths=0.3, alpha=0.4)

    for i, (sx, sy) in enumerate(starts):
        path = gradient_descent(grad_fn, sx, sy, lr=lr)
        c = COLORS[i % len(COLORS)]

        # Линия траектории
        ax.plot(path[:, 0], path[:, 1], "-", color=c, linewidth=1.5, alpha=0.9)
        # Начальная точка
        ax.plot(path[0, 0], path[0, 1], "o", color=c, markersize=5, markeredgecolor="k")
        # Конечная точка
        ax.plot(path[-1, 0], path[-1, 1], "X", color=c, markersize=8, markeredgecolor="k")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(grid_x.min(), grid_x.max())
    ax.set_ylim(grid_y.min(), grid_y.max())


# ============================================================
# 5. Главная функция
# ============================================================

def main():
    # Настройки области и сетки
    x_range = (-3, 3)
    y_range = (-3, 3)
    resolution = 300

    # === ОСНОВНЫЕ ПАРАМЕТРЫ ЗАДАЧИ ===
    lr = 0.05  # Фиксированный learning rate

    # Список проверяемых значений Delta (шаг численного дифференцирования)
    deltas = [0.5, 0.1, 0.01, 0.001, 1e-5]

    # Стартовые точки
    starts = [
        (2.0, 2.0),
        (-2.5, 1.0),
        (1.0, -2.0),
        (-1.0, -1.5),
        (0.5, 2.5),
        (-2.0, -2.5),
    ]

    # Подготовка сетки для фона
    xs = np.linspace(*x_range, resolution)
    ys = np.linspace(*y_range, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    Z = f(grid_x, grid_y)

    # Создаем папку для графиков
    os.makedirs("plots_delta_check", exist_ok=True)

    # Настраиваем большую фигуру (2 ряда по 3 графика)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
    axes_flat = axes.flatten()

    fig.suptitle(f"Сравнение значений Delta (численный градиент) | LR = {lr}", fontsize=16, fontweight="bold")

    print(f"\n{'Delta':<10} {'Start':<16} {'End':<20} {'f(end)':<12} {'Steps'}")
    print("-" * 80)

    # === Цикл по разным значениям delta ===
    for i, delta in enumerate(deltas):
        ax = axes_flat[i]

        # Создаем функцию-обертку, которая фиксирует текущее значение delta
        # Важно: используем d=delta, чтобы захватить значение
        grad_fn = lambda x, y, d=delta: grad_naive(x, y, dx=d)

        # Рисуем график
        plot_contour(ax, f"Delta = {delta}", grad_fn, starts, lr, grid_x, grid_y, Z)

        # Выводим текстовую статистику
        for sx, sy in starts:
            path = gradient_descent(grad_fn, sx, sy, lr=lr)
            xf, yf = path[-1]
            # Печатаем результат только для одного пути или всех - здесь для всех
            print(f"{delta:<10} ({sx:+5.1f},{sy:+5.1f})   ({xf:+7.4f},{yf:+7.4f})   {f(xf, yf):+10.6f}   {len(path)}")
        print("-" * 40)

    # Скрываем пустые оси, если графиков меньше, чем мест в сетке
    if len(deltas) < len(axes_flat):
        for j in range(len(deltas), len(axes_flat)):
            axes_flat[j].axis('off')

    save_path = os.path.join("plots_delta_check", "deltas_comparison.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nГрафик сохранен в: {save_path}")
    # plt.show()


if __name__ == "__main__":
    main()
