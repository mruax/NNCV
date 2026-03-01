"""
Кейс №3 — Простой перцептрон: экстраполяция временного ряда акций

Простой перцептрон (2 слоя: input → output) с аналитическим градиентным спуском.
Скользящее окно: N_in входных точек, N_out выходных, N_cross перекрёстных.
Функция активации: сигмоида.
Нормировка данных на [δ, 1−δ] для корректной работы сигмоиды.

Формулы с доски:
  1) S_j^d = Σ_i x_i^d · W_ij + 1 · W_bias_j   (суммарный сигнал)
  2) x_j^d = f(S_j^d)                              (активация)
  3) ε_d = Σ_j (x_j^d − t_j^d)²                   (ошибка на элементе d)
  4) E({W_ij}) = Σ_d ε_d                           (целевая функция)
  5) ∂ε_d/∂W_ij = (x_j^d − t_j^d) · f'(S_j^d) · x_i^d   (аналитический градиент)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import json

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10


# ============================================================
# 1. Функция активации и её производная
# ============================================================

def sigmoid(x):
    """Сигмоида: f(s) = 1 / (1 + exp(-s))"""
    x_clip = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clip))


def sigmoid_derivative(x):
    """Производная сигмоиды: f'(s) = f(s) · (1 − f(s))"""
    s = sigmoid(x)
    return s * (1.0 - s)


# ============================================================
# 2. Класс простого перцептрона
# ============================================================

class SimplePerceptron:
    """
    Простой перцептрон (два слоя: input → output).

    Архитектура:
        - Входной слой: N_in нейронов + 1 нейрон смещения (bias = 1)
        - Выходной слой: N_out нейронов с сигмоидной активацией
        - Матрица весов W: размер (N_in + 1) × N_out

    Прямой проход:
        S_j = Σ_i x_i · W_ij  (включая bias)
        output_j = sigmoid(S_j)

    Аналитический градиент (формула с доски):
        ∂ε_d / ∂W_ij = (x_j^d − t_j^d) · f'(S_j^d) · x_i^d
    """

    def __init__(self, n_input, n_output):
        self.n_input = n_input      # без bias
        self.n_output = n_output
        # Матрица весов: (n_input + 1) x n_output  (+1 для bias)
        self.W = None
        self.history_E = []         # история целевой функции E

    def init_weights(self, method='random', scale=0.5):
        """
        Инициализация весов.
        method: 'random' — случайные, 'zeros' — нули, 'xavier' — Xavier
        """
        n_in = self.n_input + 1  # +1 для bias
        n_out = self.n_output

        if method == 'random':
            self.W = np.random.uniform(-scale, scale, (n_in, n_out))
        elif method == 'zeros':
            self.W = np.zeros((n_in, n_out))
        elif method == 'xavier':
            limit = np.sqrt(6.0 / (n_in + n_out))
            self.W = np.random.uniform(-limit, limit, (n_in, n_out))
        elif method == 'small_random':
            self.W = np.random.uniform(-0.1, 0.1, (n_in, n_out))

        return self.W.copy()

    def forward(self, X_input):
        """
        Прямой проход через перцептрон.

        X_input: массив (N_samples, N_in) — входные данные (без bias)

        Возвращает:
            X_out   — выход перцептрона (N_samples, N_out)
            S       — суммарные сигналы до активации (N_samples, N_out)
            X_aug   — входы с добавленным bias (N_samples, N_in + 1)
        """
        N = X_input.shape[0]
        # Добавляем нейрон смещения (bias = 1)
        bias = np.ones((N, 1))
        X_aug = np.hstack([X_input, bias])  # (N, N_in + 1)

        # S_j = Σ_i x_i · W_ij
        S = X_aug @ self.W  # (N, N_out)

        # x_j = f(S_j)
        X_out = sigmoid(S)  # (N, N_out)

        return X_out, S, X_aug

    def compute_error(self, X_input, T):
        """
        Целевая функция: E = Σ_d Σ_j (x_j^d − t_j^d)²

        X_input: (D, N_in) — входы
        T:       (D, N_out) — эталоны
        """
        X_out, _, _ = self.forward(X_input)
        eps_d = np.sum((X_out - T) ** 2, axis=1)  # ε_d для каждого d
        E = np.sum(eps_d)
        return E

    def compute_gradient_analytical(self, X_input, T):
        """
        Аналитический градиент ∂E/∂W_ij.

        Формула с доски:
            ∂ε_d/∂W_ij = (x_j^d − t_j^d) · f'(S_j^d) · x_i^d

            ∂E/∂W_ij = Σ_d ∂ε_d/∂W_ij

        Возвращает: матрицу градиентов той же формы, что W
        """
        X_out, S, X_aug = self.forward(X_input)  # прямой прогон

        # (x_j^d − t_j^d)
        diff = X_out - T  # (D, N_out)

        # f'(S_j^d) — производная сигмоиды
        f_prime = sigmoid_derivative(S)  # (D, N_out)

        # δ_j^d = 2 · (x_j^d − t_j^d) · f'(S_j^d)   (множитель 2 от производной квадрата)
        delta = 2.0 * diff * f_prime  # (D, N_out)

        # ∂E/∂W_ij = Σ_d x_i^d · δ_j^d
        grad_W = X_aug.T @ delta  # (N_in + 1, N_out)

        return grad_W

    def compute_gradient_numerical(self, X_input, T, h=1e-5):
        """
        Численный градиент (центральные разности) — для проверки.
        """
        grad_W = np.zeros_like(self.W)
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                self.W[i, j] += h
                E_plus = self.compute_error(X_input, T)
                self.W[i, j] -= 2 * h
                E_minus = self.compute_error(X_input, T)
                self.W[i, j] += h  # восстановить
                grad_W[i, j] = (E_plus - E_minus) / (2 * h)
        return grad_W

    def train(self, X_input, T, lr=0.1, max_iter=5000, tol=1e-8,
              verbose=True, method='analytical'):
        """
        Обучение градиентным спуском.

        W' = W - λ · ∂E/∂W

        method: 'analytical' или 'numerical'
        """
        self.history_E = []

        for it in range(max_iter):
            # Вычисление градиента
            if method == 'analytical':
                grad = self.compute_gradient_analytical(X_input, T)
            else:
                grad = self.compute_gradient_numerical(X_input, T)

            # Gradient clipping
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 100:
                grad = grad * (100.0 / grad_norm)

            # Шаг градиентного спуска
            self.W -= lr * grad

            # Ошибка
            E = self.compute_error(X_input, T)
            self.history_E.append(E)

            if verbose and (it % 1000 == 0 or it == max_iter - 1):
                print(f"    Итерация {it:5d}: E = {E:.6f}, ||grad|| = {grad_norm:.4e}")

            # Критерий остановки
            if grad_norm < tol:
                if verbose:
                    print(f"    Сходимость на итерации {it} (||grad|| < {tol})")
                break

        return E

    def predict(self, X_input):
        """Предсказание (прямой проход)."""
        X_out, _, _ = self.forward(X_input)
        return X_out


# ============================================================
# 3. Формирование датасета из временного ряда
# ============================================================

def build_dataset(series, n_in=10, n_out=5, n_cross=3):
    """
    Скользящее окно с перекрёстными точками.

    Пример для n_in=10, n_out=5, n_cross=3:
        Входные точки:  [0..9]   (10 штук)
        Выходные точки: [7..11]  (5 штук, перекрытие 8,9,10 — 3 точки)

    Каждый элемент датасета (d):
        x_i = series[start : start + n_in]
        t_j = series[start + n_in - n_cross : start + n_in - n_cross + n_out]

    Окно сдвигается на 1 позицию вправо.
    """
    window_total = n_in + n_out - n_cross
    D = len(series) - window_total + 1

    if D <= 0:
        raise ValueError(
            f"Ряд слишком короткий ({len(series)}) для окна {window_total}")

    X_data = np.zeros((D, n_in))
    T_data = np.zeros((D, n_out))

    for d in range(D):
        X_data[d, :] = series[d: d + n_in]
        out_start = d + n_in - n_cross
        T_data[d, :] = series[out_start: out_start + n_out]

    return X_data, T_data


# ============================================================
# 4. Нормировка данных
# ============================================================

def normalize(data, delta=0.05):
    """
    Нормировка на [δ, 1−δ] для корректной работы сигмоиды.
    Возвращает нормированные данные и параметры (min, max, delta).
    """
    dmin = np.min(data)
    dmax = np.max(data)
    rng = dmax - dmin
    if rng < 1e-12:
        rng = 1.0
    normed = delta + (1.0 - 2 * delta) * (data - dmin) / rng
    return normed, (dmin, dmax, delta)


def denormalize(normed, params):
    """Обратная нормировка."""
    dmin, dmax, delta = params
    rng = dmax - dmin
    if rng < 1e-12:
        rng = 1.0
    return dmin + (normed - delta) / (1.0 - 2 * delta) * rng


# ============================================================
# 5. Загрузка данных
# ============================================================

def load_data(filepath=None):
    """Загрузка CSV (Heineken NV Stock Price History) или генерация синтетических данных."""
    if filepath and os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            # Ищем колонку с ценой
            col = [c for c in df.columns if 'Price' in c or 'Close' in c or 'close' in c]
            if col:
                raw = df[col[0]]
            else:
                raw = df.iloc[:, 1]
            # Очистка: убираем запятые, пробелы, конвертируем в float
            y = pd.to_numeric(raw.astype(str).str.replace(',', '').str.strip('"').str.strip(),
                              errors='coerce').values
            y = y[~np.isnan(y)]
            # Разворачиваем: CSV идёт от новых к старым, а нам нужно от старых к новым
            y = y[::-1]
            print(f"  Загружено из {filepath}: {len(y)} точек")
            print(f"  Первая цена (старая): {y[0]:.2f}, Последняя (новая): {y[-1]:.2f}")
            return y
        except Exception as e:
            print(f"  Ошибка загрузки: {e}")

    # Синтетические данные (имитация акций)
    print("  Генерация синтетических данных (имитация акций)...")
    np.random.seed(42)
    N = 300
    t = np.linspace(0, 6 * np.pi, N)
    y = (50 + 15 * np.sin(0.5 * t) + 7 * np.cos(1.3 * t)
         + 3 * np.sin(3.0 * t) + np.cumsum(np.random.normal(0, 0.3, N)))
    return y


# ============================================================
# 6. Прогнозирование в будущее
# ============================================================

def forecast_future(perceptron, last_window, n_steps, n_in, n_out, n_cross):
    """
    Итеративный прогноз: берём последние n_in точек, предсказываем n_out,
    сдвигаем окно и повторяем.
    """
    current = last_window.copy()
    all_predictions = []

    while len(all_predictions) < n_steps:
        inp = current[-n_in:].reshape(1, -1)
        pred = perceptron.predict(inp).flatten()

        # Новые точки = выход минус перекрёстные (те, что уже есть)
        new_points = pred[n_cross:]
        all_predictions.extend(new_points.tolist())

        # Сдвигаем окно
        current = np.concatenate([current, new_points])

    return np.array(all_predictions[:n_steps])


# ============================================================
# 7. Визуализация
# ============================================================

def reconstruct_series(perceptron, y_norm, n_total, n_in, n_out, n_cross, norm_params):
    """
    Восстанавливает временной ряд из предсказаний перцептрона.
    Для каждой позиции усредняет все предсказания, покрывающие эту точку.
    """
    X_ds, T_ds = build_dataset(y_norm[:n_total], n_in, n_out, n_cross)
    pred = perceptron.predict(X_ds)

    reproduced = np.zeros(n_total)
    counts = np.zeros(n_total)
    for d in range(len(X_ds)):
        out_start = d + n_in - n_cross
        for j in range(n_out):
            idx = out_start + j
            if idx < n_total:
                reproduced[idx] += denormalize(pred[d, j], norm_params)
                counts[idx] += 1
    mask = counts > 0
    reproduced[mask] /= counts[mask]
    reproduced[~mask] = np.nan
    return reproduced


def plot_all_results(y_raw, y_norm, norm_params,
                     perceptron_ana, perceptron_num,
                     n_in, n_out, n_cross,
                     train_idx, y_forecast_ana, y_forecast_num):
    """Построение основных графиков (7 панелей, 4×2 без последней)."""

    fig = plt.figure(figsize=(20, 22))

    # --- Воспроизведённые ряды (нужны для нескольких графиков) ---
    repr_ana = reconstruct_series(perceptron_ana, y_norm, train_idx,
                                  n_in, n_out, n_cross, norm_params)
    repr_num = reconstruct_series(perceptron_num, y_norm, train_idx,
                                  n_in, n_out, n_cross, norm_params)

    # ------- 1. Исходный ряд + граница train/test -------
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(y_raw, 'k-', linewidth=0.8, alpha=0.7, label='Исходные данные')
    ax1.axvline(x=train_idx, color='red', linestyle=':', linewidth=2,
                label=f'Граница train/test ({train_idx}/{len(y_raw)-train_idx})')
    ax1.set_title('Исходный временной ряд — Heineken NV', fontweight='bold')
    ax1.set_ylabel('Цена, €')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ------- 2. Нормированный ряд [δ, 1−δ] -------
    ax2 = fig.add_subplot(4, 2, 2)
    ax2.plot(y_norm, 'b-', linewidth=0.8, alpha=0.7)
    d = norm_params[2]
    ax2.axhline(y=d, color='gray', linestyle='--', alpha=0.5, label=f'δ = {d}')
    ax2.axhline(y=1 - d, color='gray', linestyle='--', alpha=0.5, label=f'1−δ = {1-d:.2f}')
    ax2.set_title('Нормированный ряд [δ, 1−δ] для сигмоиды', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ------- 3. Сходимость: оба метода на одном графике -------
    ax3 = fig.add_subplot(4, 2, 3)
    ax3.plot(perceptron_ana.history_E, 'r-', linewidth=1,
             label=f'Аналитический → E={perceptron_ana.history_E[-1]:.4f}')
    ax3.plot(perceptron_num.history_E, 'b-', linewidth=1, alpha=0.7,
             label=f'Численный → E={perceptron_num.history_E[-1]:.4f}')
    ax3.set_xlabel('Итерация')
    ax3.set_ylabel('E (целевая функция)')
    ax3.set_title('Сходимость градиентного спуска', fontweight='bold')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ------- 4. Воспроизведение: АНАЛИТИЧЕСКИЙ градиент -------
    ax4 = fig.add_subplot(4, 2, 4)
    ax4.plot(y_raw[:train_idx], 'k-', linewidth=0.8, alpha=0.4, label='Исходные')
    ax4.plot(repr_ana, 'r-', linewidth=1.5, alpha=0.8, label='Перцептрон (аналит.)')
    rmse_ana = np.sqrt(np.nanmean((repr_ana - y_raw[:train_idx]) ** 2))
    ax4.set_title(f'Воспроизведение — аналитический градиент (RMSE={rmse_ana:.2f})',
                  fontweight='bold')
    ax4.set_ylabel('Цена, €')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ------- 5. Воспроизведение: ЧИСЛЕННЫЙ градиент -------
    ax5 = fig.add_subplot(4, 2, 5)
    ax5.plot(y_raw[:train_idx], 'k-', linewidth=0.8, alpha=0.4, label='Исходные')
    ax5.plot(repr_num, 'b-', linewidth=1.5, alpha=0.8, label='Перцептрон (числ.)')
    rmse_num = np.sqrt(np.nanmean((repr_num - y_raw[:train_idx]) ** 2))
    ax5.set_title(f'Воспроизведение — численный градиент (RMSE={rmse_num:.2f})',
                  fontweight='bold')
    ax5.set_ylabel('Цена, €')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ------- 6. Прогноз в будущее (+10%) -------
    ax6 = fig.add_subplot(4, 2, 6)
    forecast_len = len(y_forecast_ana)
    x_hist = np.arange(len(y_raw))
    x_fore = np.arange(len(y_raw), len(y_raw) + forecast_len)

    ax6.plot(x_hist, y_raw, 'k-', linewidth=0.8, alpha=0.5, label='История')
    ax6.plot(x_fore, y_forecast_ana, 'r--', linewidth=2, label='Прогноз (аналит.)')
    ax6.plot(x_fore, y_forecast_num, 'b--', linewidth=2, alpha=0.7, label='Прогноз (числ.)')
    ax6.axvline(x=len(y_raw), color='orange', linestyle=':', linewidth=2)
    ax6.set_title('Прогноз в будущее (+10%)', fontweight='bold')
    ax6.set_ylabel('Цена, €')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # ------- 7. Ретроспектива: обучаем на 90%, проверяем на 10% -------
    # Обучаем отдельный перцептрон на train, прогнозируем на test
    ax7 = fig.add_subplot(4, 2, 7)

    # Прогноз перцептрона (аналит.) вперёд от конца train
    test_len = len(y_raw) - train_idx
    retro_fore_norm = forecast_future(
        perceptron_ana, y_norm[:train_idx], test_len, n_in, n_out, n_cross)
    retro_fore = denormalize(retro_fore_norm, norm_params)

    x_train_range = np.arange(train_idx)
    x_test_range = np.arange(train_idx, len(y_raw))

    ax7.plot(x_train_range, y_raw[:train_idx], 'k-', linewidth=0.6, alpha=0.3,
             label='Train (история)')
    ax7.plot(x_test_range, y_raw[train_idx:], 'go-', markersize=4, linewidth=1.5,
             label='Test (реальные)')
    ax7.plot(x_test_range[:len(retro_fore)], retro_fore[:test_len],
             'r--o', markersize=3, linewidth=1.5, label='Прогноз перцептрона')
    ax7.axvline(x=train_idx, color='purple', linestyle=':', linewidth=2,
                label='Граница 90/10')
    rmse_retro = np.sqrt(np.mean((retro_fore[:test_len] - y_raw[train_idx:]) ** 2))
    ax7.set_title(f'Ретроспектива: прогноз на 10% vs реальность (RMSE={rmse_retro:.2f})',
                  fontweight='bold')
    ax7.set_ylabel('Цена, €')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_weights_heatmap(perceptron, n_in, title=''):
    """
    Тепловая карта матрицы весов W.
    Каждая ячейка W_ij = сила связи от входного нейрона i к выходному j.
      > 0 (красный) — входной сигнал x_i УВЕЛИЧИВАЕТ суммарный сигнал S_j
      < 0 (синий)   — входной сигнал x_i УМЕНЬШАЕТ суммарный сигнал S_j
      ≈ 0 (белый)   — нейрон i слабо влияет на выход j
    Последняя строка = bias (смещение, сдвиг порога активации).
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    W = perceptron.W
    vmax = np.max(np.abs(W))
    im = ax.imshow(W, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    # Подписи строк
    row_labels = [f'x_{i+1} (точка t-{n_in-i})' for i in range(n_in)] + ['bias (=1)']
    ax.set_yticks(range(W.shape[0]))
    ax.set_yticklabels(row_labels, fontsize=8)

    # Подписи столбцов
    col_labels = [f'out_{j+1}' for j in range(W.shape[1])]
    ax.set_xticks(range(W.shape[1]))
    ax.set_xticklabels(col_labels, fontsize=9)

    ax.set_xlabel('Выходной нейрон j (прогнозируемые точки)', fontsize=11)
    ax.set_ylabel('Входной нейрон i (исходные точки + bias)', fontsize=11)

    # Значения в ячейках
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            color = 'white' if abs(W[i, j]) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{W[i,j]:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Сила связи W_ij\n(красный=усиление, синий=подавление)', fontsize=9)

    ax.set_title(
        f'Матрица весов W ({W.shape[0]}×{W.shape[1]}) {title}\n'
        f'W_ij > 0 → x_i усиливает выход j  |  W_ij < 0 → x_i подавляет выход j',
        fontweight='bold', fontsize=11)
    plt.tight_layout()
    return fig


def plot_gradient_check(perceptron, X_input, T):
    """
    Визуальное доказательство корректности аналитического градиента.

    Три карты ∂E/∂W_ij (каждая ячейка = производная по одному весу):
      1) Аналитический — по формуле: 2·(x_j − t_j)·f'(S_j)·x_i
      2) Численный     — (E(W+h) − E(W−h)) / 2h
      3) |Разница|     — если ≈ 0 → формула реализована верно
    """
    grad_ana = perceptron.compute_gradient_analytical(X_input, T)
    grad_num = perceptron.compute_gradient_numerical(X_input, T, h=1e-5)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    vmax = max(np.max(np.abs(grad_ana)), np.max(np.abs(grad_num)))

    # Аналитический
    im0 = axes[0].imshow(grad_ana, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0].set_title('Аналитический градиент ∂E/∂W_ij\n(по формуле с доски)', fontweight='bold')
    plt.colorbar(im0, ax=axes[0])

    # Численный
    im1 = axes[1].imshow(grad_num, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].set_title('Численный градиент ∂E/∂W_ij\n(конечные разности)', fontweight='bold')
    plt.colorbar(im1, ax=axes[1])

    # Разница
    diff = np.abs(grad_ana - grad_num)
    im2 = axes[2].imshow(diff, aspect='auto', cmap='hot')
    axes[2].set_title(f'|Разница| (max = {np.max(diff):.2e})\nЧёрное = совпадение ✓',
                      fontweight='bold')
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel('Выходной нейрон j')
        ax.set_ylabel('Входной нейрон i (послед. = bias)')

    plt.suptitle(
        'Верификация градиента: аналитический vs численный\n'
        'Если |разница| ≈ 0 — формула ∂ε_d/∂W_ij = 2·(x_j^d−t_j^d)·f\'(S_j^d)·x_i^d  верна',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================
# 8. Главная функция
# ============================================================

def main():
    print("=" * 80)
    print("КЕЙС №3: ПРОСТОЙ ПЕРЦЕПТРОН — ЭКСТРАПОЛЯЦИЯ ВРЕМЕННОГО РЯДА")
    print("=" * 80)

    os.makedirs("plots", exist_ok=True)

    # --- Параметры ---
    N_IN = 12           # входных точек (≥10)
    N_OUT = 6           # выходных точек (≥5)
    N_CROSS = 3         # перекрёстных точек (≥3)
    DELTA = 0.05        # отступ от 0 и 1 при нормировке
    MAX_ITER = 10000    # итерации обучения
    LR_MAIN = 0.01      # основная скорость обучения

    print(f"\nПараметры перцептрона:")
    print(f"  N_in = {N_IN}, N_out = {N_OUT}, N_cross = {N_CROSS}")
    print(f"  Нормировка: [{DELTA}, {1 - DELTA}]")
    print(f"  Активация: сигмоида")
    print(f"  Размер матрицы весов W: ({N_IN + 1}) × {N_OUT} = {(N_IN + 1) * N_OUT} параметров")

    # --- Загрузка данных ---
    print("\n" + "=" * 80)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 80)

    # Ищем CSV: сначала в текущей папке, потом в uploads
    filepath = 'Heineken NV Stock Price History.csv'
    if not os.path.exists(filepath):
        filepath = 'Heineken_NV_Stock_Price_History.csv'
    if not os.path.exists(filepath):
        # Проверяем uploads
        uploads = '/mnt/user-data/uploads'
        for f in os.listdir(uploads):
            if 'heineken' in f.lower() or 'stock' in f.lower():
                filepath = os.path.join(uploads, f)
                break

    y_raw = load_data(filepath)
    print(f"  Всего точек: {len(y_raw)}")

    # --- Нормировка ---
    y_norm, norm_params = normalize(y_raw, delta=DELTA)
    print(f"  Диапазон исходных: [{np.min(y_raw):.2f}, {np.max(y_raw):.2f}]")
    print(f"  Диапазон нормированных: [{np.min(y_norm):.4f}, {np.max(y_norm):.4f}]")

    # --- Разбиение: 90% train / 10% test ---
    split_idx = int(len(y_norm) * 0.90)
    y_train = y_norm[:split_idx]
    y_test = y_norm[split_idx:]
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}")

    # --- Формирование датасета ---
    X_train, T_train = build_dataset(y_train, N_IN, N_OUT, N_CROSS)
    print(f"\n  Датасет: {X_train.shape[0]} элементов")
    print(f"  X_train shape: {X_train.shape}, T_train shape: {T_train.shape}")

    # ================================================================
    # ЧАСТЬ 1: Обучение аналитическим градиентом
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 1: ОБУЧЕНИЕ — АНАЛИТИЧЕСКИЙ ГРАДИЕНТ")
    print(f"  λ = {LR_MAIN}, max_iter = {MAX_ITER}")
    print("=" * 80)

    perc_ana = SimplePerceptron(N_IN, N_OUT)
    W_init = perc_ana.init_weights(method='xavier')
    print(f"  Инициализация: Xavier, W shape = {perc_ana.W.shape}")

    # Проверка градиента перед обучением (на небольшой выборке)
    print("\n  --- Проверка аналитического градиента ---")
    perc_check = SimplePerceptron(N_IN, N_OUT)
    perc_check.init_weights(method='small_random')
    X_check, T_check = X_train[:3], T_train[:3]
    grad_a = perc_check.compute_gradient_analytical(X_check, T_check)
    grad_n = perc_check.compute_gradient_numerical(X_check, T_check, h=1e-5)
    max_diff = np.max(np.abs(grad_a - grad_n))
    denom = np.maximum(np.abs(grad_a), np.abs(grad_n)) + 1e-12
    rel_diff = np.max(np.abs(grad_a - grad_n) / denom)
    print(f"  Макс. абсолютная разница: {max_diff:.2e}")
    print(f"  Макс. относительная разница: {rel_diff:.2e}")
    if rel_diff < 1e-3:
        print("  ✓ Градиент корректен!")
    else:
        print("  ⚠ Есть расхождение (возможно из-за числ. точности)")

    # Обучение
    print("\n  --- Обучение ---")
    E_final_ana = perc_ana.train(X_train, T_train, lr=LR_MAIN,
                                 max_iter=MAX_ITER, method='analytical')
    print(f"  Финальная ошибка E = {E_final_ana:.6f}")

    # ================================================================
    # ЧАСТЬ 2: Обучение численным градиентом (для сравнения)
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 2: ОБУЧЕНИЕ — ЧИСЛЕННЫЙ ГРАДИЕНТ")
    print("=" * 80)

    perc_num = SimplePerceptron(N_IN, N_OUT)
    perc_num.W = W_init.copy()  # те же начальные веса

    E_final_num = perc_num.train(X_train, T_train, lr=LR_MAIN,
                                  max_iter=min(MAX_ITER, 1000),
                                  method='numerical')
    print(f"  Финальная ошибка E = {E_final_num:.6f}")

    # ================================================================
    # ЧАСТЬ 3: Эксперименты с λ
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 3: ЭКСПЕРИМЕНТЫ С λ")
    print("=" * 80)

    lambdas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    lambdas_results = []

    for lr_val in lambdas:
        perc_tmp = SimplePerceptron(N_IN, N_OUT)
        perc_tmp.W = W_init.copy()
        perc_tmp.train(X_train, T_train, lr=lr_val,
                       max_iter=5000, verbose=False, method='analytical')
        final_E = perc_tmp.history_E[-1] if perc_tmp.history_E else float('inf')
        lambdas_results.append((lr_val, perc_tmp.history_E))
        print(f"  λ = {lr_val:<6.3f}  →  E_final = {final_E:.6f}  ({len(perc_tmp.history_E)} итераций)")

    # ================================================================
    # ЧАСТЬ 4: Эксперименты с начальными весами
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 4: ЭКСПЕРИМЕНТЫ С НАЧАЛЬНОЙ ТОЧКОЙ (ИНИЦИАЛИЗАЦИЕЙ)")
    print("=" * 80)

    init_methods = ['random', 'xavier', 'small_random', 'zeros']
    for im in init_methods:
        perc_tmp = SimplePerceptron(N_IN, N_OUT)
        perc_tmp.init_weights(method=im)
        perc_tmp.train(X_train, T_train, lr=LR_MAIN,
                       max_iter=5000, verbose=False, method='analytical')
        final_E = perc_tmp.history_E[-1] if perc_tmp.history_E else float('inf')
        print(f"  Инициализация: {im:<14s}  →  E_final = {final_E:.6f}")

    # ================================================================
    # ЧАСТЬ 5: Эксперименты с кодировкой (n_in, n_out, n_cross)
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 5: ЭКСПЕРИМЕНТЫ С КОДИРОВКОЙ (размер окна)")
    print("=" * 80)

    configs = [
        (10, 5, 3),
        (12, 6, 3),
        (15, 5, 3),
        (15, 8, 4),
        (20, 10, 5),
    ]
    for ni, no, nc in configs:
        try:
            X_tmp, T_tmp = build_dataset(y_train, ni, no, nc)
            perc_tmp = SimplePerceptron(ni, no)
            perc_tmp.init_weights(method='xavier')
            perc_tmp.train(X_tmp, T_tmp, lr=0.01,
                           max_iter=5000, verbose=False, method='analytical')
            final_E = perc_tmp.history_E[-1] if perc_tmp.history_E else float('inf')
            avg_E = final_E / len(X_tmp)
            print(f"  n_in={ni:<3d} n_out={no:<3d} n_cross={nc:<3d}  |  "
                  f"W: {(ni+1)*no:<5d} парам.  |  D={len(X_tmp):<5d}  |  "
                  f"E={final_E:.4f}  E/D={avg_E:.6f}")
        except Exception as e:
            print(f"  n_in={ni} n_out={no} n_cross={nc}: Ошибка — {e}")

    # ================================================================
    # ЧАСТЬ 6: Прогноз в будущее
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 6: ПРОГНОЗ В БУДУЩЕЕ (+10%)")
    print("=" * 80)

    forecast_len = int(len(y_raw) * 0.10)
    print(f"  Прогноз на {forecast_len} точек вперёд")

    # Аналитический
    y_fore_norm_ana = forecast_future(
        perc_ana, y_norm, forecast_len, N_IN, N_OUT, N_CROSS)
    y_forecast_ana = denormalize(y_fore_norm_ana, norm_params)

    # Численный
    y_fore_norm_num = forecast_future(
        perc_num, y_norm, forecast_len, N_IN, N_OUT, N_CROSS)
    y_forecast_num = denormalize(y_fore_norm_num, norm_params)

    print(f"  Прогноз (аналит.): мин={np.min(y_forecast_ana):.2f}, макс={np.max(y_forecast_ana):.2f}")
    print(f"  Прогноз (числ.):   мин={np.min(y_forecast_num):.2f}, макс={np.max(y_forecast_num):.2f}")

    # ================================================================
    # ЧАСТЬ 7: Метрики
    # ================================================================
    print("\n" + "=" * 80)
    print("МЕТРИКИ")
    print("=" * 80)

    # На обучающей выборке
    pred_train_ana = perc_ana.predict(X_train)
    mse_train = np.mean((pred_train_ana - T_train) ** 2)
    rmse_train = np.sqrt(mse_train)

    # На тестовой выборке (если хватает данных)
    try:
        X_test, T_test = build_dataset(y_norm[split_idx - N_IN:], N_IN, N_OUT, N_CROSS)
        pred_test = perc_ana.predict(X_test)
        mse_test = np.mean((pred_test - T_test) ** 2)
        rmse_test = np.sqrt(mse_test)
        print(f"  Train RMSE (норм.): {rmse_train:.6f}")
        print(f"  Test  RMSE (норм.): {rmse_test:.6f}")
    except Exception:
        print(f"  Train RMSE (норм.): {rmse_train:.6f}")
        print(f"  Test: недостаточно данных для оценки")

    # ================================================================
    # ГРАФИКИ
    # ================================================================
    print("\n" + "=" * 80)
    print("ПОСТРОЕНИЕ ГРАФИКОВ")
    print("=" * 80)

    # 1. Основные результаты (7 панелей)
    fig1 = plot_all_results(
        y_raw, y_norm, norm_params,
        perc_ana, perc_num,
        N_IN, N_OUT, N_CROSS,
        split_idx, y_forecast_ana, y_forecast_num)
    fig1.savefig('plots/perceptron_results.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: plots/perceptron_results.png")
    plt.close(fig1)

    # 2. Проверка градиента
    perc_for_check = SimplePerceptron(N_IN, N_OUT)
    perc_for_check.init_weights(method='small_random')
    fig2 = plot_gradient_check(perc_for_check, X_train[:10], T_train[:10])
    fig2.savefig('plots/gradient_check.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: plots/gradient_check.png")
    plt.close(fig2)

    # 3. Тепловая карта весов
    fig3 = plot_weights_heatmap(perc_ana, N_IN, '(аналитический)')
    fig3.savefig('plots/weights_heatmap.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: plots/weights_heatmap.png")
    plt.close(fig3)

    # 4. Детальное сравнение λ
    fig4, axes4 = plt.subplots(2, 3, figsize=(18, 10))
    axes4 = axes4.flatten()
    for idx, (lr_val, hist) in enumerate(lambdas_results):
        if idx < len(axes4):
            axes4[idx].plot(hist, 'r-', linewidth=0.8)
            axes4[idx].set_title(f'λ = {lr_val}  →  E_fin = {hist[-1]:.4f}', fontweight='bold')
            axes4[idx].set_xlabel('Итерация')
            axes4[idx].set_ylabel('E')
            axes4[idx].set_yscale('log')
            axes4[idx].grid(True, alpha=0.3)
    plt.suptitle('Влияние λ на сходимость градиентного спуска', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig4.savefig('plots/lambda_comparison.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: plots/lambda_comparison.png")
    plt.close(fig4)

    # 5. Схема перцептрона (текстовая визуализация)
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)
    ax5.set_aspect('equal')
    ax5.axis('off')

    # Входной слой
    n_draw_in = N_IN
    y_positions_in = np.linspace(1, 9, n_draw_in + 1)  # +1 для bias
    for i, yp in enumerate(y_positions_in):
        color = 'lightblue' if i < n_draw_in else 'lightyellow'
        label = f'x_{i+1}' if i < n_draw_in else 'bias=1'
        circle = plt.Circle((2, yp), 0.3, color=color, ec='black', linewidth=1.5)
        ax5.add_patch(circle)
        ax5.text(2, yp, label, ha='center', va='center', fontsize=5)

    # Выходной слой
    n_draw_out = N_OUT
    y_positions_out = np.linspace(2, 8, n_draw_out)
    for j, yp in enumerate(y_positions_out):
        circle = plt.Circle((8, yp), 0.3, color='lightsalmon', ec='black', linewidth=1.5)
        ax5.add_patch(circle)
        ax5.text(8, yp, f'out_{j+1}', ha='center', va='center', fontsize=5)

    # Связи
    for yp_in in y_positions_in:
        for yp_out in y_positions_out:
            ax5.plot([2.3, 7.7], [yp_in, yp_out], 'gray', linewidth=0.3, alpha=0.4)

    ax5.text(2, 0.3, f'Input ({N_IN}+1 bias)', ha='center', fontsize=11, fontweight='bold')
    ax5.text(8, 0.3, f'Output ({N_OUT})', ha='center', fontsize=11, fontweight='bold')
    ax5.text(5, 9.5, f'Простой перцептрон: {(N_IN+1)*N_OUT} весов W_ij',
             ha='center', fontsize=13, fontweight='bold')
    # ax5.text(5, 0.0, 'S_j = Σ x_i · W_ij ;   x_j = σ(S_j)', ha='center', fontsize=11, style='italic')

    fig5.savefig('plots/perceptron_scheme.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: plots/perceptron_scheme.png")
    plt.close(fig5)

    # --- Сохранение результатов ---
    results = {
        'параметры': {
            'N_in': N_IN, 'N_out': N_OUT, 'N_cross': N_CROSS,
            'delta': DELTA, 'lr': LR_MAIN, 'max_iter': MAX_ITER,
            'размер_W': f'{N_IN+1}x{N_OUT}',
            'всего_параметров': (N_IN + 1) * N_OUT
        },
        'метрики': {
            'E_train_analytical': float(E_final_ana),
            'E_train_numerical': float(E_final_num),
            'RMSE_train': float(rmse_train),
        },
        'эксперименты_lambda': {
            str(lr_val): float(hist[-1]) for lr_val, hist in lambdas_results
        }
    }

    with open('plots/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("  Сохранено: plots/results.json")

    # Прогноз в CSV
    df_forecast = pd.DataFrame({
        'step': np.arange(1, forecast_len + 1),
        'forecast_analytical': y_forecast_ana,
        'forecast_numerical': y_forecast_num,
    })
    df_forecast.to_csv('plots/forecast.csv', index=False)
    print("  Сохранено: plots/forecast.csv")

    print("\n" + "=" * 80)
    print("ГОТОВО!")
    print("=" * 80)


if __name__ == "__main__":
    main()
