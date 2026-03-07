"""
Кейс №3 — Простой перцептрон: экстраполяция временного ряда акций

Компания: ПАО «Абрау-Дюрсо» (тикер ABRD, MOEX)

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
        self.n_input = n_input  # без bias
        self.n_output = n_output
        # Матрица весов: (n_input + 1) x n_output  (+1 для bias)
        self.W = None
        self.history_E = []  # история целевой функции E

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
# 5. Загрузка данных Абрау-Дюрсо
# ============================================================

def load_data(filepath=None):
    """Загрузка CSV (Абрау-Дюрсо, формат Investing.com) или генерация синтетических данных."""
    if filepath and os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            print(f"  Колонки CSV: {list(df.columns)}")

            # Ищем колонку с ценой закрытия
            # Формат Investing.com (русский): "Цена", "Откр.", "Макс.", "Мин."
            col = None
            for c in df.columns:
                if 'Цена' in c or 'Price' in c or 'Close' in c or 'close' in c:
                    col = c
                    break
            if col is None:
                col = df.columns[1]  # вторая колонка как fallback

            raw = df[col]

            # Очистка: убираем кавычки, пробелы, заменяем запятые на точки
            y = raw.astype(str).str.strip('"').str.strip()
            y = y.str.replace(',', '.', regex=False)  # 165,60 → 165.60
            y = pd.to_numeric(y, errors='coerce').values
            y = y[~np.isnan(y)]

            # Данные Investing.com идут от новых к старым — разворачиваем
            y = y[::-1]

            print(f"  Загружено из {filepath}: {len(y)} точек")
            print(f"  Первая цена (старая): {y[0]:.2f} ₽, Последняя (новая): {y[-1]:.2f} ₽")
            return y
        except Exception as e:
            print(f"  Ошибка загрузки: {e}")

    # Синтетические данные (имитация акций)
    print("  Генерация синтетических данных (имитация акций)...")
    np.random.seed(42)
    N = 300
    t = np.linspace(0, 6 * np.pi, N)
    y = (150 + 20 * np.sin(0.5 * t) + 10 * np.cos(1.3 * t)
         + 5 * np.sin(3.0 * t) + np.cumsum(np.random.normal(0, 0.5, N)))
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

def style_ax(ax, title):
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_ylabel('Цена, ₽', fontsize=10)
    ax.set_xlabel('Дни', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')


def reconstruct_series(perceptron, y_norm, n_total, n_in, n_out, n_cross, norm_params):
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


def plot_results_3_panels(y_raw, y_norm, norm_params,
                          perceptron_ana, perceptron_num,
                          n_in, n_out, n_cross,
                          train_idx, y_forecast_ana, y_forecast_num):
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    t_days = np.arange(len(y_raw))

    repr_ana = reconstruct_series(perceptron_ana, y_norm, len(y_raw), n_in, n_out, n_cross, norm_params)
    repr_num = reconstruct_series(perceptron_num, y_norm, len(y_raw), n_in, n_out, n_cross, norm_params)

    # --- График 1: Аппроксимация ---
    ax = axes[0]
    E1 = perceptron_ana.history_E[-1] if perceptron_ana.history_E else 0
    E2 = perceptron_num.history_E[-1] if perceptron_num.history_E else 0

    ax.plot(t_days, y_raw, 'g-', linewidth=1.5, alpha=0.7, label='Цена акций')
    ax.plot(t_days, repr_ana, 'r-', linewidth=2.5, alpha=0.9,
            label=f'Перцептрон (Аналитический) — аппроксимация (E={E1:.1f})')
    ax.plot(t_days, repr_num, 'b--', linewidth=2.0, alpha=0.85,
            label=f'Перцептрон (Численный) — аппроксимация (E={E2:.1f})')
    style_ax(ax, 'Аппроксимация цены акций ПАО «Абрау-Дюрсо» перцептроном')

    # --- График 2: Ретроспектива 90/10 ---
    ax = axes[1]
    t_train = t_days[:train_idx]
    t_test = t_days[train_idx:]

    retro_fore_norm_ana = forecast_future(perceptron_ana, y_norm[:train_idx], len(t_test), n_in, n_out, n_cross)
    retro_fore_ana = denormalize(retro_fore_norm_ana, norm_params)
    retro_fore_norm_num = forecast_future(perceptron_num, y_norm[:train_idx], len(t_test), n_in, n_out, n_cross)
    retro_fore_num = denormalize(retro_fore_norm_num, norm_params)

    ax.plot(t_train, y_raw[:train_idx], 'g-', linewidth=1.5, alpha=0.7, label='Обучающая выборка (90%)')
    ax.plot(t_test, y_raw[train_idx:], 'o', color='orange', markersize=4, alpha=0.9, label='Тестовая выборка (10%)')

    ax.plot(t_test, retro_fore_ana[:len(t_test)], 'r-', linewidth=2.5, alpha=0.9,
            label=f'Перцептрон (Аналитический) — прогноз на тест')
    ax.plot(t_test, retro_fore_num[:len(t_test)], 'b--', linewidth=2.0, alpha=0.85,
            label=f'Перцептрон (Численный) — прогноз на тест')

    ax.axvline(x=len(t_train), color='purple', linestyle=':', linewidth=2, alpha=0.8, label='Граница train/test')
    style_ax(ax, 'Ретроспективная проверка: 90% обучение / 10% тест')

    # --- График 3: Прогноз в будущее ---
    ax = axes[2]
    forecast_days = np.arange(len(y_raw), len(y_raw) + len(y_forecast_ana))

    ax.plot(t_days, y_raw, 'g-', linewidth=1.5, alpha=0.7, label='Цена акций (история)')
    ax.plot(forecast_days, y_forecast_ana, 'r-', linewidth=3, alpha=0.9,
            label=f'Перцептрон (Аналитический) — прогноз (+{len(y_forecast_ana)} дней)')
    ax.plot(forecast_days, y_forecast_num, 'b--', linewidth=2.5, alpha=0.85,
            label=f'Перцептрон (Численный) — прогноз (+{len(y_forecast_num)} дней)')
    ax.axvline(x=len(y_raw) - 1, color='orange', linestyle=':', linewidth=2, alpha=0.8, label='Начало прогноза')
    style_ax(ax, f'Прогноз цены акций ПАО «Абрау-Дюрсо» на {len(y_forecast_ana)} торговых дней вперёд')

    plt.tight_layout()
    return fig


# ============================================================
def main():
    print("=" * 80)
    print("КЕЙС №3: ПРОСТОЙ ПЕРЦЕПТРОН — ЭКСТРАПОЛЯЦИЯ ВРЕМЕННОГО РЯДА")
    print("Компания: ПАО «Абрау-Дюрсо» (ABRD, Московская биржа)")
    print("=" * 80)

    os.makedirs("plots", exist_ok=True)

    # --- Параметры ---
    N_IN = 12  # входных точек (≥10)
    N_OUT = 6  # выходных точек (≥5)
    N_CROSS = 3  # перекрёстных точек (≥3)
    DELTA = 0.05  # отступ от 0 и 1 при нормировке
    MAX_ITER = 10000  # итерации обучения
    LR_MAIN = 0.01  # основная скорость обучения

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
    filepath = 'ABRD (2y).csv'
    if not os.path.exists(filepath):
        # Проверяем uploads
        uploads = '/mnt/user-data/uploads'
        if os.path.exists(uploads):
            for f in os.listdir(uploads):
                if 'abrd' in f.lower() or 'абрау' in f.lower():
                    filepath = os.path.join(uploads, f)
                    break

    y_raw = load_data(filepath)
    print(f"  Всего точек: {len(y_raw)}")

    # --- Нормировка ---
    y_norm, norm_params = normalize(y_raw, delta=DELTA)
    print(f"  Диапазон исходных: [{np.min(y_raw):.2f}, {np.max(y_raw):.2f}] ₽")
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
                  f"W: {(ni + 1) * no:<5d} парам.  |  D={len(X_tmp):<5d}  |  "
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

    print(f"  Прогноз (аналит.): мин={np.min(y_forecast_ana):.2f}, макс={np.max(y_forecast_ana):.2f} ₽")
    print(f"  Прогноз (числ.):   мин={np.min(y_forecast_num):.2f}, макс={np.max(y_forecast_num):.2f} ₽")

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

    # ========================
    # ================================================================
    # ГРАФИКИ
    # ================================================================
    print("\n" + "=" * 80)
    print("ПОСТРОЕНИЕ ГРАФИКОВ")
    print("=" * 80)

    fig1 = plot_results_3_panels(
        y_raw, y_norm, norm_params,
        perc_ana, perc_num,
        N_IN, N_OUT, N_CROSS,
        split_idx, y_forecast_ana, y_forecast_num)
    fig1.savefig('plots/perceptron_results_3_panels.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: plots/perceptron_results_3_panels.png")
    plt.close(fig1)

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
