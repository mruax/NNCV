"""
Кейс №4 — Сложный перцептрон (Feed-Forward NN): экстраполяция временного ряда

Многослойная нейронная сеть прямого распространения с алгоритмом
обратного распространения ошибки (Backpropagation).

Архитектура: Input(N_in+1) → Hidden(N_hid+1) → Output(N_out)
Скользящее окно с перекрёстными точками (как в кейсе №3).
Мини-батчи, контроль переобучения (early stopping), аудит.

Формулы с доски (Backpropagation):

  Прямой проход (слой за слоем):
    S_p = Σ_q X_q · W_qp       (суммарный сигнал)
    X_p = f(S_p)                 (активация)

  Градиент по весам (для любого слоя):
    ∂E/∂W_qp = X_q · (∂E/∂S_p)

  Локальная ошибка выходного слоя K:
    (∂E/∂S_k) = (∂E/∂X_k) · f'(S_k)

  Рекуррентная формула для скрытого слоя P (через слой K):
    (∂E/∂S_p) = f'(S_p) · Σ_k (∂E/∂S_k) · W_pk

  Обновление весов:
    {W}' = {W} - λ · ∇E
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
# 2. Класс многослойного перцептрона
# ============================================================

class MultiLayerPerceptron:
    """
    Сложный перцептрон (Feed-Forward Neural Network).

    Архитектура (пример с одним скрытым слоем):
        Input (N_in + 1 bias) --W1--> Hidden (N_hid + 1 bias) --W2--> Output (N_out)

    Поддерживает произвольное число скрытых слоёв.

    Backpropagation (формулы с доски):
        ∂E/∂W_qp = X_q · (∂E/∂S_p)
        (∂E/∂S_k) = 2·(X_k - t_k) · f'(S_k)           — выходной слой
        (∂E/∂S_p) = f'(S_p) · Σ_k (∂E/∂S_k) · W_pk    — скрытый слой
    """

    def __init__(self, layer_sizes):
        """
        layer_sizes: список размеров слоёв, например [12, 20, 6]
                     означает 12 входов, 20 скрытых, 6 выходов.
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.weights = []       # список матриц W
        self.history_E_train = []
        self.history_E_test = []

    def init_weights(self, method='xavier'):
        """Инициализация весов для всех слоёв."""
        self.weights = []
        for i in range(self.n_layers - 1):
            n_in = self.layer_sizes[i] + 1   # +1 для bias
            n_out = self.layer_sizes[i + 1]

            if method == 'xavier':
                limit = np.sqrt(6.0 / (n_in + n_out))
                W = np.random.uniform(-limit, limit, (n_in, n_out))
            elif method == 'random':
                W = np.random.uniform(-0.5, 0.5, (n_in, n_out))
            elif method == 'small_random':
                W = np.random.uniform(-0.1, 0.1, (n_in, n_out))
            elif method == 'he':
                W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            else:
                W = np.zeros((n_in, n_out))

            self.weights.append(W)

        return [w.copy() for w in self.weights]

    def forward(self, X_input):
        """
        Прямой проход (Forward pass).

        Возвращает:
            activations — список выходов каждого слоя (включая входной)
            sums        — список суммарных сигналов S каждого слоя (кроме входного)
            augmented   — список входов с bias каждого слоя
        """
        activations = [X_input]
        sums = []
        augmented = []

        current = X_input
        for i, W in enumerate(self.weights):
            # Добавляем bias
            N = current.shape[0]
            bias = np.ones((N, 1))
            current_aug = np.hstack([current, bias])
            augmented.append(current_aug)

            # S = X_aug · W
            S = current_aug @ W
            sums.append(S)

            # X = f(S)
            current = sigmoid(S)
            activations.append(current)

        return activations, sums, augmented

    def compute_error(self, X_input, T):
        """E = Σ_d Σ_k (X_k^d - t_k^d)²"""
        activations, _, _ = self.forward(X_input)
        output = activations[-1]
        return np.sum((output - T) ** 2)

    def backprop(self, X_input, T):
        """
        Обратное распространение ошибки (Backpropagation).

        Формулы с доски:
            Выходной слой:  ∂E/∂S_k = 2·(X_k - t_k) · f'(S_k)
            Скрытый слой:   ∂E/∂S_p = f'(S_p) · Σ_k (∂E/∂S_k) · W_pk
            Градиент весов: ∂E/∂W_qp = X_q · (∂E/∂S_p)

        Возвращает: список градиентов для каждой матрицы весов.
        """
        activations, sums, augmented = self.forward(X_input)

        # Список градиентов (по одному на каждую матрицу весов)
        grads = [None] * len(self.weights)

        # --- Выходной слой ---
        output = activations[-1]
        diff = output - T                           # (X_k - t_k)
        f_prime = sigmoid_derivative(sums[-1])      # f'(S_k)
        delta = 2.0 * diff * f_prime                # ∂E/∂S_k

        # Градиент весов последнего слоя: ∂E/∂W = X_aug^T · delta
        grads[-1] = augmented[-1].T @ delta

        # --- Скрытые слои (обратный ход) ---
        for i in range(len(self.weights) - 2, -1, -1):
            # Рекуррентная формула: ∂E/∂S_p = f'(S_p) · Σ_k (∂E/∂S_k) · W_pk
            # delta текущий = delta следующего слоя, протянутый назад через веса
            W_no_bias = self.weights[i + 1][:-1, :]   # убираем строку bias
            delta = (delta @ W_no_bias.T) * sigmoid_derivative(sums[i])

            # Градиент весов: ∂E/∂W_qp = X_q · (∂E/∂S_p)
            grads[i] = augmented[i].T @ delta

        return grads

    def backprop_numerical(self, X_input, T, h=1e-5):
        """Численный градиент для проверки backprop."""
        grads = []
        for layer_idx in range(len(self.weights)):
            grad_W = np.zeros_like(self.weights[layer_idx])
            for i in range(grad_W.shape[0]):
                for j in range(grad_W.shape[1]):
                    self.weights[layer_idx][i, j] += h
                    E_plus = self.compute_error(X_input, T)
                    self.weights[layer_idx][i, j] -= 2 * h
                    E_minus = self.compute_error(X_input, T)
                    self.weights[layer_idx][i, j] += h
                    grad_W[i, j] = (E_plus - E_minus) / (2 * h)
            grads.append(grad_W)
        return grads

    def train(self, X_train, T_train, X_test=None, T_test=None,
              lr=0.01, max_iter=10000, batch_size=None,
              audit_every=100, early_stop_patience=500,
              verbose=True):
        """
        Обучение с мини-батчами, аудитом и early stopping.

        batch_size: размер мини-батча (None = весь датасет)
        audit_every: каждые N итераций считать ошибку на test
        early_stop_patience: остановка если E_test растёт N итераций подряд
        """
        self.history_E_train = []
        self.history_E_test = []

        D = X_train.shape[0]
        if batch_size is None:
            batch_size = D

        best_E_test = float('inf')
        best_weights = [w.copy() for w in self.weights]
        patience_counter = 0
        best_iter = 0

        for it in range(max_iter):
            # --- Формируем мини-батч ---
            indices = np.random.permutation(D)
            for start in range(0, D, batch_size):
                end = min(start + batch_size, D)
                batch_idx = indices[start:end]

                X_batch = X_train[batch_idx]
                T_batch = T_train[batch_idx]

                # Backpropagation
                grads = self.backprop(X_batch, T_batch)

                # Gradient clipping
                for g_idx in range(len(grads)):
                    g_norm = np.linalg.norm(grads[g_idx])
                    if g_norm > 100:
                        grads[g_idx] = grads[g_idx] * (100.0 / g_norm)

                # Обновление весов: W' = W - λ · ∂E/∂W
                for g_idx in range(len(self.weights)):
                    self.weights[g_idx] -= lr * grads[g_idx]

            # --- Аудит ---
            if it % audit_every == 0 or it == max_iter - 1:
                E_train = self.compute_error(X_train, T_train)
                self.history_E_train.append(E_train)

                if X_test is not None and T_test is not None:
                    E_test = self.compute_error(X_test, T_test)
                    self.history_E_test.append(E_test)

                    # Early stopping: сохраняем лучшие веса
                    if E_test < best_E_test:
                        best_E_test = E_test
                        best_weights = [w.copy() for w in self.weights]
                        patience_counter = 0
                        best_iter = it
                    else:
                        patience_counter += audit_every

                    if verbose and (it % (audit_every * 10) == 0 or it == max_iter - 1):
                        print(f"    Итерация {it:5d}: E_train={E_train:.4f}, "
                              f"E_test={E_test:.4f}, best_iter={best_iter}")

                    if patience_counter >= early_stop_patience:
                        if verbose:
                            print(f"    Early stopping на итерации {it} "
                                  f"(лучшая: {best_iter}, E_test={best_E_test:.4f})")
                        break
                else:
                    if verbose and (it % (audit_every * 10) == 0 or it == max_iter - 1):
                        print(f"    Итерация {it:5d}: E_train={E_train:.4f}")

        # Восстанавливаем лучшие веса
        if X_test is not None:
            self.weights = best_weights
            if verbose:
                print(f"    Восстановлены лучшие веса (итерация {best_iter})")

        return self.history_E_train[-1] if self.history_E_train else float('inf')

    def predict(self, X_input):
        """Предсказание."""
        activations, _, _ = self.forward(X_input)
        return activations[-1]


# ============================================================
# 3. Формирование датасета (скользящее окно)
# ============================================================

def build_dataset(series, n_in=12, n_out=10, n_cross=3):
    """Скользящее окно с перекрёстными точками (как в кейсе №3)."""
    window_total = n_in + n_out - n_cross
    D = len(series) - window_total + 1
    if D <= 0:
        raise ValueError(f"Ряд слишком короткий ({len(series)}) для окна {window_total}")

    X_data = np.zeros((D, n_in))
    T_data = np.zeros((D, n_out))
    for d in range(D):
        X_data[d, :] = series[d: d + n_in]
        out_start = d + n_in - n_cross
        T_data[d, :] = series[out_start: out_start + n_out]
    return X_data, T_data


# ============================================================
# 4. Нормировка
# ============================================================

def normalize(data, delta=0.05):
    dmin, dmax = np.min(data), np.max(data)
    rng = dmax - dmin if (dmax - dmin) > 1e-12 else 1.0
    normed = delta + (1.0 - 2 * delta) * (data - dmin) / rng
    return normed, (dmin, dmax, delta)


def denormalize(normed, params):
    dmin, dmax, delta = params
    rng = dmax - dmin if (dmax - dmin) > 1e-12 else 1.0
    return dmin + (normed - delta) / (1.0 - 2 * delta) * rng


# ============================================================
# 5. Загрузка данных
# ============================================================

def load_data(filepath=None):
    """Загрузка CSV Heineken NV."""
    if filepath and os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            col = [c for c in df.columns if 'Price' in c or 'Close' in c or 'close' in c]
            raw = df[col[0]] if col else df.iloc[:, 1]
            y = pd.to_numeric(raw.astype(str).str.replace(',', '').str.strip('"').str.strip(),
                              errors='coerce').values
            y = y[~np.isnan(y)][::-1]
            print(f"  Загружено из {filepath}: {len(y)} точек")
            print(f"  Первая (старая): {y[0]:.2f}, Последняя (новая): {y[-1]:.2f}")
            return y
        except Exception as e:
            print(f"  Ошибка: {e}")

    print("  Генерация синтетических данных...")
    np.random.seed(42)
    N = 300
    t = np.linspace(0, 6 * np.pi, N)
    return 50 + 15*np.sin(0.5*t) + 7*np.cos(1.3*t) + 3*np.sin(3*t) + np.cumsum(np.random.normal(0, 0.3, N))


# ============================================================
# 6. Прогнозирование в будущее
# ============================================================

def forecast_future(net, y_norm, n_steps, n_in, n_out, n_cross):
    """
    Итеративный прогноз: на каждом шаге добавляем 1 точку.
    Для каждой новой точки усредняем предсказания из нескольких окон.
    """
    current = list(y_norm.copy())

    for step in range(n_steps):
        # Собираем предсказания для следующей точки из нескольких смещённых окон
        predictions = []
        target_pos = len(current)  # позиция, которую хотим предсказать

        # Основное предсказание: последние n_in точек
        inp = np.array(current[-n_in:]).reshape(1, -1)
        pred = net.predict(inp).flatten()
        # Эта точка находится на позиции n_cross (первая "новая" после перекрытия)
        # relative to start of output window
        # output covers positions: len(current)-n_cross .. len(current)-n_cross+n_out-1
        # target_pos = len(current), output starts at len(current)-n_cross
        # so index in pred = n_cross
        if n_cross < n_out:
            predictions.append(pred[n_cross])

        # Дополнительные окна со сдвигом назад (если есть предыдущие предсказания)
        for shift in range(1, min(n_cross, n_out - n_cross)):
            if len(current) - n_in - shift >= 0:
                inp_shifted = np.array(current[-(n_in + shift): -(shift) if shift > 0 else len(current)]).reshape(1, -1)
                if inp_shifted.shape[1] == n_in:
                    pred_shifted = net.predict(inp_shifted).flatten()
                    idx_in_pred = n_cross + shift
                    if idx_in_pred < n_out:
                        predictions.append(pred_shifted[idx_in_pred])

        # Среднее предсказание
        if predictions:
            new_val = np.mean(predictions)
        else:
            new_val = pred[n_cross] if n_cross < n_out else pred[-1]

        current.append(float(new_val))

    return np.array(current[len(y_norm):][:n_steps])


def reconstruct_series(net, y_norm, n_total, n_in, n_out, n_cross, norm_params):
    """Восстановление ряда из предсказаний."""
    X_ds, _ = build_dataset(y_norm[:n_total], n_in, n_out, n_cross)
    pred = net.predict(X_ds)
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


# ============================================================
# 7. Визуализация
# ============================================================

def plot_all_results(y_raw, y_norm, norm_params, nets, net_labels,
                     n_in, n_out, n_cross, train_idx,
                     y_forecasts, best_net):
    """Основные графики."""
    fig = plt.figure(figsize=(20, 22))

    # 1. Исходный ряд
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(y_raw, 'k-', linewidth=0.8, alpha=0.7, label='Heineken NV')
    ax1.axvline(x=train_idx, color='red', linestyle=':', linewidth=2,
                label=f'Граница train/test ({train_idx}/{len(y_raw)-train_idx})')
    ax1.set_title('Исходный временной ряд — Heineken NV', fontweight='bold')
    ax1.set_ylabel('Цена, EUR')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Нормированный ряд
    ax2 = fig.add_subplot(4, 2, 2)
    ax2.plot(y_norm, 'b-', linewidth=0.8, alpha=0.7)
    d = norm_params[2]
    ax2.axhline(y=d, color='gray', linestyle='--', alpha=0.5, label=f'delta={d}')
    ax2.axhline(y=1-d, color='gray', linestyle='--', alpha=0.5, label=f'1-delta={1-d:.2f}')
    ax2.set_title('Нормированный ряд [delta, 1-delta]', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Сходимость E_train + E_test (контроль переобучения)
    ax3 = fig.add_subplot(4, 2, 3)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for idx, (net, label) in enumerate(zip(nets, net_labels)):
        c = colors[idx % len(colors)]
        ax3.plot(net.history_E_train, '-', color=c, linewidth=1, label=f'{label} train')
        if net.history_E_test:
            ax3.plot(net.history_E_test, '--', color=c, linewidth=1, alpha=0.7, label=f'{label} test')
    ax3.set_xlabel('Аудит (каждые N итераций)')
    ax3.set_ylabel('E')
    ax3.set_title('Сходимость и контроль переобучения', fontweight='bold')
    ax3.set_yscale('log')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # 4. Воспроизведение лучшей моделью
    ax4 = fig.add_subplot(4, 2, 4)
    repr_best = reconstruct_series(best_net, y_norm, train_idx,
                                    n_in, n_out, n_cross, norm_params)
    ax4.plot(y_raw[:train_idx], 'k-', linewidth=0.8, alpha=0.4, label='Исходные')
    ax4.plot(repr_best, 'r-', linewidth=1.5, alpha=0.8, label='Лучшая сеть')
    rmse = np.sqrt(np.nanmean((repr_best - y_raw[:train_idx])**2))
    ax4.set_title(f'Воспроизведение на train (RMSE={rmse:.2f})', fontweight='bold')
    ax4.set_ylabel('Цена, EUR')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Сравнение архитектур (воспроизведение)
    ax5 = fig.add_subplot(4, 2, 5)
    ax5.plot(y_raw[:train_idx], 'k-', linewidth=0.8, alpha=0.3, label='Исходные')
    for idx, (net, label) in enumerate(zip(nets, net_labels)):
        c = colors[idx % len(colors)]
        r = reconstruct_series(net, y_norm, train_idx, n_in, n_out, n_cross, norm_params)
        rmse_i = np.sqrt(np.nanmean((r - y_raw[:train_idx])**2))
        ax5.plot(r, '-', color=c, linewidth=1, alpha=0.7, label=f'{label} (RMSE={rmse_i:.2f})')
    ax5.set_title('Сравнение архитектур (разное кол-во нейронов)', fontweight='bold')
    ax5.set_ylabel('Цена, EUR')
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)

    # 6. Прогноз в будущее
    ax6 = fig.add_subplot(4, 2, 6)
    x_hist = np.arange(len(y_raw))
    ax6.plot(x_hist, y_raw, 'k-', linewidth=0.8, alpha=0.5, label='История')
    for idx, (fore, label) in enumerate(zip(y_forecasts, net_labels)):
        c = colors[idx % len(colors)]
        x_fore = np.arange(len(y_raw), len(y_raw) + len(fore))
        ax6.plot(x_fore, fore, '--', color=c, linewidth=1.5, label=f'{label}')
    ax6.axvline(x=len(y_raw), color='orange', linestyle=':', linewidth=2)
    ax6.set_title('Прогноз в будущее (+15%)', fontweight='bold')
    ax6.set_ylabel('Цена, EUR')
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3)

    # 7. Ретроспектива (прогноз на test vs реальность)
    ax7 = fig.add_subplot(4, 2, 7)
    test_len = len(y_raw) - train_idx
    retro = forecast_future(best_net, y_norm[:train_idx], test_len, n_in, n_out, n_cross)
    retro_denorm = denormalize(retro, norm_params)
    x_test = np.arange(train_idx, len(y_raw))
    ax7.plot(np.arange(train_idx), y_raw[:train_idx], 'k-', linewidth=0.6, alpha=0.3, label='Train')
    ax7.plot(x_test, y_raw[train_idx:], 'go-', markersize=4, linewidth=1.5, label='Test (реальные)')
    ax7.plot(x_test[:len(retro_denorm)], retro_denorm[:test_len], 'r--o', markersize=3,
             linewidth=1.5, label='Прогноз')
    ax7.axvline(x=train_idx, color='purple', linestyle=':', linewidth=2)
    rmse_retro = np.sqrt(np.mean((retro_denorm[:test_len] - y_raw[train_idx:])**2))
    ax7.set_title(f'Ретроспектива: прогноз 10% vs реальность (RMSE={rmse_retro:.2f})', fontweight='bold')
    ax7.set_ylabel('Цена, EUR')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_gradient_check(net, X_input, T):
    """Верификация backprop vs численный градиент."""
    grads_ana = net.backprop(X_input, T)
    grads_num = net.backprop_numerical(X_input, T, h=1e-5)

    n_layers = len(grads_ana)
    fig, axes = plt.subplots(n_layers, 3, figsize=(18, 5 * n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)

    for layer in range(n_layers):
        ga, gn = grads_ana[layer], grads_num[layer]
        vmax = max(np.max(np.abs(ga)), np.max(np.abs(gn)))
        diff = np.abs(ga - gn)

        im0 = axes[layer, 0].imshow(ga, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[layer, 0].set_title(f'Слой {layer+1}: Backprop (аналит.)', fontweight='bold')
        plt.colorbar(im0, ax=axes[layer, 0])

        im1 = axes[layer, 1].imshow(gn, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[layer, 1].set_title(f'Слой {layer+1}: Численный', fontweight='bold')
        plt.colorbar(im1, ax=axes[layer, 1])

        im2 = axes[layer, 2].imshow(diff, aspect='auto', cmap='hot')
        axes[layer, 2].set_title(f'|Разница| (max={np.max(diff):.2e})', fontweight='bold')
        plt.colorbar(im2, ax=axes[layer, 2])

    plt.suptitle('Верификация Backpropagation: аналитический vs численный градиент',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_network_scheme(layer_sizes):
    """Схема архитектуры сети."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(-0.5, len(layer_sizes) * 3 + 0.5)
    max_n = max(layer_sizes) + 1  # +1 for bias
    ax.set_ylim(-1, max_n + 1)
    ax.set_aspect('equal')
    ax.axis('off')

    layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(len(layer_sizes) - 2)] + ['Output']
    layer_colors = ['lightblue'] + ['lightgreen'] * (len(layer_sizes) - 2) + ['lightsalmon']

    positions = []
    for l_idx, (n_neurons, name, color) in enumerate(zip(layer_sizes, layer_names, layer_colors)):
        x = l_idx * 3
        n_draw = min(n_neurons, 8)
        has_bias = (l_idx < len(layer_sizes) - 1)
        total = n_draw + (1 if has_bias else 0)
        ys = np.linspace(0.5, max_n - 0.5, total)
        layer_pos = []

        for i, yp in enumerate(ys):
            if has_bias and i == len(ys) - 1:
                c = plt.Circle((x, yp), 0.25, color='lightyellow', ec='black', lw=1.2)
                ax.text(x, yp, 'b', ha='center', va='center', fontsize=7)
            else:
                c = plt.Circle((x, yp), 0.25, color=color, ec='black', lw=1.2)
                label = str(i+1) if n_neurons <= 8 else ('...' if i == n_draw//2 else str(i+1))
                ax.text(x, yp, label, ha='center', va='center', fontsize=7)
            ax.add_patch(c)
            layer_pos.append((x, yp))

        if n_neurons > 8:
            ax.text(x, -0.5, f'({n_neurons})', ha='center', fontsize=8)

        ax.text(x, max_n + 0.3, f'{name}\n({n_neurons})', ha='center', fontsize=9, fontweight='bold')
        positions.append(layer_pos)

    # Связи
    for l in range(len(positions) - 1):
        for p1 in positions[l]:
            for p2 in positions[l + 1]:
                ax.plot([p1[0]+0.25, p2[0]-0.25], [p1[1], p2[1]], 'gray', lw=0.2, alpha=0.3)

    total_params = sum((layer_sizes[i]+1) * layer_sizes[i+1] for i in range(len(layer_sizes)-1))
    ax.text(len(layer_sizes)*1.5, -0.8, f'Всего параметров: {total_params}',
            ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig


# ============================================================
# 8. Главная функция
# ============================================================

def main():
    print("=" * 80)
    print("КЕЙС №4: СЛОЖНЫЙ ПЕРЦЕПТРОН (FEED-FORWARD NN) — BACKPROPAGATION")
    print("=" * 80)

    os.makedirs("plots", exist_ok=True)
    np.random.seed(42)

    # --- Параметры ---
    N_IN = 12
    N_OUT = 10          # предсказываем 10 точек (требование задания)
    N_CROSS = 5         # больше перекрытие = стабильнее прогноз
    DELTA = 0.05
    MAX_ITER = 8000
    LR = 0.01
    BATCH_SIZE = 32
    AUDIT_EVERY = 50
    EARLY_STOP = 2000

    # Конфигурации скрытых слоёв для сравнения
    HIDDEN_CONFIGS = [5, 10, 20]

    print(f"\nПараметры:")
    print(f"  N_in={N_IN}, N_out={N_OUT}, N_cross={N_CROSS}")
    print(f"  Нормировка: [{DELTA}, {1-DELTA}]")
    print(f"  lr={LR}, batch_size={BATCH_SIZE}, max_iter={MAX_ITER}")
    print(f"  Скрытые слои для сравнения: {HIDDEN_CONFIGS}")

    # --- Загрузка ---
    print("\n" + "=" * 80)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 80)

    filepath = 'Heineken NV Stock Price History.csv'
    if not os.path.exists(filepath):
        filepath = 'Heineken_NV_Stock_Price_History.csv'
    if not os.path.exists(filepath):
        uploads = '/mnt/user-data/uploads'
        if os.path.exists(uploads):
            for f in os.listdir(uploads):
                if 'heineken' in f.lower() or 'stock' in f.lower():
                    filepath = os.path.join(uploads, f)
                    break

    y_raw = load_data(filepath)
    y_norm, norm_params = normalize(y_raw, delta=DELTA)
    print(f"  Диапазон: [{np.min(y_raw):.2f}, {np.max(y_raw):.2f}]")

    # --- Разбиение 90/10 ---
    split_idx = int(len(y_norm) * 0.90)
    y_train_full = y_norm[:split_idx]
    y_test_full = y_norm[split_idx:]

    X_train, T_train = build_dataset(y_train_full, N_IN, N_OUT, N_CROSS)

    # Тестовый датасет (из конца train + test)
    X_test, T_test = build_dataset(y_norm[split_idx - N_IN:], N_IN, N_OUT, N_CROSS)

    print(f"  Train: {X_train.shape[0]} элементов, Test: {X_test.shape[0]} элементов")

    # ================================================================
    # ЧАСТЬ 1: Проверка Backpropagation
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 1: ВЕРИФИКАЦИЯ BACKPROPAGATION")
    print("=" * 80)

    net_check = MultiLayerPerceptron([N_IN, 5, N_OUT])
    net_check.init_weights(method='small_random')

    grads_a = net_check.backprop(X_train[:3], T_train[:3])
    grads_n = net_check.backprop_numerical(X_train[:3], T_train[:3])

    for layer in range(len(grads_a)):
        max_diff = np.max(np.abs(grads_a[layer] - grads_n[layer]))
        denom = np.maximum(np.abs(grads_a[layer]), np.abs(grads_n[layer])) + 1e-12
        rel_diff = np.max(np.abs(grads_a[layer] - grads_n[layer]) / denom)
        print(f"  Слой {layer+1} ({grads_a[layer].shape}): "
              f"max|diff|={max_diff:.2e}, max_rel={rel_diff:.2e}", end='')
        print("  ✓" if rel_diff < 1e-3 else "  ⚠")

    # ================================================================
    # ЧАСТЬ 2: Сравнение архитектур
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 2: СРАВНЕНИЕ АРХИТЕКТУР (скрытый слой: 5, 10, 20 нейронов)")
    print("=" * 80)

    trained_nets = []
    net_labels = []
    y_forecasts = []

    for n_hid in HIDDEN_CONFIGS:
        layers = [N_IN, n_hid, N_OUT]
        total_params = sum((layers[i]+1)*layers[i+1] for i in range(len(layers)-1))
        label = f'Hid={n_hid} ({total_params} пар.)'
        print(f"\n  --- {label} ---")

        net = MultiLayerPerceptron(layers)
        net.init_weights(method='xavier')

        net.train(X_train, T_train, X_test, T_test,
                  lr=LR, max_iter=MAX_ITER, batch_size=BATCH_SIZE,
                  audit_every=AUDIT_EVERY, early_stop_patience=EARLY_STOP)

        trained_nets.append(net)
        net_labels.append(label)

        # Прогноз
        forecast_len = int(len(y_raw) * 0.15)
        fore_norm = forecast_future(net, y_norm, forecast_len, N_IN, N_OUT, N_CROSS)
        y_forecasts.append(denormalize(fore_norm, norm_params))

    # Лучшая сеть по E_test
    best_idx = 0
    best_e = float('inf')
    for i, net in enumerate(trained_nets):
        if net.history_E_test:
            e = min(net.history_E_test)
            if e < best_e:
                best_e = e
                best_idx = i
    best_net = trained_nets[best_idx]
    print(f"\n  Лучшая архитектура: {net_labels[best_idx]}")

    # ================================================================
    # ЧАСТЬ 3: Метрики
    # ================================================================
    print("\n" + "=" * 80)
    print("МЕТРИКИ")
    print("=" * 80)

    for net, label in zip(trained_nets, net_labels):
        pred_tr = net.predict(X_train)
        rmse_tr = np.sqrt(np.mean((pred_tr - T_train)**2))
        pred_te = net.predict(X_test)
        rmse_te = np.sqrt(np.mean((pred_te - T_test)**2))
        print(f"  {label}: RMSE_train={rmse_tr:.4f}, RMSE_test={rmse_te:.4f}")

    # ================================================================
    # ЧАСТЬ 4: Эксперименты с λ
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 4: ЭКСПЕРИМЕНТЫ С lambda")
    print("=" * 80)

    lambdas = [0.001, 0.005, 0.01, 0.05, 0.1]
    for lr_val in lambdas:
        net_tmp = MultiLayerPerceptron([N_IN, 10, N_OUT])
        net_tmp.init_weights(method='xavier')
        net_tmp.train(X_train, T_train, X_test, T_test,
                      lr=lr_val, max_iter=3000, batch_size=BATCH_SIZE,
                      audit_every=50, early_stop_patience=1000, verbose=False)
        e_tr = net_tmp.history_E_train[-1] if net_tmp.history_E_train else float('inf')
        e_te = min(net_tmp.history_E_test) if net_tmp.history_E_test else float('inf')
        print(f"  lambda={lr_val:<6.3f} -> E_train={e_tr:.4f}, best_E_test={e_te:.4f}")

    # ================================================================
    # ЧАСТЬ 5: Эксперимент с двумя скрытыми слоями
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 5: ДВА СКРЫТЫХ СЛОЯ")
    print("=" * 80)

    deep_configs = [
        [N_IN, 10, 5, N_OUT],
        [N_IN, 15, 10, N_OUT],
        [N_IN, 20, 10, N_OUT],
    ]
    for layers in deep_configs:
        total_p = sum((layers[i]+1)*layers[i+1] for i in range(len(layers)-1))
        label = f'{layers} ({total_p} пар.)'
        net_tmp = MultiLayerPerceptron(layers)
        net_tmp.init_weights(method='xavier')
        net_tmp.train(X_train, T_train, X_test, T_test,
                      lr=LR, max_iter=3000, batch_size=BATCH_SIZE,
                      audit_every=50, early_stop_patience=1000, verbose=False)
        e_te = min(net_tmp.history_E_test) if net_tmp.history_E_test else float('inf')
        pred_te = net_tmp.predict(X_test)
        rmse_te = np.sqrt(np.mean((pred_te - T_test)**2))
        print(f"  {label}: best_E_test={e_te:.4f}, RMSE_test={rmse_te:.4f}")

    # ================================================================
    # ГРАФИКИ
    # ================================================================
    print("\n" + "=" * 80)
    print("ПОСТРОЕНИЕ ГРАФИКОВ")
    print("=" * 80)

    # 1. Основные результаты
    fig1 = plot_all_results(y_raw, y_norm, norm_params,
                            trained_nets, net_labels,
                            N_IN, N_OUT, N_CROSS, split_idx,
                            y_forecasts, best_net)
    fig1.savefig('plots/mlp_results.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: plots/mlp_results.png")
    plt.close(fig1)

    # 2. Проверка градиента
    fig2 = plot_gradient_check(net_check, X_train[:5], T_train[:5])
    fig2.savefig('plots/backprop_check.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: plots/backprop_check.png")
    plt.close(fig2)

    # 3. Схема сети
    fig3 = plot_network_scheme([N_IN, HIDDEN_CONFIGS[best_idx], N_OUT])
    fig3.savefig('plots/network_scheme.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: plots/network_scheme.png")
    plt.close(fig3)

    # --- Сохранение ---
    results = {
        'параметры': {
            'N_in': N_IN, 'N_out': N_OUT, 'N_cross': N_CROSS,
            'delta': DELTA, 'lr': LR, 'batch_size': BATCH_SIZE,
            'max_iter': MAX_ITER
        },
        'сравнение_архитектур': {}
    }
    for net, label in zip(trained_nets, net_labels):
        pred_te = net.predict(X_test)
        results['сравнение_архитектур'][label] = {
            'RMSE_test': float(np.sqrt(np.mean((pred_te - T_test)**2))),
            'best_E_test': float(min(net.history_E_test)) if net.history_E_test else None
        }

    with open('plots/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("  Сохранено: plots/results.json")

    print("\n" + "=" * 80)
    print("ГОТОВО!")
    print("=" * 80)


if __name__ == "__main__":
    main()
