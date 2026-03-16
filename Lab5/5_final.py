"""
Кейс №5 — Ограниченная машина Больцмана (RBM): экстраполяция временного ряда

Restricted Boltzmann Machine — ассоциативная нейронная сеть
со стохастическими нейронами и алгоритмом обучения (приближение Хинтона).

Архитектура: Visible (N_v) ↔ Hidden (N_h)
    - Нет связей внутри слоёв
    - Полносвязные связи между слоями (матрица W)
    - У каждого нейрона своё смещение: a_i (visible), b_j (hidden)

Модель стохастического нейрона (лекция):

  Суммарный сигнал:
    S = a + Σ_i X_i · W_ij         (для hidden-нейрона j)
    S = b + Σ_j H_j · W_ij         (для visible-нейрона i)

  Вероятность активации (сигмоида):
    P(h_j = 1 | V, W) = σ(b_j + Σ_i V_i · W_ij)
    P(v_i = 1 | H, W) = σ(a_i + Σ_j H_j · W_ij)

  Сэмплирование (метод Монте-Карло):
    h_j = 1 если P(h_j=1) > r,  иначе 0   (r ~ Uniform(0,1))

Обучение — приближение Хинтона порядка k:

  Правило обновления весов (градиентный подъём):
    ΔW_ij = λ · (⟨V_i · H_j⟩_data − ⟨V_i · H_j⟩_model)
    Δa_i  = λ · (⟨V_i⟩_data       − ⟨V_i⟩_model)
    Δb_j  = λ · (⟨H_j⟩_data       − ⟨H_j⟩_model)

  Приближение Хинтона порядка 1 (ε=1):
    ⟨·⟩_data  ≈ среднее по мини-батчу пар (V⁰, H⁰) порядка 0
    ⟨·⟩_model ≈ среднее по мини-батчу пар (V¹, H¹) порядка 1
      где V⁰ — элемент датасета, H⁰ — сэмплировано от V⁰,
          V¹ — реконструировано от H⁰, H¹ — сэмплировано от V¹.

Экстраполяция временного ряда (лекция):
  1. Кодируем N_in правых точек ряда в бинарный вектор V (видимый слой).
  2. Прямой прогон V → H (сэмплирование).
  3. Сдвигаем данные: крайние точки → влево, правая часть → нули.
  4. Обратный прогон H → V' (реконструкция).
  5. Повторяем процедуру K_avg раз, усредняем → мат.ожидание.
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
# 1. Функция активации (сигмоида) — модель Больцмана
# ============================================================

def sigmoid(x):
    """σ(s) = 1 / (1 + exp(−s))  — вероятность P(x=1)"""
    x_clip = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clip))


# ============================================================
# 2. Класс RBM — Ограниченная машина Больцмана
# ============================================================

class RBM:
    """
    Restricted Boltzmann Machine (Ограниченная машина Больцмана).

    Два слоя: visible (V) и hidden (H).
    Параметры: W (матрица связей), a (смещения visible), b (смещения hidden).

    Обучение: приближение Хинтона порядка k.

    Формулы с лекции:
        Прямой прогон:  P(h_j=1 | V) = σ(b_j + Σ_i V_i · W_ij)
        Обратный прогон: P(v_i=1 | H) = σ(a_i + Σ_j H_j · W_ij)
        Сэмплирование:  x = 1 если P > r (r ~ U(0,1)), иначе 0

    Правило обновления (приближение Хинтона):
        ΔW = λ · (⟨V·Hᵀ⟩_data − ⟨V·Hᵀ⟩_model)
        Δa = λ · (⟨V⟩_data − ⟨V⟩_model)
        Δb = λ · (⟨H⟩_data − ⟨H⟩_model)
    """

    def __init__(self, n_visible, n_hidden):
        """
        n_visible: число нейронов видимого слоя
        n_hidden:  число нейронов скрытого слоя
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = None   # матрица весов W_{ij}
        self.a = None   # смещения видимого слоя a_i
        self.b = None   # смещения скрытого слоя b_j
        self.history_error = []

    def init_weights(self, method='small_random'):
        """Инициализация параметров W, a, b."""
        if method == 'small_random':
            self.W = np.random.uniform(-0.01, 0.01,
                                       (self.n_visible, self.n_hidden))
        elif method == 'xavier':
            limit = np.sqrt(6.0 / (self.n_visible + self.n_hidden))
            self.W = np.random.uniform(-limit, limit,
                                       (self.n_visible, self.n_hidden))
        elif method == 'zeros':
            self.W = np.zeros((self.n_visible, self.n_hidden))
        else:
            self.W = np.random.randn(self.n_visible, self.n_hidden) * 0.01

        self.a = np.zeros(self.n_visible)    # a_i = 0
        self.b = np.zeros(self.n_hidden)     # b_j = 0

        return self.W.copy(), self.a.copy(), self.b.copy()

    # ----------------------------------------------------------
    # Прямой прогон: V → H
    # ----------------------------------------------------------
    def v_to_h_prob(self, V):
        """
        P(h_j = 1 | V, W) = σ(b_j + Σ_i V_i · W_ij)

        V: shape (D, n_visible)
        Возвращает: P_h shape (D, n_hidden)
        """
        return sigmoid(V @ self.W + self.b)

    def v_to_h_sample(self, V):
        """
        Сэмплирование H из P(H|V).
        Метод Монте-Карло: h_j = 1 если P(h_j=1) > r, иначе 0.
        """
        P_h = self.v_to_h_prob(V)
        return (P_h > np.random.rand(*P_h.shape)).astype(np.float64), P_h

    # ----------------------------------------------------------
    # Обратный прогон: H → V
    # ----------------------------------------------------------
    def h_to_v_prob(self, H):
        """
        P(v_i = 1 | H, W) = σ(a_i + Σ_j H_j · W_ij)

        H: shape (D, n_hidden)
        Возвращает: P_v shape (D, n_visible)
        """
        return sigmoid(H @ self.W.T + self.a)

    def h_to_v_sample(self, H):
        """
        Сэмплирование V из P(V|H).
        Для непрерывных данных используем саму вероятность (среднее поле).
        """
        P_v = self.h_to_v_prob(H)
        # Для непрерывных данных лучше возвращать вероятность, а не бинарный сэмпл
        return P_v, P_v

    # ----------------------------------------------------------
    # Приближение Хинтона порядка k
    # ----------------------------------------------------------
    def hinton_step(self, V_data, k=1):
        """
        Один шаг приближения Хинтона порядка k.

        Алгоритм (лекция):
          1. Порядок 0 (по данным):
             V⁰ = V_data (из датасета)
             H⁰ ~ P(H | V⁰)       — сэмплируем

          2. Пинг-понг k шагов:
             for step in 1..k:
                V^step ~ P(V | H^{step-1})
                H^step ~ P(H | V^step)

          3. Порядок k (по модели):
             V^k, H^k — пара порядка k

          4. Поправки:
             ΔW = ⟨V⁰·H⁰ᵀ⟩ − ⟨V^k·H^kᵀ⟩
             Δa = ⟨V⁰⟩ − ⟨V^k⟩
             Δb = ⟨H⁰⟩ − ⟨H^k⟩

        Возвращает: dW, da, db, ошибка реконструкции
        """
        D = V_data.shape[0]

        # --- Порядок 0 (по данным) ---
        H0_sample, H0_prob = self.v_to_h_sample(V_data)

        # --- Пинг-понг k шагов ---
        V_k = V_data.copy()
        H_k = H0_sample.copy()
        H_k_prob = H0_prob.copy()

        for step in range(k):
            V_k, V_k_prob = self.h_to_v_sample(H_k)
            H_k, H_k_prob = self.v_to_h_sample(V_k)

        # --- Поправки (формулы с лекции) ---
        # ΔW = (1/D) · (V⁰ᵀ · H⁰_prob − V^kᵀ · H^k_prob)
        dW = (V_data.T @ H0_prob - V_k.T @ H_k_prob) / D

        # Δa = (1/D) · Σ_m (V⁰_m − V^k_m)
        da = np.mean(V_data - V_k, axis=0)

        # Δb = (1/D) · Σ_m (H⁰_prob_m − H^k_prob_m)
        db = np.mean(H0_prob - H_k_prob, axis=0)

        # Ошибка реконструкции (для аудита)
        recon_error = np.mean((V_data - V_k) ** 2)

        return dW, da, db, recon_error

    # ----------------------------------------------------------
    # Обучение
    # ----------------------------------------------------------
    def train(self, X_train, lr=0.01, max_iter=5000, batch_size=None,
              k=1, audit_every=100, momentum=0.5, weight_decay=0.0001,
              verbose=True):
        """
        Обучение RBM приближением Хинтона с мини-батчами.

        X_train:      данные shape (D, n_visible)
        lr:           скорость обучения (λ)
        k:            порядок CD (обычно k=1)
        batch_size:   размер мини-батча
        momentum:     импульс для ускорения сходимости
        weight_decay: регуляризация весов
        """
        self.history_error = []
        D = X_train.shape[0]
        if batch_size is None:
            batch_size = D

        # Инициализация импульса
        dW_prev = np.zeros_like(self.W)
        da_prev = np.zeros_like(self.a)
        db_prev = np.zeros_like(self.b)

        for it in range(max_iter):
            # --- Перемешиваем и берём мини-батч ---
            indices = np.random.permutation(D)
            epoch_error = 0.0
            n_batches = 0

            for start in range(0, D, batch_size):
                end = min(start + batch_size, D)
                batch_idx = indices[start:end]
                V_batch = X_train[batch_idx]

                # Приближение Хинтона порядка k
                dW, da, db, recon_err = self.hinton_step(V_batch, k=k)
                epoch_error += recon_err
                n_batches += 1

                # Обновление с импульсом и регуляризацией
                dW_prev = momentum * dW_prev + lr * (dW - weight_decay * self.W)
                da_prev = momentum * da_prev + lr * da
                db_prev = momentum * db_prev + lr * db

                self.W += dW_prev
                self.a += da_prev
                self.b += db_prev

            # --- Аудит ---
            avg_error = epoch_error / n_batches
            if it % audit_every == 0 or it == max_iter - 1:
                self.history_error.append(avg_error)
                if verbose and (it % (audit_every * 5) == 0 or it == max_iter - 1):
                    print(f"    Итерация {it:5d}: recon_error={avg_error:.6f}")

        return self.history_error[-1] if self.history_error else float('inf')

    # ----------------------------------------------------------
    # Реконструкция (прямой + обратный прогон)
    # ----------------------------------------------------------
    def reconstruct(self, V, n_gibbs=1):
        """
        Реконструкция: V → H → V'
        n_gibbs: число шагов Гиббса (пинг-понга).
        """
        H, _ = self.v_to_h_sample(V)
        for _ in range(n_gibbs):
            V_recon, _ = self.h_to_v_sample(H)
            H, _ = self.v_to_h_sample(V_recon)
        V_recon, _ = self.h_to_v_sample(H)
        return V_recon

    def reconstruct_mean(self, V, n_avg=100, n_gibbs=1):
        """
        Реконструкция с усреднением (мат.ожидание).
        Лекция: «каждую точку получаете не один раз, а тысячу раз,
                  и усредняете — мат.ожидание».
        """
        accum = np.zeros_like(V)
        for _ in range(n_avg):
            accum += self.reconstruct(V, n_gibbs)
        return accum / n_avg


# ============================================================
# 3. Бинаризация данных для RBM
# ============================================================

def float_to_binary(values, n_bits=8):
    """
    Кодируем каждое значение [0,1] в бинарный вектор длины n_bits.
    Термометрическое кодирование: value=0.6, n_bits=8 → [1,1,1,1,1,0,0,0]
    """
    result = np.zeros((len(values), n_bits))
    for i, val in enumerate(values):
        n_ones = int(round(val * n_bits))
        n_ones = max(0, min(n_bits, n_ones))
        result[i, :n_ones] = 1.0
    return result


def binary_to_float(binary, n_bits=8):
    """Декодируем бинарный вектор обратно в [0,1]."""
    return np.sum(binary.reshape(-1, n_bits), axis=1) / n_bits


def encode_window(window, n_bits=8):
    """Кодируем окно из нормированных значений в бинарный вектор."""
    bits = float_to_binary(window, n_bits)
    return bits.flatten()


def decode_window(binary_vector, n_points, n_bits=8):
    """Декодируем бинарный вектор обратно в значения."""
    return binary_to_float(binary_vector, n_bits)


# ============================================================
# 4. Формирование датасета (скользящее окно)
# ============================================================

def build_dataset(series, n_in=12, n_bits=8):
    """
    Скользящее окно: каждый элемент — окно из n_in точек,
    закодированное в бинарный вектор длины n_in * n_bits.
    """
    D = len(series) - n_in + 1
    if D <= 0:
        raise ValueError(f"Ряд слишком короткий ({len(series)}) для окна {n_in}")

    n_visible = n_in * n_bits
    X_data = np.zeros((D, n_visible))
    for d in range(D):
        window = series[d: d + n_in]
        X_data[d, :] = encode_window(window, n_bits)
    return X_data


# ============================================================
# 5. Нормировка
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
# 6. Загрузка данных
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
# 7. Экстраполяция через RBM
# ============================================================

def extrapolate_rbm(rbm, y_norm, n_in, n_bits, n_steps, n_avg=500, n_cross=3):
    """
    Экстраполяция временного ряда методом RBM (лекция):

    Алгоритм:
      1. Берём последние n_in точек нормированного ряда.
      2. Кодируем в бинарный вектор V (видимый слой).
      3. Прямой прогон V → H (получаем скрытое представление).
      4. Сдвигаем окно: убираем левые (n_out−n_cross) точек,
         добавляем нули справа.
      5. Обратный прогон H → V' (реконструкция).
      6. Повторяем n_avg раз и усредняем → мат.ожидание.
      7. Декодируем правую часть → новые точки прогноза.
      8. Добавляем новые точки к ряду, повторяем.

    «Модель вероятностная, если сделать один раз — будет шуметь,
     поэтому накопите мат.ожидание» — из лекции.
    """
    current = list(y_norm.copy())
    n_out = n_in - n_cross  # сколько новых точек за один шаг

    for step in range(0, n_steps, max(1, n_out)):
        # Берём последние n_in точек
        window = np.array(current[-n_in:])
        V_input = encode_window(window, n_bits).reshape(1, -1)

        # Усреднение (мат.ожидание) по n_avg реализациям
        V_accum = np.zeros_like(V_input)
        for _ in range(n_avg):
            # Прямой прогон
            H, _ = rbm.v_to_h_sample(V_input)

            # Реконструкция
            V_recon = rbm.h_to_v_prob(H)
            V_accum += V_recon

        V_mean = V_accum / n_avg

        # Декодируем всё окно
        decoded = decode_window(V_mean.flatten(), n_in, n_bits)

        # Берём новые точки (правее перекрытия)
        new_points = decoded[n_cross:]

        # Добавляем столько точек, сколько нужно
        for p in new_points:
            if len(current) - len(y_norm) >= n_steps:
                break
            current.append(float(np.clip(p, 0.01, 0.99)))

    forecast = np.array(current[len(y_norm):])[:n_steps]
    return forecast


# ============================================================
# 8. Визуализация
# ============================================================

def plot_all_results(y_raw, y_norm, norm_params, rbms, rbm_labels,
                     n_in, n_bits, train_idx, y_forecasts, best_rbm):
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

    # 3. Сходимость (ошибка реконструкции)
    ax3 = fig.add_subplot(4, 2, 3)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for idx, (rbm, label) in enumerate(zip(rbms, rbm_labels)):
        c = colors[idx % len(colors)]
        ax3.plot(rbm.history_error, '-', color=c, linewidth=1, label=label)
    ax3.set_xlabel('Аудит (каждые N итераций)')
    ax3.set_ylabel('Ошибка реконструкции (MSE)')
    ax3.set_title('Сходимость обучения RBM', fontweight='bold')
    ax3.set_yscale('log')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Реконструкция лучшей моделью
    ax4 = fig.add_subplot(4, 2, 4)
    # Восстановление ряда через прямой-обратный прогон
    X_ds = build_dataset(y_norm[:train_idx], n_in, n_bits)
    X_recon = best_rbm.reconstruct_mean(X_ds, n_avg=50)
    reproduced = np.zeros(train_idx)
    counts = np.zeros(train_idx)
    for d_idx in range(len(X_ds)):
        decoded = decode_window(X_recon[d_idx], n_in, n_bits)
        for j in range(n_in):
            pos = d_idx + j
            if pos < train_idx:
                reproduced[pos] += denormalize(decoded[j], norm_params)
                counts[pos] += 1
    mask = counts > 0
    reproduced[mask] /= counts[mask]
    reproduced[~mask] = np.nan

    ax4.plot(y_raw[:train_idx], 'k-', linewidth=0.8, alpha=0.4, label='Исходные')
    ax4.plot(reproduced, 'r-', linewidth=1.5, alpha=0.8, label='RBM (лучшая)')
    rmse = np.sqrt(np.nanmean((reproduced[mask] - y_raw[:train_idx][mask])**2))
    ax4.set_title(f'Воспроизведение на train (RMSE={rmse:.2f})', fontweight='bold')
    ax4.set_ylabel('Цена, EUR')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Сравнение архитектур (реконструкция)
    ax5 = fig.add_subplot(4, 2, 5)
    ax5.plot(y_raw[:train_idx], 'k-', linewidth=0.8, alpha=0.3, label='Исходные')
    for idx, (rbm, label) in enumerate(zip(rbms, rbm_labels)):
        c = colors[idx % len(colors)]
        X_r = rbm.reconstruct_mean(X_ds, n_avg=30)
        repr_i = np.zeros(train_idx)
        cnt_i = np.zeros(train_idx)
        for d_idx in range(len(X_ds)):
            dec = decode_window(X_r[d_idx], n_in, n_bits)
            for j in range(n_in):
                pos = d_idx + j
                if pos < train_idx:
                    repr_i[pos] += denormalize(dec[j], norm_params)
                    cnt_i[pos] += 1
        m_i = cnt_i > 0
        repr_i[m_i] /= cnt_i[m_i]
        rmse_i = np.sqrt(np.nanmean((repr_i[m_i] - y_raw[:train_idx][m_i])**2))
        ax5.plot(repr_i, '-', color=c, linewidth=1, alpha=0.7,
                 label=f'{label} (RMSE={rmse_i:.2f})')
    ax5.set_title('Сравнение архитектур RBM', fontweight='bold')
    ax5.set_ylabel('Цена, EUR')
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)

    # 6. Прогноз в будущее
    ax6 = fig.add_subplot(4, 2, 6)
    x_hist = np.arange(len(y_raw))
    ax6.plot(x_hist, y_raw, 'k-', linewidth=0.8, alpha=0.5, label='История')
    for idx, (fore, label) in enumerate(zip(y_forecasts, rbm_labels)):
        c = colors[idx % len(colors)]
        x_fore = np.arange(len(y_raw), len(y_raw) + len(fore))
        ax6.plot(x_fore, fore, '-', color=c, linewidth=2, alpha=0.7, label=f'Прогноз: {label}')
    ax6.axvline(x=len(y_raw), color='green', linestyle='--', linewidth=1.5,
                label='Начало экстраполяции')
    ax6.set_title('Экстраполяция временного ряда (RBM)', fontweight='bold')
    ax6.set_ylabel('Цена, EUR')
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3)

    # 7. Визуализация весов W
    ax7 = fig.add_subplot(4, 2, 7)
    im = ax7.imshow(best_rbm.W.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax7.set_title('Матрица весов W (лучшая RBM)', fontweight='bold')
    ax7.set_xlabel('Visible нейроны')
    ax7.set_ylabel('Hidden нейроны')
    plt.colorbar(im, ax=ax7)

    # 8. Распределение весов
    ax8 = fig.add_subplot(4, 2, 8)
    ax8.hist(best_rbm.W.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax8.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax8.set_title('Распределение весов W', fontweight='bold')
    ax8.set_xlabel('Значение веса')
    ax8.set_ylabel('Частота')
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_rbm_scheme(n_visible, n_hidden):
    """Схема архитектуры RBM."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Архитектура RBM (Ограниченная машина Больцмана)', fontweight='bold', fontsize=14)

    # Показываем максимум 8 нейронов каждого слоя
    n_v_show = min(n_visible, 8)
    n_h_show = min(n_hidden, 8)

    # Позиции
    v_x = np.linspace(1, 9, n_v_show)
    h_x = np.linspace(1.5, 8.5, n_h_show)
    v_y = 1.0
    h_y = 5.0

    # Связи
    for vx in v_x:
        for hx in h_x:
            ax.plot([vx, hx], [v_y + 0.3, h_y - 0.3],
                    '-', color='gray', alpha=0.15, linewidth=0.5)

    # Visible neurons
    for i, vx in enumerate(v_x):
        circle = plt.Circle((vx, v_y), 0.3, color='cornflowerblue',
                           ec='darkblue', linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(vx, v_y, f'V{i}', ha='center', va='center', fontsize=7, fontweight='bold')

    # Hidden neurons
    for j, hx in enumerate(h_x):
        circle = plt.Circle((hx, h_y), 0.3, color='lightcoral',
                           ec='darkred', linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(hx, h_y, f'H{j}', ha='center', va='center', fontsize=7, fontweight='bold')

    # Подписи
    ax.text(5, -0.3, f'Видимый слой (Visible): {n_visible} нейронов',
            ha='center', fontsize=11, color='darkblue')
    ax.text(5, 6.2, f'Скрытый слой (Hidden): {n_hidden} нейронов',
            ha='center', fontsize=11, color='darkred')

    # Стрелки вперёд-назад
    ax.annotate('', xy=(0.3, 3.5), xytext=(0.3, 2.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.annotate('', xy=(0.3, 2.5), xytext=(0.3, 3.5),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    ax.text(-0.5, 3.0, 'V↔H\n(симм.)', ha='center', fontsize=8, fontweight='bold')

    # Формулы
    ax.text(5, -0.8,
            r'P(h$_j$=1|V) = σ(b$_j$ + Σ V$_i$·W$_{ij}$)    |    '
            r'P(v$_i$=1|H) = σ(a$_i$ + Σ H$_j$·W$_{ij}$)',
            ha='center', fontsize=9, style='italic', color='gray')

    if n_visible > 8:
        ax.text(v_x[-1] + 0.5, v_y, '...', fontsize=14, ha='center', va='center')
    if n_hidden > 8:
        ax.text(h_x[-1] + 0.5, h_y, '...', fontsize=14, ha='center', va='center')

    plt.tight_layout()
    return fig


def plot_reconstruction_examples(rbm, X_data, y_norm, n_in, n_bits, norm_params,
                                  n_examples=5, n_avg=100):
    """Примеры реконструкции отдельных окон."""
    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 3 * n_examples))
    if n_examples == 1:
        axes = [axes]

    indices = np.linspace(0, len(X_data) - 1, n_examples, dtype=int)

    for ax, idx in zip(axes, indices):
        original = decode_window(X_data[idx], n_in, n_bits)
        original_denorm = denormalize(original, norm_params)

        # Реконструкция с усреднением
        V_in = X_data[idx:idx+1]
        V_recon = rbm.reconstruct_mean(V_in, n_avg=n_avg)
        reconstructed = decode_window(V_recon.flatten(), n_in, n_bits)
        recon_denorm = denormalize(reconstructed, norm_params)

        x_pos = np.arange(idx, idx + n_in)
        ax.plot(x_pos, original_denorm, 'ko-', markersize=4, linewidth=1.5,
                label='Оригинал', alpha=0.7)
        ax.plot(x_pos, recon_denorm, 'rs--', markersize=4, linewidth=1.5,
                label='Реконструкция RBM', alpha=0.7)
        rmse = np.sqrt(np.mean((original_denorm - recon_denorm)**2))
        ax.set_title(f'Окно #{idx} (RMSE={rmse:.2f} €)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Цена, EUR')

    axes[-1].set_xlabel('Индекс точки')
    fig.suptitle('Примеры реконструкции окон (V → H → V\')', fontweight='bold', fontsize=13)
    plt.tight_layout()
    return fig


def plot_hinton_comparison(rbm_class, X_train, n_visible, n_hidden, lr, max_iter, batch_size):
    """Сравнение порядков приближения Хинтона: ε=1, 3, 5, 10."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    colors = ['red', 'blue', 'green', 'orange']
    k_values = [1, 3, 5, 10]

    for k_val, c in zip(k_values, colors):
        rbm_tmp = rbm_class(n_visible, n_hidden)
        rbm_tmp.init_weights(method='small_random')
        rbm_tmp.train(X_train, lr=lr, max_iter=min(max_iter, 2000),
                      batch_size=batch_size, k=k_val,
                      audit_every=20, verbose=False)
        ax.plot(rbm_tmp.history_error, '-', color=c, linewidth=1.5,
                label=f'ε={k_val}')

    ax.set_xlabel('Аудит')
    ax.set_ylabel('Ошибка реконструкции')
    ax.set_title('Сравнение порядков приближения Хинтона', fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def main():
    print("=" * 80)
    print("КЕЙС №5 — ОГРАНИЧЕННАЯ МАШИНА БОЛЬЦМАНА (RBM)")
    print("Экстраполяция временного ряда стоимости акций Heineken NV")
    print("=" * 80)

    # --- Параметры ---
    N_IN = 12          # размер окна (число точек)
    N_BITS = 8         # бит на точку (термометрическое кодирование)
    N_CROSS = 3        # перекрёстные точки
    DELTA = 0.05       # отступ нормировки
    LR = 0.01          # скорость обучения λ
    MAX_ITER = 5000    # максимум итераций
    BATCH_SIZE = 32    # размер мини-батча
    HINTON_K = 1       # порядок приближения Хинтона (ε)
    HIDDEN_CONFIGS = [32, 64, 128]  # варианты числа скрытых нейронов
    N_AVG = 500        # число усреднений для экстраполяции

    N_VISIBLE = N_IN * N_BITS  # число видимых нейронов

    os.makedirs('plots', exist_ok=True)

    # --- Загрузка данных ---
    print("\n  Загрузка данных...")
    filepath = '/mnt/user-data/uploads/Heineken_NV_Stock_Price_History.csv'
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

    X_train = build_dataset(y_train_full, N_IN, N_BITS)
    print(f"  Видимый слой: {N_VISIBLE} нейронов ({N_IN} точек × {N_BITS} бит)")
    print(f"  Train: {X_train.shape[0]} элементов ({X_train.shape[1]} бит каждый)")

    # ================================================================
    # ЧАСТЬ 1: Проверка кодирования/декодирования
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 1: ВЕРИФИКАЦИЯ БИНАРНОГО КОДИРОВАНИЯ")
    print("=" * 80)

    test_vals = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    for val in test_vals:
        bits = float_to_binary(np.array([val]), N_BITS)
        restored = binary_to_float(bits, N_BITS)
        print(f"  {val:.2f} → {bits.flatten().astype(int)} → {restored[0]:.4f}"
              f"  (ошибка: {abs(val - restored[0]):.4f})")

    # Проверка полного окна
    test_window = y_norm[100:100 + N_IN]
    encoded = encode_window(test_window, N_BITS)
    decoded = decode_window(encoded, N_IN, N_BITS)
    print(f"\n  Тест полного окна (длина {N_IN}):")
    print(f"    Исходные:       {test_window[:5].round(3)} ...")
    print(f"    Декодированные: {decoded[:5].round(3)} ...")
    print(f"    Max ошибка: {np.max(np.abs(test_window - decoded)):.4f}")

    # ================================================================
    # ЧАСТЬ 2: Сравнение архитектур (разное число скрытых нейронов)
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 2: СРАВНЕНИЕ АРХИТЕКТУР RBM")
    print(f"  (скрытых нейронов: {HIDDEN_CONFIGS})")
    print("=" * 80)

    trained_rbms = []
    rbm_labels = []
    y_forecasts = []

    for n_hid in HIDDEN_CONFIGS:
        total_params = N_VISIBLE * n_hid + N_VISIBLE + n_hid
        label = f'Hid={n_hid} ({total_params} пар.)'
        print(f"\n  --- {label} ---")

        rbm = RBM(N_VISIBLE, n_hid)
        rbm.init_weights(method='small_random')

        rbm.train(X_train, lr=LR, max_iter=MAX_ITER, batch_size=BATCH_SIZE,
                  k=HINTON_K, audit_every=50, verbose=True)

        trained_rbms.append(rbm)
        rbm_labels.append(label)

        # Прогноз
        forecast_len = int(len(y_raw) * 0.15)
        fore_norm = extrapolate_rbm(rbm, y_norm, N_IN, N_BITS,
                                     forecast_len, n_avg=N_AVG, n_cross=N_CROSS)
        y_forecasts.append(denormalize(fore_norm, norm_params))

    # Лучшая RBM по ошибке реконструкции
    best_idx = 0
    best_err = float('inf')
    for i, rbm in enumerate(trained_rbms):
        if rbm.history_error:
            e = min(rbm.history_error)
            if e < best_err:
                best_err = e
                best_idx = i
    best_rbm = trained_rbms[best_idx]
    print(f"\n  Лучшая архитектура: {rbm_labels[best_idx]}")

    # ================================================================
    # ЧАСТЬ 3: Метрики реконструкции
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 3: МЕТРИКИ РЕКОНСТРУКЦИИ")
    print("=" * 80)

    for rbm, label in zip(trained_rbms, rbm_labels):
        X_recon = rbm.reconstruct_mean(X_train, n_avg=30)
        mse = np.mean((X_train - X_recon) ** 2)
        # Также считаем RMSE в исходных единицах
        reproduced = np.zeros(len(y_train_full))
        counts = np.zeros(len(y_train_full))
        for d_idx in range(len(X_train)):
            dec = decode_window(X_recon[d_idx], N_IN, N_BITS)
            for j in range(N_IN):
                pos = d_idx + j
                if pos < len(y_train_full):
                    reproduced[pos] += denormalize(dec[j], norm_params)
                    counts[pos] += 1
        m = counts > 0
        reproduced[m] /= counts[m]
        rmse = np.sqrt(np.nanmean((reproduced[m] - y_raw[:len(y_train_full)][m])**2))
        print(f"  {label}: MSE_binary={mse:.6f}, RMSE_price={rmse:.2f} €")

    # ================================================================
    # ЧАСТЬ 4: Сравнение порядков приближения Хинтона
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 4: СРАВНЕНИЕ ПОРЯДКОВ ПРИБЛИЖЕНИЯ ХИНТОНА")
    print("=" * 80)

    k_values = [1, 3, 5, 10]
    for k_val in k_values:
        rbm_tmp = RBM(N_VISIBLE, 64)
        rbm_tmp.init_weights(method='small_random')
        rbm_tmp.train(X_train, lr=LR, max_iter=2000, batch_size=BATCH_SIZE,
                      k=k_val, audit_every=50, verbose=False)
        final_err = rbm_tmp.history_error[-1] if rbm_tmp.history_error else float('inf')
        print(f"  ε={k_val:<2d}: final_recon_error={final_err:.6f}")

    # ================================================================
    # ЧАСТЬ 5: Эксперименты с λ
    # ================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 5: ЭКСПЕРИМЕНТЫ С lambda")
    print("=" * 80)

    lambdas = [0.001, 0.005, 0.01, 0.05, 0.1]
    for lr_val in lambdas:
        rbm_tmp = RBM(N_VISIBLE, 64)
        rbm_tmp.init_weights(method='small_random')
        rbm_tmp.train(X_train, lr=lr_val, max_iter=2000, batch_size=BATCH_SIZE,
                      k=1, audit_every=50, verbose=False)
        final_err = rbm_tmp.history_error[-1] if rbm_tmp.history_error else float('inf')
        min_err = min(rbm_tmp.history_error) if rbm_tmp.history_error else float('inf')
        print(f"  lambda={lr_val:<6.3f} -> final_err={final_err:.6f}, "
              f"min_err={min_err:.6f}")

    # ================================================================
    # ГРАФИКИ
    # ================================================================
    print("\n" + "=" * 80)
    print("ПОСТРОЕНИЕ ГРАФИКОВ")
    print("=" * 80)

    # 1. Основные результаты
    fig1 = plot_all_results(y_raw, y_norm, norm_params,
                            trained_rbms, rbm_labels,
                            N_IN, N_BITS, split_idx,
                            y_forecasts, best_rbm)
    fig1.savefig('plots/rbm_results.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: plots/rbm_results.png")
    plt.close(fig1)

    # 2. Схема RBM
    fig2 = plot_rbm_scheme(N_VISIBLE, HIDDEN_CONFIGS[best_idx])
    fig2.savefig('plots/rbm_scheme.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: plots/rbm_scheme.png")
    plt.close(fig2)

    # 3. Примеры реконструкции
    fig3 = plot_reconstruction_examples(best_rbm, X_train, y_norm, N_IN, N_BITS,
                                         norm_params, n_examples=5, n_avg=100)
    fig3.savefig('plots/rbm_reconstruction.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: plots/rbm_reconstruction.png")
    plt.close(fig3)

    # 4. Сравнение порядков приближения Хинтона
    fig4 = plot_hinton_comparison(RBM, X_train, N_VISIBLE, 64,
                              LR, MAX_ITER, BATCH_SIZE)
    fig4.savefig('plots/rbm_hinton_comparison.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: plots/rbm_hinton_comparison.png")
    plt.close(fig4)

    # --- Сохранение результатов ---
    results = {
        'параметры': {
            'N_in': N_IN, 'N_bits': N_BITS, 'N_cross': N_CROSS,
            'N_visible': N_VISIBLE, 'delta': DELTA,
            'lr': LR, 'batch_size': BATCH_SIZE,
            'max_iter': MAX_ITER, 'hinton_k': HINTON_K,
            'N_avg': N_AVG
        },
        'сравнение_архитектур': {}
    }
    for rbm, label in zip(trained_rbms, rbm_labels):
        results['сравнение_архитектур'][label] = {
            'final_recon_error': float(rbm.history_error[-1]) if rbm.history_error else None,
            'min_recon_error': float(min(rbm.history_error)) if rbm.history_error else None,
            'n_hidden': rbm.n_hidden
        }

    with open('plots/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("  Сохранено: plots/results.json")

    print("\n" + "=" * 80)
    print("ГОТОВО!")
    print("=" * 80)


if __name__ == "__main__":
    main()
