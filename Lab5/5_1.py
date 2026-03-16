"""
Кейс №5 — Ограниченная машина Больцмана (RBM)
Версия 2.0: С калибровкой прогноза и сглаживанием шума Монте-Карло.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (14, 8)


def sigmoid(x):
    x_clip = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clip))


class RestrictedBoltzmannMachine:
    def __init__(self, n_vis, n_hid):
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.W = np.random.normal(0, 0.05, (n_vis, n_hid))
        self.a = np.zeros(n_vis)
        self.b = np.zeros(n_hid)

    def sample_prob(self, probs):
        return (probs > np.random.rand(*probs.shape)).astype(float)

    def forward(self, v):
        return sigmoid(v @ self.W + self.b)

    def backward(self, h):
        return sigmoid(h @ self.W.T + self.a)

    def train(self, data, lr=0.01, epochs=2000, batch_size=32):
        errors = []
        for ep in range(epochs):
            np.random.shuffle(data)
            batch_errors = []
            for i in range(0, len(data), batch_size):
                v0 = data[i:i + batch_size]

                # CD-1 (Contrastive Divergence)
                ph0 = self.forward(v0)
                h0 = self.sample_prob(ph0)

                pv1 = self.backward(h0)
                v1 = self.sample_prob(pv1)
                ph1 = self.forward(v1)
                h1 = self.sample_prob(ph1)

                dW = (v0.T @ ph0 - v1.T @ ph1) / len(v0)
                da = np.mean(v0 - v1, axis=0)
                db = np.mean(ph0 - ph1, axis=0)

                self.W += lr * dW
                self.a += lr * da
                self.b += lr * db

                batch_errors.append(np.mean((v0 - pv1) ** 2))

            errors.append(np.mean(batch_errors))
            if ep % 200 == 0 or ep == epochs - 1:
                print(f"Итерация {ep}: Ошибка реконструкции = {errors[-1]:.4f}")
        return errors

    def extrapolate(self, known_v, missing_mask, iterations=2000):
        results = []
        v_current = known_v.copy()
        v_current[missing_mask] = 0.0

        for _ in range(iterations):
            ph = self.forward(v_current)
            h = self.sample_prob(ph)
            pv = self.backward(h)

            # Сохраняем именно вероятности для меньшего шума
            results.append(pv)

            # Обновляем известные точки, чтобы они не размывались
            v_current = pv.copy()
            v_current[~missing_mask] = known_v[~missing_mask]

        return np.mean(results, axis=0)


def build_dataset(series, window_size):
    D = len(series) - window_size + 1
    X_data = np.zeros((D, window_size))
    for d in range(D):
        X_data[d, :] = series[d: d + window_size]
    return X_data


def normalize(data):
    dmin, dmax = np.min(data), np.max(data)
    rng = dmax - dmin if (dmax - dmin) > 1e-12 else 1.0
    return (data - dmin) / rng, (dmin, rng)


def denormalize(normed, params):
    dmin, rng = params
    return normed * rng + dmin


def smooth_curve(points, factor=0.6):
    """Экспоненциальное сглаживание для устранения шума Монте-Карло"""
    smoothed = []
    for point in points:
        if smoothed:
            prev = smoothed[-1]
            smoothed.append(prev * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return np.array(smoothed)


def main():
    print("=" * 60)
    print("КЕЙС №5: ОГРАНИЧЕННАЯ МАШИНА БОЛЬЦМАНА (RBM) - V2 (Улучшенная)")
    print("=" * 60)

    WINDOW_SIZE = 22
    N_KNOWN = 12
    N_HIDDEN = 100  # Увеличили размерность скрытого слоя
    EPOCHS = 3000  # Увеличили количество итераций
    LR = 0.02

    # 1. Загрузка данных
    filepath = 'Heineken NV Stock Price History.csv'
    if not os.path.exists(filepath):
        print("Файл не найден! Используется синтетика.")
        np.random.seed(42)
        N = 300
        t = np.linspace(0, 6 * np.pi, N)
        y_raw = 50 + 15 * np.sin(0.5 * t) + 7 * np.cos(1.3 * t) + np.cumsum(np.random.normal(0, 0.3, N))
    else:
        df = pd.read_csv(filepath)
        col = [c for c in df.columns if 'Price' in c or 'Close' in c]
        raw = df[col[0]] if col else df.iloc[:, 1]
        y_raw = pd.to_numeric(raw.astype(str).str.replace(',', '').str.strip('"').str.strip(), errors='coerce').values
        y_raw = y_raw[~np.isnan(y_raw)][::-1]

    y_norm, norm_params = normalize(y_raw)

    split_idx = int(len(y_norm) * 0.90)
    y_train = y_norm[:split_idx]

    X_train = build_dataset(y_train, WINDOW_SIZE)

    # 2. Обучение
    rbm = RestrictedBoltzmannMachine(n_vis=WINDOW_SIZE, n_hid=N_HIDDEN)
    print("\nОбучение RBM запущено...")
    rbm.train(X_train, lr=LR, epochs=EPOCHS, batch_size=32)

    # 3. Экстраполяция
    print("\nЭкстраполяция (Монте-Карло 2000 итераций)...")
    known_sequence = y_norm[split_idx - N_KNOWN: split_idx]

    v_input = np.zeros(WINDOW_SIZE)
    v_input[:N_KNOWN] = known_sequence
    missing_mask = np.array([False] * N_KNOWN + [True] * (WINDOW_SIZE - N_KNOWN))

    reconstructed_v = rbm.extrapolate(v_input, missing_mask, iterations=2000)
    forecast_norm = reconstructed_v[N_KNOWN:]

    # Денормализация
    forecast_raw = denormalize(forecast_norm, norm_params)

    # --- УЛУЧШЕНИЕ 1: Сглаживание ---
    forecast_smoothed = smooth_curve(forecast_raw, factor=0.7)

    # --- УЛУЧШЕНИЕ 2: Калибровка (Сдвиг) ---
    # Находим разницу между последней известной точкой и первой точкой прогноза
    last_known_price = y_raw[split_idx - 1]
    shift = last_known_price - forecast_smoothed[0]

    # Сдвигаем весь прогноз, чтобы он бесшовно соединялся с графиком
    forecast_calibrated = forecast_smoothed + shift

    # 4. Визуализация
    plt.figure(figsize=(15, 6))
    plt.plot(np.arange(len(y_raw)), y_raw, 'k-', alpha=0.6, label='Фактические данные')
    plt.axvline(x=split_idx, color='r', linestyle='--', label='Граница Train/Test')

    x_forecast = np.arange(split_idx, split_idx + len(forecast_calibrated))
    plt.plot(x_forecast, forecast_calibrated, 'b-o', linewidth=2.5, markersize=5,
             label='Улучшенный прогноз RBM (со сдвигом)')

    # Покажем для сравнения, каким был прогноз "до" калибровки (полупрозрачным)
    plt.plot(x_forecast, forecast_raw, 'g--', alpha=0.3, label='Сырой прогноз RBM (до калибровки)')

    plt.title('Экстраполяция временного ряда (RBM). Версия с калибровкой и сглаживанием', fontweight='bold',
              fontsize=14)
    plt.ylabel('Цена акции', fontsize=12)
    plt.xlabel('Временные отсчеты', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.4, linestyle='--')

    # Увеличим масштаб в области прогноза для наглядности
    plt.xlim(split_idx - 50, split_idx + len(forecast_calibrated) + 5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()