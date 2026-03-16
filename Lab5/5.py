"""
Кейс №5 — Ограниченная машина Больцмана (RBM) для экстраполяции временного ряда.
Обучение по алгоритму Contrastive Divergence (CD-1) Джеффри Хинтона.
Сэмплирование методом Монте-Карло, реконструкция и усреднение прогнозов.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (14, 10)


def sigmoid(x):
    """Сигмоида: f(s) = 1 / (1 + exp(-s))"""
    x_clip = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clip))


class RestrictedBoltzmannMachine:
    def __init__(self, n_vis, n_hid):
        self.n_vis = n_vis
        self.n_hid = n_hid

        # Инициализация весов малым случайным шумом
        self.W = np.random.normal(0, 0.1, (n_vis, n_hid))
        self.a = np.zeros(n_vis)  # Смещение видимого слоя
        self.b = np.zeros(n_hid)  # Смещение скрытого слоя

    def sample_prob(self, probs):
        """Операция сэмплирования Монте-Карло: превращает вероятности в конкретные состояния 0 или 1"""
        return (probs > np.random.rand(*probs.shape)).astype(float)

    def forward(self, v):
        """Прямой прогон: P(H=1|V)"""
        return sigmoid(v @ self.W + self.b)

    def backward(self, h):
        """Обратный прогон: P(V=1|H)"""
        return sigmoid(h @ self.W.T + self.a)

    def train(self, data, lr=0.01, epochs=1000, batch_size=32):
        """Обучение алгоритмом CD-1 (Contrastive Divergence) по мини-батчам"""
        errors = []
        for ep in range(epochs):
            np.random.shuffle(data)
            batch_errors = []

            for i in range(0, len(data), batch_size):
                v0 = data[i:i + batch_size]

                # --- Матожидание по данным (нулевой порядок) ---
                ph0 = self.forward(v0)
                h0 = self.sample_prob(ph0)

                # --- Матожидание по модели (первый порядок, пинг-понг Хинтона) ---
                pv1 = self.backward(h0)
                v1 = self.sample_prob(pv1)
                ph1 = self.forward(v1)
                h1 = self.sample_prob(
                    ph1)  # Хотя Хинтон рекомендует использовать вероятности ph1 для обновления весов, сэмплируем для чистоты эксперимента

                # --- Обновление весов и смещений ---
                dW = (v0.T @ ph0 - v1.T @ ph1) / len(v0)
                da = np.mean(v0 - v1, axis=0)
                db = np.mean(ph0 - ph1, axis=0)

                self.W += lr * dW
                self.a += lr * da
                self.b += lr * db

                # Расчет ошибки реконструкции (для аудита)
                batch_errors.append(np.mean((v0 - pv1) ** 2))

            errors.append(np.mean(batch_errors))

            if ep % 100 == 0 or ep == epochs - 1:
                print(f"Итерация {ep}: Ошибка реконструкции = {errors[-1]:.4f}")

        return errors

    def extrapolate(self, known_v, missing_mask, iterations=1000):
        """
        Экстраполяция (поиск рекомендаций/реконструкция).
        Подаем известные данные, неизвестные зануляем. Прогоняем вперед-назад 1000 раз.
        """
        results = []
        v_current = known_v.copy()
        v_current[missing_mask] = 0.0  # Зануляем неизвестные "правые" точки

        for _ in range(iterations):
            # Прямой прогон
            ph = self.forward(v_current)
            h = self.sample_prob(ph)

            # Обратный прогон
            pv = self.backward(h)

            # Сохраняем вероятности (рекомендуется для снижения шума) или сэмплированные значения
            results.append(pv)

        # Возвращаем матожидание (усреднение) по всем итерациям
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


def main():
    print("=" * 60)
    print("КЕЙС №5: ОГРАНИЧЕННАЯ МАШИНА БОЛЬЦМАНА (RBM)")
    print("=" * 60)

    # Параметры
    WINDOW_SIZE = 22  # 12 известных точек + 10 экстраполируемых
    N_KNOWN = 12
    N_HIDDEN = 50
    EPOCHS = 2000
    LR = 0.05

    # 1. Загрузка и подготовка данных
    filepath = 'Heineken NV Stock Price History.csv'
    if not os.path.exists(filepath):
        print("Файл не найден, генерация синтетического ряда...")
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

    # Датасет для обучения (только train)
    split_idx = int(len(y_norm) * 0.90)
    y_train = y_norm[:split_idx]

    X_train = build_dataset(y_train, WINDOW_SIZE)
    print(f"Обучающая выборка: {X_train.shape[0]} окон")

    # 2. Обучение RBM
    rbm = RestrictedBoltzmannMachine(n_vis=WINDOW_SIZE, n_hid=N_HIDDEN)
    print("\nНачало обучения RBM...")
    errors = rbm.train(X_train, lr=LR, epochs=EPOCHS, batch_size=32)

    # 3. Экстраполяция на тестовом участке
    print("\nЭкстраполяция (Монте-Карло 1000 итераций)...")

    # Берем последние известные точки из обучающей выборки
    known_sequence = y_norm[split_idx - N_KNOWN: split_idx]

    # Формируем вектор для RBM
    v_input = np.zeros(WINDOW_SIZE)
    v_input[:N_KNOWN] = known_sequence

    # Маска: True там, где данные неизвестны (правые точки)
    missing_mask = np.array([False] * N_KNOWN + [True] * (WINDOW_SIZE - N_KNOWN))

    # Реконструкция
    reconstructed_v = rbm.extrapolate(v_input, missing_mask, iterations=1000)

    # Извлекаем предсказанную часть
    forecast_norm = reconstructed_v[N_KNOWN:]
    forecast = denormalize(forecast_norm, norm_params)

    # 4. Визуализация
    plt.figure(figsize=(15, 6))
    plt.plot(np.arange(len(y_raw)), y_raw, 'k-', alpha=0.5, label='Фактические данные')
    plt.axvline(x=split_idx, color='r', linestyle='--', label='Граница Train/Test')

    x_forecast = np.arange(split_idx, split_idx + len(forecast))
    plt.plot(x_forecast, forecast, 'r-o', linewidth=2, label='Экстраполяция RBM')

    plt.title('Прогноз временного ряда моделью RBM (1000 итераций Монте-Карло)', fontweight='bold')
    plt.ylabel('Цена')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    print("Экстраполяция успешно завершена.")


if __name__ == "__main__":
    main()
