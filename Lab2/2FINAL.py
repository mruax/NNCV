import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import json

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10


class HarmonicModel:
    """
    Оптимизированная модель гармоник.
    Включает жадную инициализацию и адаптивный градиентный спуск.
    """
    def __init__(self, n_harmonics=5):
        self.n_harmonics = n_harmonics
        self.params = None
        self.n_params = 1 + n_harmonics * 3
        self.history = {'method1': [], 'method2': []}

    def model(self, t, params):
        """Формула 1: m(t) = B + Σ A_k * cos(ω_k * t + δ_k)"""
        B = params[0]
        result = np.full_like(t, B, dtype=float)
        for k in range(self.n_harmonics):
            idx = 1 + k * 3
            A = params[idx]
            w = params[idx + 1]
            d = params[idx + 2]
            result += A * np.cos(w * t + d)
        return result

    def error_E(self, params, t, y):
        """Формула 2: E = Σ [m(t) - y]²"""
        diff = self.model(t, params) - y
        return np.sum(diff ** 2)

    def gradient_E(self, params, t, y):
        """Аналитический градиент E"""
        diff = self.model(t, params) - y
        grad = np.zeros_like(params)

        grad[0] = 2 * np.sum(diff) # dE/dB

        for k in range(self.n_harmonics):
            idx = 1 + k * 3
            A = params[idx]
            w = params[idx + 1]
            d = params[idx + 2]

            arg = w * t + d
            cos_arg = np.cos(arg)
            sin_arg = np.sin(arg)

            grad[idx] = 2 * np.sum(diff * cos_arg)                 # dE/dA
            grad[idx + 1] = 2 * np.sum(diff * A * (-sin_arg) * t)  # dE/dw
            grad[idx + 2] = 2 * np.sum(diff * A * (-sin_arg))      # dE/dd

        return grad

    def epsilon(self, params, t, y):
        """Формула 4: ε = Σ (dE/dp)²"""
        grad = self.gradient_E(params, t, y)
        return np.sum(grad ** 2)

    def gradient_epsilon_numerical(self, params, t, y, delta=1e-8):
        """Численный градиент для ε"""
        grad_eps = np.zeros_like(params)
        base_eps = self.epsilon(params, t, y)

        for i in range(len(params)):
            p_copy = params.copy()
            p_copy[i] += delta
            new_eps = self.epsilon(p_copy, t, y)
            grad_eps[i] = (new_eps - base_eps) / delta

        return grad_eps

    def initialize_params_greedy(self, t, y):
        """
        Умная инициализация (Greedy Search).
        Последовательно находит гармоники, исключая случайность.
        """
        print("  [Init] Запуск инициализации параметров...")
        B_init = np.mean(y)
        residual = y - B_init

        params = [B_init]

        # Сетка частот: ищем от 1 до 50 полных колебаний на отрезке
        w_grid = np.linspace(2 * np.pi, 50 * 2 * np.pi, 200)

        for k in range(self.n_harmonics):
            best_mse = float('inf')
            best_p = (0, 1, 0) # A, w, d

            for w_try in w_grid:
                # Оценка A и d через корреляцию (быстрая прикидка)
                c_term = np.cos(w_try * t)
                s_term = np.sin(w_try * t)

                a_coef = 2 * np.mean(residual * c_term)
                b_coef = 2 * np.mean(residual * s_term)

                A_est = np.sqrt(a_coef**2 + b_coef**2)
                d_est = np.arctan2(-b_coef, a_coef)

                current_wave = A_est * np.cos(w_try * t + d_est)
                mse = np.mean((residual - current_wave)**2)

                if mse < best_mse:
                    best_mse = mse
                    best_p = (A_est, w_try, d_est)

            A_found, w_found, d_found = best_p
            params.extend([A_found, w_found, d_found])
            residual = residual - A_found * np.cos(w_found * t + d_found)
            # print(f"    -> Гармоника {k+1}: w={w_found:.2f}, A={A_found:.2f}")

        return np.array(params)

    def gradient_descent_E(self, t, y, lr=0.002, max_iter=5000, verbose=True):
        """
        Оптимизированный Метод 1 (МНК)
        """
        if verbose:
            print("\n" + "="*60)
            print("МЕТОД 1: Градиентный спуск для минимизации E (МНК)")
            print(f"Learning rate λ (base) = {lr}, max_iter = {max_iter}")
            print("="*60)

        # 1. Инициализация
        self.params = self.initialize_params_greedy(t, y)
        E_init = self.error_E(self.params, t, y)
        if verbose:
            print(f"Начальная ошибка E = {E_init:.2f}")

        # 2. Вектор скоростей обучения (Adaptive LR)
        # Частота w требует шаг в 200 раз меньше, чем амплитуда!
        lr_vector = np.zeros_like(self.params)
        lr_vector[0] = 1.0       # B
        for k in range(self.n_harmonics):
            idx = 1 + k*3
            lr_vector[idx]   = 1.0      # A
            lr_vector[idx+1] = 0.005    # w (scale down!)
            lr_vector[idx+2] = 0.5      # d

        current_lr = lr
        history = []

        for i in range(max_iter):
            grad = self.gradient_E(self.params, t, y)

            # Clipping (защита от взрыва градиента)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1000:
                grad = grad * (1000 / grad_norm)

            # Update с адаптивным вектором
            self.params -= current_lr * (grad * lr_vector)

            loss = self.error_E(self.params, t, y)
            history.append(loss)

            if verbose and (i % 500 == 0 or i == max_iter-1):
                print(f"  Итерация {i:4d}: E = {loss:.4f}, ||grad|| = {grad_norm:.2e}")

            # Annealing (уменьшаем шаг к концу)
            if i > max_iter * 0.8:
                current_lr = lr * 0.1

        self.history['method1'] = history
        return self.params, loss

    def gradient_descent_epsilon(self, t, y, lr=1e-5, max_iter=3000, verbose=True):
        """
        Оптимизированный Метод 2 (МОП)
        """
        if verbose:
            print("\n" + "="*60)
            print("МЕТОД 2: Градиентный спуск для минимизации ε (МОП)")
            print(f"Learning rate λ = {lr}, max_iter = {max_iter}")
            print("="*60)

        # Стартуем с хорошей инициализации
        self.params = self.initialize_params_greedy(t, y)
        eps_init = self.epsilon(self.params, t, y)
        if verbose:
            print(f"Начальное ε = {eps_init:.2e}")

        # Для epsilon w тоже очень чувствителен
        lr_vector = np.ones_like(self.params)
        for k in range(self.n_harmonics):
            lr_vector[1 + k*3 + 1] = 1e-4

        history = []

        for i in range(max_iter):
            grad_eps = self.gradient_epsilon_numerical(self.params, t, y)

            # Нормализация градиента (критично для ε)
            gnorm = np.linalg.norm(grad_eps)
            if gnorm > 1.0:
                grad_eps = grad_eps / gnorm

            self.params -= lr * grad_eps * lr_vector

            eps_val = self.epsilon(self.params, t, y)
            E_val = self.error_E(self.params, t, y)

            # Сохраняем E для совместимости с графиками из 2_1.py
            history.append(E_val)

            if verbose and (i % 500 == 0 or i == max_iter-1):
                print(f"  Итерация {i:4d}: ε = {eps_val:.2e}, E = {E_val:.2f}")

        self.history['method2'] = history
        return self.params, self.error_E(self.params, t, y)


    def predict(self, t):
        return self.model(t, self.params)

    def calculate_metrics(self, y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2, 'MAPE': mape}


def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        col = [c for c in df.columns if 'Close' in c or 'Price' in c][0]
        y = df[col].values
        # Нормализация времени в [0, 1]
        t = np.linspace(0, 1, len(y))
        return t, y, df
    except Exception as e:
        print(f"Ошибка загрузки: {e}. Генерирую синтетические данные.")
        t = np.linspace(0, 1, 200)
        y = 40 + 10*t + 5*np.cos(10*t) + 2*np.cos(30*t) + np.random.normal(0, 0.5, 200)
        return t, y, pd.DataFrame({'Close': y})


def plot_results(t, y, model1, model2, t_test=None, y_test=None,
                 t_forecast=None, y_forecast1=None, y_forecast2=None):
    fig = plt.figure(figsize=(18, 12))

    # График 1: Метод 1 - полные данные + прогноз
    ax1 = plt.subplot(3, 2, 1)
    y_pred1 = model1.predict(t)
    ax1.plot(t, y, 'o', markersize=2, alpha=0.4, label='Исходные данные', color='gray')
    ax1.plot(t, y_pred1, '-', linewidth=2.5, label='Метод 1 (МНК)', color='red')
    if t_forecast is not None:
        ax1.plot(t_forecast, y_forecast1, '--', linewidth=2.5, label='Прогноз МНК', color='darkred')
        ax1.axvline(x=t[-1], color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax1.set_title('Метод 1: МНК - Минимизация E', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График 2: Метод 2 - полные данные + прогноз
    ax2 = plt.subplot(3, 2, 2)
    y_pred2 = model2.predict(t)
    ax2.plot(t, y, 'o', markersize=2, alpha=0.4, label='Исходные данные', color='gray')
    ax2.plot(t, y_pred2, '-', linewidth=2.5, label='Метод 2 (МОП)', color='blue')
    if t_forecast is not None:
        if np.max(np.abs(y_forecast2)) < 1e6:
            ax2.plot(t_forecast, y_forecast2, '--', linewidth=2.5, label='Прогноз МОП', color='darkblue')
        ax2.axvline(x=t[-1], color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax2.set_title('Метод 2: МОП - Минимизация ε', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # График 3: Остатки Метод 1
    ax3 = plt.subplot(3, 2, 3)
    residuals1 = y - y_pred1
    ax3.plot(t, residuals1, 'o-', markersize=2, alpha=0.6, color='red')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax3.set_title(f'Остатки Метод 1 (RMSE = {np.sqrt(np.mean(residuals1**2)):.4f})')
    ax3.grid(True, alpha=0.3)

    # График 4: Остатки Метод 2
    ax4 = plt.subplot(3, 2, 4)
    residuals2 = y - y_pred2
    ax4.plot(t, residuals2, 'o-', markersize=2, alpha=0.6, color='blue')
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax4.set_title(f'Остатки Метод 2 (RMSE = {np.sqrt(np.mean(residuals2**2)):.4f})')
    ax4.grid(True, alpha=0.3)

    # График 5: Сравнение методов
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(t, y, 'o', markersize=2, alpha=0.3, label='Данные', color='lightgray')
    ax5.plot(t, y_pred1, '-', linewidth=2, label='Метод 1 (МНК)', color='red', alpha=0.8)
    ax5.plot(t, y_pred2, '-', linewidth=2, label='Метод 2 (МОП)', color='blue', alpha=0.8)
    ax5.set_title('Сравнение двух методов', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # График 6: Сходимость (история ошибок)
    ax6 = plt.subplot(3, 2, 6)
    if len(model1.history['method1']) > 0:
        ax6.plot(model1.history['method1'], '-', linewidth=2, label='Метод 1 (МНК)', color='red')
    if len(model2.history['method2']) > 0:
        hist2 = np.array(model2.history['method2'])
        if np.max(hist2) < 1e6:
            ax6.plot(hist2, '-', linewidth=2, label='Метод 2 (МОП)', color='blue')
    ax6.set_xlabel('Итерация')
    ax6.set_ylabel('Ошибка E')
    ax6.set_title('Сходимость градиентного спуска (по E)')
    ax6.set_yscale('log')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    print("="*80)
    print("ЛАБОРАТОРНАЯ РАБОТА: АППРОКСИМАЦИЯ ВРЕМЕННОГО РЯДА (Оптимизированная)")
    print("="*80)

    # Загрузка
    filepath = 'Heineken NV Stock Price History.csv'
    t, y, df = load_data(filepath)

    print(f"\nДанные загружены: {len(y)} точек")

    # --- ЧАСТЬ 1: Обучение на всех данных ---
    print("\n" + "="*80)
    print("ЧАСТЬ 1: ОБУЧЕНИЕ НА ВСЕХ ДАННЫХ")
    print("="*80)

    model1 = HarmonicModel(n_harmonics=5)
    model2 = HarmonicModel(n_harmonics=5)

    # Запуск оптимизированных методов
    # Для МНК нужен небольшой LR (0.002), но адаптивный
    params1, E1 = model1.gradient_descent_E(t, y, lr=0.002, max_iter=5000, verbose=True)

    # Для МОП нужен побольше LR (0.05), так как градиент нормирован
    params2, E2 = model2.gradient_descent_epsilon(t, y, lr=0.05, max_iter=3000, verbose=True)

    metrics1 = model1.calculate_metrics(y, model1.predict(t))
    metrics2 = model2.calculate_metrics(y, model2.predict(t))

    print("\nМЕТРИКИ (FULL):")
    print(f"M1 (E): R2={metrics1['R²']:.4f}, RMSE={metrics1['RMSE']:.4f}")
    print(f"M2 (ε): R2={metrics2['R²']:.4f}, RMSE={metrics2['RMSE']:.4f}")


    # --- ЧАСТЬ 2: Ретроспектива (90/10) ---
    print("\n" + "="*80)
    print("ЧАСТЬ 2: РЕТРОСПЕКТИВНАЯ ПРОВЕРКА (90% обучение / 10% тест)")
    print("="*80)

    split_idx = int(len(t) * 0.9)
    t_train = t[:split_idx]
    y_train = y[:split_idx]
    t_test = t[split_idx:]
    y_test = y[split_idx:]

    m1_r = HarmonicModel(n_harmonics=5)
    m2_r = HarmonicModel(n_harmonics=5)

    m1_r.gradient_descent_E(t_train, y_train, lr=0.002, max_iter=5000, verbose=False)
    m2_r.gradient_descent_epsilon(t_train, y_train, lr=0.05, max_iter=3000, verbose=False)

    met1_test = m1_r.calculate_metrics(y_test, m1_r.predict(t_test))
    met2_test = m2_r.calculate_metrics(y_test, m2_r.predict(t_test))

    print(f"RMSE на тесте: M1={met1_test['RMSE']:.4f}, M2={met2_test['RMSE']:.4f}")


    # --- ЧАСТЬ 3: Прогноз (+15%) ---
    print("\n" + "="*80)
    print("ЧАСТЬ 3: ПРОГНОЗ В БУДУЩЕЕ (15%)")
    print("="*80)

    forecast_len = int(len(t) * 0.15)
    t_step = (t[-1] - t[0]) / (len(t) - 1)
    t_forecast = np.linspace(t[-1] + t_step, t[-1] + forecast_len * t_step, forecast_len)

    y_for1 = model1.predict(t_forecast)
    y_for2 = model2.predict(t_forecast)

    # --- ВИЗУАЛИЗАЦИЯ И СОХРАНЕНИЕ ---
    print("\nСоздание графиков...")

    # 1. Основной график (6 панелей)
    fig1 = plot_results(t, y, model1, model2, t_test, y_test, t_forecast, y_for1, y_for2)
    plt.savefig('heineken_harmonic_analysis.png', dpi=300, bbox_inches='tight')
    print("- heineken_harmonic_analysis.png сохранен")

    # 2. Детальная экстраполяция (как в 2_1.py)
    fig2, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Ретро
    ax = axes[0]
    ax.plot(t_train, y_train, 'g.', label='Train')
    ax.plot(t_test, y_test, 'orange', marker='.', linestyle='none', label='Test')
    ax.plot(t, m1_r.predict(t), 'r-', alpha=0.6, label='M1 (Retro)')
    if np.max(np.abs(m2_r.predict(t))) < 1e6:
        ax.plot(t, m2_r.predict(t), 'b--', alpha=0.6, label='M2 (Retro)')
    ax.axvline(t[split_idx], color='purple', linestyle=':')
    ax.set_title('Ретроспективная проверка (90/10)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Прогноз
    ax = axes[1]
    ax.plot(t, y, 'k.', alpha=0.3, label='History')
    ax.plot(t, model1.predict(t), 'r-', alpha=0.4, label='M1 Fit')
    ax.plot(t_forecast, y_for1, 'r--', linewidth=2, label='Forecast M1')
    if np.max(np.abs(y_for2)) < 1e6:
        ax.plot(t_forecast, y_for2, 'b--', linewidth=2, label='Forecast M2')
    ax.set_title('Прогноз в будущее (+15%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('heineken_extrapolation.png', dpi=300, bbox_inches='tight')
    print("- heineken_extrapolation.png сохранен")

    # Сохранение CSV
    df_res = pd.DataFrame({'t': t_forecast, 'M1': y_for1, 'M2': y_for2})
    df_res.to_csv('forecast.csv', index=False)
    print("- forecast.csv сохранен")

    # Сохранение JSON
    results = {
        'metrics_full': {'M1': metrics1, 'M2': metrics2},
        'metrics_test': {'M1': met1_test, 'M2': met2_test}
    }
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("- results.json сохранен")

if __name__ == "__main__":
    main()
