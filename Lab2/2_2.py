"""
Лабораторная работа: Аппроксимация акций Heineken
Модель из 5 гармоник - два метода оптимизации + экстраполяция

УЛУЧШЕННАЯ ВЕРСИЯ с адаптивным learning rate
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Настройка для русских шрифтов
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10


class HarmonicModel:
    """
    Модель из 5 гармоник для аппроксимации временного ряда
    """

    def __init__(self, n_harmonics=5):
        self.n_harmonics = n_harmonics
        self.params = None
        self.n_params = 1 + n_harmonics * 3
        self.history = {'method1': [], 'method2': []}

    def model(self, t, params):
        """m(t) = B + Σ A_k * cos(ω_k * t + δ_k)"""
        B = params[0]
        result = B * np.ones_like(t, dtype=float)

        for k in range(self.n_harmonics):
            idx = 1 + k * 3
            A_k = params[idx]
            omega_k = params[idx + 1]
            delta_k = params[idx + 2]
            result += A_k * np.cos(omega_k * t + delta_k)

        return result

    def error_E(self, params, t, y):
        """E = Σ[m(t_i) - y_i]²"""
        model_vals = self.model(t, params)
        residuals = model_vals - y
        return np.sum(residuals ** 2)

    def gradient_E(self, params, t, y):
        """Градиент E по всем параметрам"""
        diff = self.model(t, params) - y
        grad = np.zeros_like(params)

        # ∂E/∂B
        grad[0] = 2 * np.sum(diff)

        # Для каждой гармоники
        for k in range(self.n_harmonics):
            idx = 1 + k * 3
            A_k = params[idx]
            omega_k = params[idx + 1]
            delta_k = params[idx + 2]

            cos_term = np.cos(omega_k * t + delta_k)
            sin_term = np.sin(omega_k * t + delta_k)

            grad[idx] = 2 * np.sum(diff * cos_term)
            grad[idx + 1] = 2 * np.sum(diff * A_k * (-sin_term) * t)
            grad[idx + 2] = 2 * np.sum(diff * A_k * (-sin_term))

        return grad

    def compute_all_partials(self, params, t, y):
        """Вычисляет все частные производные для ε"""
        diff = self.model(t, params) - y
        partials = {}

        partials['dE_dB'] = 2 * np.sum(diff)

        for k in range(self.n_harmonics):
            idx = 1 + k * 3
            A_k = params[idx]
            omega_k = params[idx + 1]
            delta_k = params[idx + 2]

            cos_term = np.cos(omega_k * t + delta_k)
            sin_term = np.sin(omega_k * t + delta_k)

            partials[f'dE_dA{k}'] = 2 * np.sum(diff * cos_term)
            partials[f'dE_dw{k}'] = 2 * np.sum(diff * A_k * (-sin_term) * t)
            partials[f'dE_dd{k}'] = 2 * np.sum(diff * A_k * (-sin_term))

        return partials

    def epsilon(self, params, t, y):
        """ε = Σ[(∂E/∂param)²]"""
        partials = self.compute_all_partials(params, t, y)
        return sum(v ** 2 for v in partials.values())

    def gradient_epsilon_numerical(self, params, t, y, delta=1e-8):
        """Численный градиент ε"""
        grad = np.zeros_like(params)
        eps0 = self.epsilon(params, t, y)

        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += delta
            eps_plus = self.epsilon(params_plus, t, y)
            grad[i] = (eps_plus - eps0) / delta

        return grad

    def initialize_params_better(self, t, y):
        """
        УЛУЧШЕННАЯ инициализация:
        - Лучшая оценка B
        - Меньшие начальные амплитуды
        - Более разнообразные частоты
        """
        B = np.mean(y)
        y_std = np.std(y)
        n_points = len(t)

        params = [B]

        # Более разнообразные периоды
        periods = [
            120.0 / n_points,  # Очень долгая
            60.0 / n_points,  # Долгая
            30.0 / n_points,  # Средняя
            15.0 / n_points,  # Короткая
            10.0 / n_points  # Очень короткая
        ]

        for k in range(self.n_harmonics):
            T_k = periods[k]
            omega = 2 * np.pi / T_k if T_k > 0 else 2 * np.pi * (k + 1)

            # Меньшие начальные амплитуды
            amplitude = y_std / (3 * (k + 1))

            # Случайная начальная фаза для разнообразия
            delta = np.random.uniform(-np.pi, np.pi)

            params.extend([amplitude, omega, delta])

        return np.array(params)

    def gradient_descent_E_adam(self, t, y, lr=0.01, max_iter=10000, tol=1e-8, verbose=True):
        """
        МЕТОД 1: Градиентный спуск с Adam оптимизатором
        """
        if verbose:
            print("\n" + "=" * 70)
            print("МЕТОД 1: Градиентный спуск для E (МНК) с Adam")
            print(f"Learning rate λ = {lr}, max_iter = {max_iter}")
            print("=" * 70)

        params = self.initialize_params_better(t, y)
        E_init = self.error_E(params, t, y)

        if verbose:
            print(f"Начальная ошибка E = {E_init:.4f}")
            print(f"Начальное B = {params[0]:.4f}")

        # Adam параметры
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        m = np.zeros_like(params)  # первый момент
        v = np.zeros_like(params)  # второй момент

        history = []
        best_params = params.copy()
        best_E = E_init

        for iteration in range(max_iter):
            grad = self.gradient_E(params, t, y)
            grad_norm = np.linalg.norm(grad)

            E_current = self.error_E(params, t, y)
            history.append(E_current)

            # Сохраняем лучшие параметры
            if E_current < best_E:
                best_E = E_current
                best_params = params.copy()

            if grad_norm < tol:
                if verbose:
                    print(f"Сходимость на итерации {iteration}, ||grad|| = {grad_norm:.2e}")
                break

            # Adam update
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)

            m_hat = m / (1 - beta1 ** (iteration + 1))
            v_hat = v / (1 - beta2 ** (iteration + 1))

            params = params - lr * m_hat / (np.sqrt(v_hat) + epsilon)

            if verbose and (iteration % 1000 == 0 or iteration == max_iter - 1):
                print(
                    f"  Итерация {iteration:5d}: E = {E_current:.4f}, best E = {best_E:.4f}, ||grad|| = {grad_norm:.2e}")

        self.params = best_params
        E_final = self.error_E(self.params, t, y)
        self.history['method1'] = history

        if verbose:
            print(f"\nЛучшая ошибка E = {E_final:.4f}")
            print(f"Улучшение: {(1 - E_final / E_init) * 100:.2f}%")
            print(f"\nПараметры модели:")
            print(f"  B = {self.params[0]:.6f}")
            for k in range(self.n_harmonics):
                idx = 1 + k * 3
                print(f"  Гармоника {k + 1}: A={self.params[idx]:.6f}, "
                      f"ω={self.params[idx + 1]:.6f}, δ={self.params[idx + 2]:.6f}")

        return self.params, E_final

    def gradient_descent_epsilon_adam(self, t, y, lr=1e-7, max_iter=10000, tol=1e-6, verbose=True):
        """
        МЕТОД 2: Градиентный спуск для ε (МОП) с Adam
        """
        if verbose:
            print("\n" + "=" * 70)
            print("МЕТОД 2: Градиентный спуск для ε (МОП) с Adam")
            print(f"Learning rate λ = {lr}, max_iter = {max_iter}")
            print("=" * 70)

        params = self.initialize_params_better(t, y)
        eps_init = self.epsilon(params, t, y)
        E_init = self.error_E(params, t, y)

        if verbose:
            print(f"Начальное ε = {eps_init:.4e}")
            print(f"Начальная ошибка E = {E_init:.4f}")
            print(f"Начальное B = {params[0]:.4f}")

        # Adam параметры
        beta1 = 0.9
        beta2 = 0.999
        epsilon_adam = 1e-8
        m = np.zeros_like(params)
        v = np.zeros_like(params)

        history = []
        best_params = params.copy()
        best_E = E_init

        for iteration in range(max_iter):
            grad = self.gradient_epsilon_numerical(params, t, y)
            grad_norm = np.linalg.norm(grad)

            # Клиппинг градиента
            max_grad_norm = 1e12
            if grad_norm > max_grad_norm:
                grad = grad * (max_grad_norm / grad_norm)
                grad_norm = max_grad_norm

            E_current = self.error_E(params, t, y)
            history.append(E_current)

            if E_current < best_E:
                best_E = E_current
                best_params = params.copy()

            # Проверка на взрыв
            if np.any(np.abs(params) > 1e6):
                if verbose:
                    print(f"СТОП: параметры взорвались на итерации {iteration}")
                break

            if grad_norm < tol:
                if verbose:
                    print(f"Сходимость на итерации {iteration}, ||grad|| = {grad_norm:.2e}")
                break

            # Adam update
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)

            m_hat = m / (1 - beta1 ** (iteration + 1))
            v_hat = v / (1 - beta2 ** (iteration + 1))

            params = params - lr * m_hat / (np.sqrt(v_hat) + epsilon_adam)

            if verbose and (iteration % 1000 == 0 or iteration == max_iter - 1):
                eps_current = self.epsilon(params, t, y)
                print(f"  Итерация {iteration:5d}: E = {E_current:.4f}, best E = {best_E:.4f}, ε = {eps_current:.2e}")

        self.params = best_params
        eps_final = self.epsilon(self.params, t, y)
        E_final = self.error_E(self.params, t, y)
        self.history['method2'] = history

        if verbose:
            print(f"\nФинальное ε = {eps_final:.4e}")
            print(f"Лучшая ошибка E = {E_final:.4f}")
            if eps_init > 0:
                print(f"Улучшение ε: {(1 - eps_final / eps_init) * 100:.2f}%")
            if E_init > 0:
                print(f"Улучшение E: {(1 - E_final / E_init) * 100:.2f}%")
            print(f"\nПараметры модели:")
            print(f"  B = {self.params[0]:.6f}")
            for k in range(self.n_harmonics):
                idx = 1 + k * 3
                print(f"  Гармоника {k + 1}: A={self.params[idx]:.6f}, "
                      f"ω={self.params[idx + 1]:.6f}, δ={self.params[idx + 2]:.6f}")

        return self.params, E_final

    def predict(self, t):
        """Предсказание значений"""
        if self.params is None:
            raise ValueError("Модель не обучена!")
        return self.model(t, self.params)

    def calculate_metrics(self, y_true, y_pred):
        """Метрики качества"""
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape
        }


def load_data(filepath):
    """Загрузка данных из CSV"""
    df = pd.read_csv(filepath)

    price_col = None
    for col in ['Close', 'Adj Close', 'close', 'Price', 'price']:
        if col in df.columns:
            price_col = col
            break

    if price_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        price_col = numeric_cols[-1]

    y = df[price_col].values
    t = np.arange(len(y), dtype=float)
    t_norm = (t - t[0]) / (t[-1] - t[0]) if len(t) > 1 else t

    return t_norm, y, df


def plot_results(t, y, model1, model2, t_test=None, y_test=None,
                 t_forecast=None, y_forecast1=None, y_forecast2=None):
    """Визуализация результатов"""

    fig = plt.figure(figsize=(18, 12))

    # График 1: Метод 1
    ax1 = plt.subplot(3, 2, 1)
    y_pred1 = model1.predict(t)
    ax1.plot(t, y, 'o', markersize=2, alpha=0.4, label='Исходные данные', color='gray')
    ax1.plot(t, y_pred1, '-', linewidth=2.5, label='Метод 1 (МНК)', color='red')

    if t_forecast is not None and y_forecast1 is not None:
        ax1.plot(t_forecast, y_forecast1, '--', linewidth=2.5,
                 label='Прогноз МНК', color='darkred')
        ax1.axvline(x=t[-1], color='orange', linestyle=':', linewidth=2, alpha=0.7)

    ax1.set_xlabel('Нормализованное время', fontsize=11)
    ax1.set_ylabel('Цена акций', fontsize=11)
    ax1.set_title('Метод 1: МНК с Adam', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # График 2: Метод 2
    ax2 = plt.subplot(3, 2, 2)
    y_pred2 = model2.predict(t)
    ax2.plot(t, y, 'o', markersize=2, alpha=0.4, label='Исходные данные', color='gray')
    ax2.plot(t, y_pred2, '-', linewidth=2.5, label='Метод 2 (МОП)', color='blue')

    if t_forecast is not None and y_forecast2 is not None:
        if np.max(np.abs(y_forecast2)) < 1e6:
            ax2.plot(t_forecast, y_forecast2, '--', linewidth=2.5,
                     label='Прогноз МОП', color='darkblue')
        ax2.axvline(x=t[-1], color='orange', linestyle=':', linewidth=2, alpha=0.7)

    ax2.set_xlabel('Нормализованное время', fontsize=11)
    ax2.set_ylabel('Цена акций', fontsize=11)
    ax2.set_title('Метод 2: МОП с Adam', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # График 3: Остатки Метод 1
    ax3 = plt.subplot(3, 2, 3)
    residuals1 = y - y_pred1
    ax3.plot(t, residuals1, 'o-', markersize=2, alpha=0.6, color='red')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax3.set_xlabel('Нормализованное время', fontsize=11)
    ax3.set_ylabel('Остатки', fontsize=11)
    ax3.set_title(f'Остатки Метод 1 (RMSE = {np.sqrt(np.mean(residuals1 ** 2)):.4f})',
                  fontsize=11)
    ax3.grid(True, alpha=0.3)

    # График 4: Остатки Метод 2
    ax4 = plt.subplot(3, 2, 4)
    residuals2 = y - y_pred2
    ax4.plot(t, residuals2, 'o-', markersize=2, alpha=0.6, color='blue')
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax4.set_xlabel('Нормализованное время', fontsize=11)
    ax4.set_ylabel('Остатки', fontsize=11)
    ax4.set_title(f'Остатки Метод 2 (RMSE = {np.sqrt(np.mean(residuals2 ** 2)):.4f})',
                  fontsize=11)
    ax4.grid(True, alpha=0.3)

    # График 5: Сравнение
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(t, y, 'o', markersize=2, alpha=0.3, label='Данные', color='lightgray')
    ax5.plot(t, y_pred1, '-', linewidth=2, label='Метод 1 (МНК)', color='red', alpha=0.8)
    ax5.plot(t, y_pred2, '-', linewidth=2, label='Метод 2 (МОП)', color='blue', alpha=0.8)
    ax5.set_xlabel('Нормализованное время', fontsize=11)
    ax5.set_ylabel('Цена акций', fontsize=11)
    ax5.set_title('Сравнение двух методов', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # График 6: Сходимость
    ax6 = plt.subplot(3, 2, 6)
    if len(model1.history['method1']) > 0:
        ax6.plot(model1.history['method1'], '-', linewidth=2, label='Метод 1 (МНК)',
                 color='red', alpha=0.8)
    if len(model2.history['method2']) > 0:
        hist2 = np.array(model2.history['method2'])
        if np.max(hist2) < 1e6:
            ax6.plot(hist2, '-', linewidth=2, label='Метод 2 (МОП)',
                     color='blue', alpha=0.8)
    ax6.set_xlabel('Итерация', fontsize=11)
    ax6.set_ylabel('Ошибка E', fontsize=11)
    ax6.set_title('Сходимость (логшкала)', fontsize=11)
    ax6.set_yscale('log')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Основная функция"""

    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА: АППРОКСИМАЦИЯ ВРЕМЕННОГО РЯДА АКЦИЙ HEINEKEN")
    print("Модель из 5 гармоник - УЛУЧШЕННАЯ ВЕРСИЯ с Adam")
    print("=" * 80)

    # Загрузка данных
    filepath = 'Heineken NV Stock Price History.csv'
    t, y, df = load_data(filepath)

    print(f"\nДанные загружены: {len(y)} точек")
    print(f"Диапазон цен: [{np.min(y):.2f}, {np.max(y):.2f}]")
    print(f"Среднее: {np.mean(y):.2f}, СКО: {np.std(y):.2f}")

    # ========================================================================
    # ЧАСТЬ 1: Обучение на всех данных
    # ========================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 1: ОБУЧЕНИЕ НА ВСЕХ ДАННЫХ")
    print("=" * 80)

    model1 = HarmonicModel(n_harmonics=5)
    model2 = HarmonicModel(n_harmonics=5)

    # Метод 1 с Adam
    params1, E1 = model1.gradient_descent_E_adam(t, y, lr=0.01, max_iter=10000, verbose=True)

    # Метод 2 с Adam
    params2, E2 = model2.gradient_descent_epsilon_adam(t, y, lr=1e-7, max_iter=10000, verbose=True)

    # Метрики
    y_pred1 = model1.predict(t)
    y_pred2 = model2.predict(t)

    metrics1 = model1.calculate_metrics(y, y_pred1)
    metrics2 = model2.calculate_metrics(y, y_pred2)

    print("\n" + "-" * 70)
    print("МЕТРИКИ НА ВСЕХ ДАННЫХ:")
    print("-" * 70)
    print("Метод 1 (МНК + Adam):")
    for k, v in metrics1.items():
        print(f"  {k:6s} = {v:.6f}")

    print("\nМетод 2 (МОП + Adam):")
    for k, v in metrics2.items():
        print(f"  {k:6s} = {v:.6f}")

    # ========================================================================
    # ЧАСТЬ 2: Ретроспективная проверка
    # ========================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 2: РЕТРОСПЕКТИВНАЯ ПРОВЕРКА (90/10)")
    print("=" * 80)

    split_idx = int(len(t) * 0.9)
    t_train, y_train = t[:split_idx], y[:split_idx]
    t_test, y_test = t[split_idx:], y[split_idx:]

    print(f"Обучение: {len(t_train)} точек, Тест: {len(t_test)} точек")

    model1_retro = HarmonicModel(n_harmonics=5)
    model2_retro = HarmonicModel(n_harmonics=5)

    model1_retro.gradient_descent_E_adam(t_train, y_train, lr=0.01, max_iter=10000, verbose=False)
    model2_retro.gradient_descent_epsilon_adam(t_train, y_train, lr=1e-7, max_iter=10000, verbose=False)

    y_pred1_test = model1_retro.predict(t_test)
    y_pred2_test = model2_retro.predict(t_test)

    metrics1_test = model1_retro.calculate_metrics(y_test, y_pred1_test)
    metrics2_test = model2_retro.calculate_metrics(y_test, y_pred2_test)

    print("\nМЕТРИКИ НА ТЕСТЕ:")
    print("Метод 1 (МНК + Adam):")
    for k, v in metrics1_test.items():
        print(f"  {k:6s} = {v:.6f}")

    print("\nМетод 2 (МОП + Adam):")
    for k, v in metrics2_test.items():
        print(f"  {k:6s} = {v:.6f}")

    # ========================================================================
    # ЧАСТЬ 3: Прогноз
    # ========================================================================
    print("\n" + "=" * 80)
    print("ЧАСТЬ 3: ПРОГНОЗ В БУДУЩЕЕ (15%)")
    print("=" * 80)

    forecast_len = int(len(t) * 0.15)
    t_step = (t[-1] - t[0]) / (len(t) - 1)
    t_forecast = np.linspace(t[-1] + t_step, t[-1] + forecast_len * t_step, forecast_len)

    y_forecast1 = model1.predict(t_forecast)
    y_forecast2 = model2.predict(t_forecast)

    print(f"Прогноз на {forecast_len} точек")
    print(f"Прогноз МНК: min={np.min(y_forecast1):.2f}, max={np.max(y_forecast1):.2f}")

    if np.max(np.abs(y_forecast2)) < 1e6:
        print(f"Прогноз МОП: min={np.min(y_forecast2):.2f}, max={np.max(y_forecast2):.2f}")

    # ========================================================================
    # ВИЗУАЛИЗАЦИЯ
    # ========================================================================
    print("\n" + "=" * 80)
    print("СОЗДАНИЕ ГРАФИКОВ...")
    print("=" * 80)

    fig = plot_results(t, y, model1, model2,
                       t_test=t_test, y_test=y_test,
                       t_forecast=t_forecast,
                       y_forecast1=y_forecast1,
                       y_forecast2=y_forecast2)

    plt.savefig('outputs/heineken_harmonic_analysis.png', dpi=300, bbox_inches='tight')
    print("- График сохранен: heineken_harmonic_analysis.png")

    # Сохранение результатов
    results = {
        'method1': {'params': params1.tolist(), 'E': float(E1), 'metrics': metrics1},
        'method2': {'params': params2.tolist(), 'E': float(E2), 'metrics': metrics2},
        'method1_test': {'metrics': metrics1_test},
        'method2_test': {'metrics': metrics2_test},
    }

    import json
    with open('outputs/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("- Результаты сохранены: results.json")

    forecast_df = pd.DataFrame({
        't_normalized': t_forecast,
        'forecast_mnk': y_forecast1,
        'forecast_mop': y_forecast2
    })
    forecast_df.to_csv('outputs/forecast.csv', index=False)
    print("- Прогнозы сохранены: forecast.csv")

    print("\n" + "=" * 80)
    print("ИТОГОВЫЕ ВЫВОДЫ")
    print("=" * 80)
    print(f"\nНа полных данных:")
    print(f"  Метод 1 (МНК + Adam): R² = {metrics1['R²']:.4f}, RMSE = {metrics1['RMSE']:.4f}")
    print(f"  Метод 2 (МОП + Adam): R² = {metrics2['R²']:.4f}, RMSE = {metrics2['RMSE']:.4f}")

    print(f"\nНа тесте (10%):")
    print(f"  Метод 1: RMSE = {metrics1_test['RMSE']:.4f}, MAPE = {metrics1_test['MAPE']:.2f}%")
    print(f"  Метод 2: RMSE = {metrics2_test['RMSE']:.4f}, MAPE = {metrics2_test['MAPE']:.2f}%")

    if metrics1['RMSE'] < metrics2['RMSE']:
        print("\n✓ На полных данных лучше: Метод 1 (МНК)")
    else:
        print("\n✓ На полных данных лучше: Метод 2 (МОП)")

    if metrics1_test['RMSE'] < metrics2_test['RMSE']:
        print("✓ На тесте лучше: Метод 1 (МНК)")
    else:
        print("✓ На тесте лучше: Метод 2 (МОП)")


if __name__ == '__main__':
    main()
