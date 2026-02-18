import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, c, k, pi
from scipy.signal import butter, filtfilt
import matplotlib.gridspec as gridspec


class AdvancedTurbinePyrometer:
    def __init__(self, sample_rate=2e6, duration=0.02):
        self.sample_rate = sample_rate
        self.duration = duration
        self.time = np.arange(0, duration, 1 / sample_rate)

    def planck_radiation(self, wavelength, temperature):
        """Излучение черного тела по закону Планка"""
        wl = wavelength * 1e-6  # перевод в метры
        return (2 * pi * h * c ** 2 / wl ** 5) * \
            (1 / (np.exp(h * c / (wl * k * temperature)) - 1))

    def blade_temperature_profile(self, position, blade_length,
                                  root_temp=800, tip_temp=1600, profile_type='linear'):
        """
        Температурный профиль вдоль лопатки

        Parameters:
        position - позиция вдля лопатки (0 - корень, 1 - кончик)
        blade_length - длина лопатки
        root_temp - температура у корня
        tip_temp - температура на кончике
        profile_type - тип профиля: 'linear', 'parabolic', 'exponential'
        """
        if profile_type == 'linear':
            return root_temp + (tip_temp - root_temp) * position
        elif profile_type == 'parabolic':
            return root_temp + (tip_temp - root_temp) * (position ** 1.5)
        elif profile_type == 'exponential':
            return root_temp + (tip_temp - root_temp) * (1 - np.exp(-3 * position))
        else:
            return root_temp + (tip_temp - root_temp) * position

    def generate_continuous_signal(self, rpm=12000, num_blades=60,
                                   blade_length=0.1, probe_position=0.7,
                                   base_root_temp=800, base_tip_temp=1600,
                                   temp_variation=50, vibration_freq=850):
        """
        Генерация непрерывного сигнала с учетом температурных градиентов

        Parameters:
        probe_position - позиция зонда вдоль лопатки (0-1)
        blade_length - длина лопатки в метрах
        """
        # Параметры вращения
        rotation_freq = rpm / 60  # Гц
        blade_passing_freq = rotation_freq * num_blades  # Гц
        blade_period = 1 / blade_passing_freq

        print(f"Частота вращения: {rotation_freq:.1f} Гц")
        print(f"Частота следования лопаток: {blade_passing_freq:.1f} Гц")

        # Угловая позиция зонда (предполагаем, что зонд смотрит тангенциально)
        theta_probe = 2 * pi * probe_position / num_blades

        # Создание непрерывного углового сигнала
        angular_position = 2 * pi * rotation_freq * self.time

        # Определение, какая лопатка в поле зрения в каждый момент времени
        blade_angles = np.linspace(0, 2 * pi, num_blades, endpoint=False)

        # Температурные профили для каждой лопатки (разные из-за неравномерного нагрева)
        blade_profiles = []
        for i in range(num_blades):
            # Случайные вариации температурного профиля для каждой лопатки
            root_temp_var = base_root_temp + np.random.uniform(-temp_variation, temp_variation)
            tip_temp_var = base_tip_temp + np.random.uniform(-temp_variation, temp_variation)
            blade_profiles.append((root_temp_var, tip_temp_var))

        # Расчет температуры в поле зрения в каждый момент времени
        temperature_signal = np.zeros_like(self.time)

        for i, angle in enumerate(angular_position):
            # Находим ближайшую лопатку
            relative_angles = np.mod(angle - blade_angles, 2 * pi)
            closest_blade_idx = np.argmin(np.abs(relative_angles))

            # Позиция вдоль лопатки (зависит от угла)
            # Чем больше угол между зондом и лопаткой, тем ближе к краю лопатки мы видим
            blade_relative_angle = relative_angles[closest_blade_idx]
            effective_position = probe_position + 0.3 * np.sin(blade_relative_angle * 2)
            effective_position = np.clip(effective_position, 0, 1)

            # Температура в этой точке лопатки
            root_temp, tip_temp = blade_profiles[closest_blade_idx]
            temp = self.blade_temperature_profile(effective_position, blade_length,
                                                  root_temp, tip_temp, 'parabolic')

            temperature_signal[i] = temp

        # Добавление динамических эффектов
        # 1. Вибрация
        vibration = 0.02 * np.sin(2 * pi * vibration_freq * self.time)
        # 2. Медленные температурные дрейфы
        drift = 10 * np.sin(2 * pi * 5 * self.time)  # 5 Гц дрейф
        # 3. Быстрые пульсации от горения
        combustion_pulsation = 5 * np.sin(2 * pi * 120 * self.time + np.random.uniform(0, 2 * pi))

        temperature_signal *= (1 + vibration)
        temperature_signal += drift + combustion_pulsation

        # Добавление шума
        noise = np.random.normal(0, 8, len(self.time))  # Шум измерения
        temperature_signal += noise

        return temperature_signal, blade_passing_freq, blade_profiles

    def simulate_optical_response(self, temperature_signal, wavelength=2.0):
        """Преобразование температуры в оптический сигнал"""
        optical_power = np.array([self.planck_radiation(wavelength, T)
                                  for T in temperature_signal])

        # Нормализация и добавление нелинейности датчика
        optical_power = optical_power / np.max(optical_power)

        return optical_power

    def plot_detailed_analysis(self, time, temp_signal, optical_signal, blade_profiles):
        """Детальный анализ сигнала"""
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig)

        # График 1: Температурный сигнал во времени
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(time * 1000, temp_signal, 'b-', linewidth=1, alpha=0.8)
        ax1.set_ylabel('Температура (K)')
        ax1.set_title('Непрерывный температурный сигнал с лопаток ТВД')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, time[-1] * 1000)

        """# График 2: Оптический сигнал
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(time * 1000, optical_signal, 'r-', linewidth=1, alpha=0.8)
        ax2.set_ylabel('Оптическая мощность (отн. ед.)')
        ax2.set_xlabel('Время (мс)')
        ax2.set_title('Оптический сигнал на входе пирометра')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, time[-1] * 1000)"""

        # График 3: Температурные профили лопаток
        ax3 = fig.add_subplot(gs[1, 0])
        positions = np.linspace(0, 1, 100)
        for i, (root_temp, tip_temp) in enumerate(blade_profiles[:8]):  # Первые 8 лопаток
            profile = [self.blade_temperature_profile(p, 0.1, root_temp, tip_temp, 'parabolic')
                       for p in positions]
            ax3.plot(positions, profile, label=f'Лопатка {i + 1}')
        ax3.set_xlabel('Позиция вдоль лопатки')
        ax3.set_ylabel('Температура (K)')
        ax3.set_title('Температурные профили лопаток')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # График 4: Гистограмма температур
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(temp_signal, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax4.set_xlabel('Температура (K)')
        ax4.set_ylabel('Частота')
        ax4.set_title('Распределение температур в сигнале')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Статистика
        print("\n=== ДЕТАЛЬНАЯ СТАТИСТИКА ===")
        print(f"Средняя температура: {np.mean(temp_signal):.1f} K")
        print(f"Стандартное отклонение: {np.std(temp_signal):.1f} K")
        print(f"Максимальная температура: {np.max(temp_signal):.1f} K")
        print(f"Минимальная температура: {np.min(temp_signal):.1f} K")
        print(f"Размах температур: {np.max(temp_signal) - np.min(temp_signal):.1f} K")

        # Анализ градиентов
        grad_profiles = []
        for root_temp, tip_temp in blade_profiles:
            root_temp_calc = self.blade_temperature_profile(0, 0.1, root_temp, tip_temp, 'parabolic')
            tip_temp_calc = self.blade_temperature_profile(1, 0.1, root_temp, tip_temp, 'parabolic')
            grad_profiles.append(tip_temp_calc - root_temp_calc)

        print(f"Средний градиент по лопаткам: {np.mean(grad_profiles):.1f} K")
        print(f"Максимальный градиент: {np.max(grad_profiles):.1f} K")


# Запуск улучшенной модели
if __name__ == "__main__":
    # Создание продвинутой модели
    advanced_pyrometer = AdvancedTurbinePyrometer(sample_rate=2e6, duration=0.01)

    # Генерация непрерывного сигнала с градиентами
    temp_signal, blade_freq, blade_profiles = advanced_pyrometer.generate_continuous_signal(
        rpm=15000,
        num_blades=72,
        blade_length=0.12,
        probe_position=0.6,  # Зонд смотрит на 60% длины лопатки
        base_root_temp=850,
        base_tip_temp=1650,
        temp_variation=40,
        vibration_freq=920
    )

    # Преобразование в оптический сигнал
    optical_signal = advanced_pyrometer.simulate_optical_response(temp_signal, wavelength=1.6)

    # Детальный анализ
    advanced_pyrometer.plot_detailed_analysis(
        advanced_pyrometer.time,
        temp_signal,
        optical_signal,
        blade_profiles
    )

    # Дополнительно: спектральный анализ
    plt.figure(figsize=(12, 4))

    # БПФ температурного сигнала
    n = len(temp_signal)
    fft_temp = np.fft.fft(temp_signal - np.mean(temp_signal))
    freqs = np.fft.fftfreq(n, 1 / advanced_pyrometer.sample_rate)

    positive_idx = freqs > 0
    freqs_pos = freqs[positive_idx]
    fft_magnitude = np.abs(fft_temp[positive_idx])

    plt.plot(freqs_pos / 1000, 20 * np.log10(fft_magnitude / np.max(fft_magnitude)))
    plt.axvline(x=blade_freq / 1000, color='r', linestyle='--',
                label=f'Частота лопаток: {blade_freq / 1000:.1f} кГц')
    plt.axvline(x=5, color='g', linestyle='--', label='Низкочастотный дрейф (5 Гц)')
    plt.axvline(x=120, color='orange', linestyle='--', label='Пульсации горения (120 Гц)')

    plt.xlabel('Частота (кГц)')
    plt.ylabel('Амплитуда (дБ)')
    plt.title('Спектр температурного сигнала')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 200)
    plt.show()
