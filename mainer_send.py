import SoapySDR
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Конфигурация SDR
FREQ_CHANNEL_1 = 400.0e6  # Начальная частота
SAMPLE_RATE = 2e6         # Частота дискретизации
GAIN = 30                 # Усиление приёма
JAMMING_THRESHOLD = -15   # Порог глушения (dB)
NUM_AVG = 10              # Количество усреднений
BUFF_SIZE = 1024          # Размер FFT
SCAN_RANGE = 100e6        # Диапазон поиска новой частоты
FREQ_STEP = 1e6           # Шаг сканирования

# Путь для записи данных
output_dir = '/home/dreumn/Desktop/project'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = os.path.join(output_dir, 'frequency_data.txt')


# Подключаем HackRF для приёма
sdr = SoapySDR.Device({'driver': 'hackrf'})
sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, SAMPLE_RATE)
sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, GAIN)

sr = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, "CF32")
sdr.activateStream(sr)


# === ПЕРЕДАЧА ПАКЕТА ===
def transmit_packet(freq):
    """Передаёт короткий синусоидальный сигнал-пакет на заданной частоте"""
    duration = 0.1  # 100 мс
    freq_offset = 100e3  # Частота смещения в сигнале (100 кГц)
    amplitude = 0.5  # Амплитуда (макс. 1.0)

    t = np.arange(0, duration, 1 / SAMPLE_RATE)
    signal = amplitude * np.exp(2j * np.pi * freq_offset * t)  # синус

    tx = SoapySDR.Device({'driver': 'hackrf'})
    tx.setSampleRate(SoapySDR.SOAPY_SDR_TX, 0, SAMPLE_RATE)
    tx.setGain(SoapySDR.SOAPY_SDR_TX, 0, 30)
    tx.setFrequency(SoapySDR.SOAPY_SDR_TX, 0, freq)

    stream = tx.setupStream(SoapySDR.SOAPY_SDR_TX, "CF32")
    tx.activateStream(stream)
    tx.writeStream(stream, [signal.astype(np.complex64)], len(signal))
    tx.deactivateStream(stream)
    tx.closeStream(stream)

    print(f"📡 Передан видимый пакет на {freq/1e6:.1f} МГц (смещение +{freq_offset/1e3:.0f} кГц)")


# === СКАНИРОВАНИЕ ===
def get_power_spectrum(center_freq):
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, center_freq)
    buff = np.zeros(BUFF_SIZE, dtype=np.complex64)
    timeoutUs = int(1e6)
    ret = sdr.readStream(sr, [buff], len(buff), timeoutUs)
    
    if ret.ret > 0:
        spectrum = np.fft.fftshift(np.fft.fft(buff[:ret.ret]))
        power = 10 * np.log10(np.abs(spectrum) ** 2 + 1e-10)
        return power
    else:
        return np.array([])

def scan_spectrum(center_freq):
    power_acc = np.zeros(BUFF_SIZE)
    for _ in range(NUM_AVG):
        power = get_power_spectrum(center_freq)
        if len(power) > 0:
            power_acc += power
    return power_acc / NUM_AVG

def find_best_frequency(current_freq):
    best_freq = current_freq
    min_noise = float('inf')
    for freq in np.arange(current_freq - SCAN_RANGE / 2, current_freq + SCAN_RANGE / 2, FREQ_STEP):
        power_avg = scan_spectrum(freq)
        avg_power = np.mean(power_avg)
        if avg_power < min_noise:
            min_noise = avg_power
            best_freq = freq
    return best_freq


# === МОНИТОРИНГ + ПЕРЕДАЧА ===
def monitor_channel():
    current_freq = FREQ_CHANNEL_1
    freq_list = [current_freq]

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Радиочастотный спектр")
    ax.set_xlabel("Частота (МГц)")
    ax.set_ylabel("Мощность (dB)")
    ax.set_xlim(0, 700)
    ax.set_ylim(0, -100)

    lines = []
    last_tx_time = time.time()

    try:
        with open(output_file, 'w') as f:
            while True:
                power_avg = scan_spectrum(current_freq)
                freqs = np.linspace(current_freq - SAMPLE_RATE/2, current_freq + SAMPLE_RATE/2, BUFF_SIZE) / 1e6

                if len(lines) < len(freq_list):
                    line, = ax.plot([], [], label=f"{current_freq / 1e6:.1f} МГц")
                    lines.append(line)
                line = lines[len(freq_list) - 1]
                line.set_xdata(freqs)
                line.set_ydata(power_avg)

                ax.autoscale_view()
                plt.draw()

                avg_power = np.mean(power_avg)
                print(f"🔍 Средняя мощность на {current_freq / 1e6:.1f} МГц: {avg_power:.2f} dB")
                f.write(f"Частота: {current_freq / 1e6:.1f} МГц, Средняя мощность: {avg_power:.2f} dB\n")

                # === Глушение ===
                if avg_power > JAMMING_THRESHOLD:
                    print(f"⚠️ Обнаружено глушение на {current_freq / 1e6:.1f} МГц!")
                    next_freq = find_best_frequency(current_freq)
                    if next_freq != current_freq:
                        print(f"🔁 Переключение на {next_freq / 1e6:.1f} МГц...")
                        current_freq = next_freq
                        freq_list.append(current_freq)
                        transmit_packet(current_freq)

                # === Периодическая передача пакета ===
                if time.time() - last_tx_time > 10:
                    transmit_packet(current_freq)
                    last_tx_time = time.time()

                plt.pause(0.1)

    except KeyboardInterrupt:
        print("⏹ Мониторинг завершён.")
    finally:
        sdr.deactivateStream(sr)
        sdr.closeStream(sr)
        plt.show()


# === ЗАПУСК ===
monitor_channel()
