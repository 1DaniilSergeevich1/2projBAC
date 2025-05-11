import SoapySDR
import numpy as np
import SoapySDR as soapy
from SoapySDR import *  # Импортируем все из SoapySDR
import sys  # Для аргументов

try:
    arg = float(str(sys.argv[1])+"e6")
except Exception:
    arg = 0

# Параметры SDR для глушения канала
SAMPLE_RATE = 2e6  # в МГц
AMPLITUDE = 0.9  # Увеличиваем амплитуду сигнала (-1 до 1)
TARGET_FREQUENCY = arg if arg != 0 else 85.5e6   # e6 => в МГц
BUFFER_SIZE = 262144  # Увеличенный размер буфера для стабильной передачи

# Функция генерации белого шума
def generate_noise(samples=BUFFER_SIZE):
    return (np.random.uniform(-AMPLITUDE, AMPLITUDE, samples) +
            1j * np.random.uniform(-AMPLITUDE, AMPLITUDE, samples)).astype(np.complex64)

# Инициализация HackRF через SoapySDR
try:
    sdr = soapy.Device({'driver': 'hackrf'})
    print("HackRF успешно найден!")
except Exception as e:
    print(f"Ошибка: {e}")
    exit(1)

# Настройка параметров
sdr.setSampleRate(SOAPY_SDR_TX, 0, SAMPLE_RATE)
sdr.setFrequency(SOAPY_SDR_TX, 0, TARGET_FREQUENCY)
sdr.setGain(SOAPY_SDR_TX, 0, 47)  # Усиление
sdr.setBandwidth(SOAPY_SDR_TX, 0, SAMPLE_RATE)  # Сужаем полосу

# Создаем поток передачи
tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])
sdr.activateStream(tx_stream)

# Запись текущей частоты в файл
output_file = "amming_log.txt"

print(f"⚡️ Глушение запущено на {TARGET_FREQUENCY / 1e6} МГц с усилением сигнала")

try:
    noise = generate_noise()
    with open(output_file, 'w') as f:
        while True:
            f.write(f"Глушение на частоте {TARGET_FREQUENCY / 1e6} МГц\n")
            status = sdr.writeStream(tx_stream, [noise], len(noise))
            if status.ret < 0:
                print(f"Ошибка передачи: {status.ret}")
                break

except KeyboardInterrupt:
    print("Глушение остановлено.")
    sdr.deactivateStream(tx_stream)
    sdr.closeStream(tx_stream)
