import matplotlib.pyplot as plt
import numpy as np

# Алгоритм прямого одномерного преобразования Фурье
def DFTforward(N, y):
  F = np.zeros(N, complex)
  for i in range(N):
      for j in range(N):
          F[i] = F[i] + y[j] * np.exp(-1j * 2 * np.pi * (j * i / N))
  return F

# Алгоритм обратного одномерного преобразования Фурье
def DFTbackward(N, F):
  y = np.zeros(N, complex)
  for j in range(N):
      for i in range(N):
          y[j] = y[j] + F[i] * np.exp(1j * 2 * np.pi * (j * i / N))
  return y

# Идеальный низкочастотный фильтр
def AmplitudeFilter(D0, F, N, fs):
  Fnormalize = np.zeros(N)
  Fm = 2 * abs(F) / N  # перевод в амплитуду сигнала
  Fnormalize = Fm / max(Fm)
  k = np.arange(0, N, 1)  # отсчеты
  m = 2 * np.pi * (k / N) * fs  # перевод из отсчетов в спектр частот сигнала
  plt.plot(m, Fnormalize)
  plt.show()
  for i in range(0, N - 1, 1):
    if Fnormalize[i] < D0:
      F[i] = 0
  Fm = 2 * abs(F) / N
  plt.plot(m, Fm)
  plt.show()
  return F

if __name__ == "__main__":
  # Задание параметров сигнала
  fs = 200 #задание частоты дескретизации сигнала
  t = np.arange(0, 3.14*2, 1/fs) #задание интеравала(периода) дискретизации исходного сигнала
  y = 1*np.cos(0.5 * 6 * t) + 6 * np.sin(6 + (2 * 6 + 1) * t) #задание исходного сигнала
  y = y + np.random.randn(len(y))

  # вывод графиков исходных сигналов
  plt.plot(t, y)
  plt.show()

  N = len(t) #количество точек(отсчетов) дескретезации для исходного сигнала


  # прямое ДПФ
  F = DFTforward(N,y)
  Fm = 2*abs(F) / N #перевод в амплитуду сигнала 1
  k = np.arange(0, N, 1) #отсчеты
  m = 2*np.pi*(k / N) * fs #перевод из отсчетов в спектр частот сигнала 1

  Fnoise = AmplitudeFilter(0.1, F, N, fs)

  # обратное ДПФ
  y = (DFTbackward(N, Fnoise)) / N #перевод в амплитуду сигнала 1
  t = np.arange(0, N, 1) #отсчеты

  # вывод графиков обратно преобразованных сигналов
  plt.plot(t, y)
  plt.show()
