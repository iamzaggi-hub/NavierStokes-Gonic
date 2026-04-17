import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# Cargar energía
data = np.loadtxt('ns3d_G3D_128.csv', delimiter=',', skiprows=1)
t = data[:,0]
E = data[:,1]

# Ajuste exponencial
p = np.polyfit(t, np.log(E), 1)
E_trend = np.exp(p[1]) * np.exp(p[0] * t)
E_res = E - E_trend

# FFT
dt = np.mean(np.diff(t))
N = len(E_res)
freq = fftfreq(N, dt)
fft_vals = fft(E_res)
power = np.abs(fft_vals[:N//2])**2
freq_pos = freq[:N//2]

# Encontrar picos en el espectro (hasta frecuencia máxima relevante, p.ej. 5)
peaks_idx, _ = find_peaks(power, height=0.05*np.max(power))
peak_freqs = freq_pos[peaks_idx]
print("Picos detectados (Hz):", peak_freqs)

# Cargar ceros de Riemann
zeros = np.loadtxt('zeros_100k.txt')   # primera columna, asumiendo valores γ
# Tomar primeros 1000
N_zeros = 1000
g = zeros[:N_zeros]

# Calcular diferencias entre ceros (todas las combinaciones hasta cierto límite)
diffs = []
for i in range(N_zeros):
    for j in range(i+1, min(i+50, N_zeros)):   # solo diferencias entre ceros cercanos
        d = abs(g[j] - g[i])
        if d < 10.0:   # limitamos a frecuencias menores de 10 Hz
            diffs.append(d)

# Agrupar en histograma suave para ver concentraciones
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.semilogy(freq_pos, power, label='Espectro energía residual')
for f in peak_freqs:
    plt.axvline(f, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Frecuencia (1/s)')
plt.ylabel('Potencia')
plt.title('Espectro de energía (G3D)')
plt.xlim(0, 5)
plt.grid(True)

plt.subplot(1,2,2)
plt.hist(diffs, bins=50, alpha=0.7, label='Diferencias entre ceros')
plt.xlabel('Frecuencia (1/s)')
plt.ylabel('Conteo')
plt.title('Histograma de diferencias γₙ - γₘ')
for f in peak_freqs:
    plt.axvline(f, color='r', linestyle='--', alpha=0.5)
plt.grid(True)
plt.tight_layout()
plt.show()

# Buscar coincidencias
tol = 0.05   # tolerancia en Hz
matches = []
for pf in peak_freqs:
    for d in diffs:
        if abs(pf - d) < tol:
            matches.append((pf, d))
print("Coincidencias (pico, diferencia):", matches[:20])
