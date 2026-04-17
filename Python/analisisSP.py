import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# Cargar datos
data = np.loadtxt('energia_G3D_larga.csv', delimiter=',', skiprows=1)
t = data[:,0]
E = data[:,1]

# Ajuste exponencial para eliminar tendencia
p = np.polyfit(t, np.log(E), 1)
E_trend = np.exp(p[1]) * np.exp(p[0] * t)
E_res = E - E_trend

# FFT
dt = np.mean(np.diff(t))
N = len(E_res)
freq = fftfreq(N, dt)[:N//2]
fft_vals = fft(E_res)[:N//2]
power = np.abs(fft_vals)**2

# Detectar picos
peaks, _ = find_peaks(power, height=0.1*np.max(power))
peak_freqs = freq[peaks]

print("Picos detectados (Hz):", peak_freqs)

# Graficar
plt.figure(figsize=(10,4))
plt.semilogy(freq, power, 'b-')
plt.scatter(peak_freqs, power[peaks], color='red', label='Picos')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Potencia espectral')
plt.title('Espectro de energía residual - G3D (λ=2/9), T=20 s, malla 64³')
plt.grid(True)
plt.legend()
plt.xlim(0, 1.5)
plt.tight_layout()
plt.savefig('espectro_G3D_larga.png', dpi=150)
plt.show()

# Comparar con diferencias de ceros de Riemann (cargar zeros_100k.txt)
zeros = np.loadtxt('zeros_100k.txt')
g = zeros[:10000]
diffs = []
for i in range(len(g)):
    for j in range(i+1, min(i+200, len(g))):
        d = abs(g[j] - g[i])
        if d < 2.0:
            diffs.append(d)
diffs = np.array(diffs)

# Histograma de diferencias
plt.figure(figsize=(10,4))
plt.hist(diffs, bins=100, alpha=0.5, label='Diferencias entre ceros (|γi-γj|)')
for pf in peak_freqs:
    plt.axvline(pf, color='red', linestyle='--', linewidth=1, label=f'Pico {pf:.3f} Hz' if pf==peak_freqs[0] else '')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Conteo')
plt.title('Comparación de picos espectrales con diferencias de ceros')
plt.legend()
plt.xlim(0, 1.0)
plt.grid(True)
plt.savefig('picos_vs_ceros_larga.png', dpi=150)
plt.show()

# Coincidencias exactas (dentro de la resolución de 0.05 Hz)
res = 0.05
matches = []
for pf in peak_freqs:
    for d in diffs:
        if abs(pf - d) < res:
            matches.append((pf, d))
print("Coincidencias (pico, diferencia):", matches[:20])
