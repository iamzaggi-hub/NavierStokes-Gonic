#!/usr/bin/env python3
# analisis_espectral_100s.py
# Lee energia_enstrofia_G3D_100s.csv, calcula espectros y busca coincidencias con ceros de Riemann

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import pearsonr

# ------------------------------------------------------------
# 1. Cargar datos
# ------------------------------------------------------------
data = np.loadtxt('energia_enstrofia_G3D_100s.csv', delimiter=',', skiprows=1)
t = data[:,0]
E = data[:,1]
Z = data[:,2]   # enstrofía

print(f"Puntos temporales: {len(t)}")
print(f"Tiempo total: {t[-1]:.2f} s, dt medio: {np.mean(np.diff(t)):.4f} s")

# ------------------------------------------------------------
# 2. Remover tendencia exponencial (energía) y lineal (enstrofía)
# ------------------------------------------------------------
# Energía: ajuste exponencial E(t) = a * exp(b*t)
# Tomamos log para linealizar
logE = np.log(E + 1e-12)  # evitar log(0)
pE = np.polyfit(t, logE, 1)
E_trend = np.exp(pE[1]) * np.exp(pE[0] * t)
E_res = E - E_trend

# Enstrofía: ajuste lineal en escala log (también exponencial)
logZ = np.log(Z + 1e-12)
pZ = np.polyfit(t, logZ, 1)
Z_trend = np.exp(pZ[1]) * np.exp(pZ[0] * t)
Z_res = Z - Z_trend

print(f"Tendencia energía: exp({pE[0]:.4f} * t) * {np.exp(pE[1]):.4f}")
print(f"Tendencia enstrofía: exp({pZ[0]:.4f} * t) * {np.exp(pZ[1]):.4f}")

# ------------------------------------------------------------
# 3. FFT y espectro de potencia
# ------------------------------------------------------------
dt = np.mean(np.diff(t))
N = len(t)
freq = fftfreq(N, dt)[:N//2]

def espectro(serie):
    fft_vals = fft(serie)[:N//2]
    power = np.abs(fft_vals)**2
    return freq, power

freq_E, power_E = espectro(E_res)
freq_Z, power_Z = espectro(Z_res)

# Detectar picos (umbral relativo)
peaks_E, _ = find_peaks(power_E, height=0.05*np.max(power_E))
peaks_Z, _ = find_peaks(power_Z, height=0.05*np.max(power_Z))

freq_peaks_E = freq_E[peaks_E]
freq_peaks_Z = freq_Z[peaks_Z]

print("\nPicos en espectro de energía (Hz):", freq_peaks_E)
print("Picos en espectro de enstrofía (Hz):", freq_peaks_Z)

# ------------------------------------------------------------
# 4. Cargar ceros de Riemann y calcular diferencias
# ------------------------------------------------------------
try:
    zeros = np.loadtxt('zeros_100k.txt')
    # asumimos una columna con valores gamma
    g = zeros.flatten()
    print(f"\nCargados {len(g)} ceros de Riemann")

    # Calcular diferencias entre ceros (solo las pequeñas, hasta 2 Hz)
    diffs = []
    # Para no hacer O(N^2), muestreamos los primeros 5000 ceros y buscamos diferencias < 2 Hz
    # Esto es suficiente para tener un histograma representativo
    g_sample = g[:5000]
    for i in range(len(g_sample)):
        for j in range(i+1, min(i+200, len(g_sample))):
            d = abs(g_sample[j] - g_sample[i])
            if d < 2.0:
                diffs.append(d)
    diffs = np.array(diffs)
    print(f"Diferencias calculadas: {len(diffs)} (valores < 2 Hz)")
except Exception as e:
    print(f"Error al cargar ceros: {e}")
    diffs = np.array([])

# ------------------------------------------------------------
# 5. Comparación de picos con diferencias de ceros
# ------------------------------------------------------------
if len(diffs) > 0:
    # Histograma de diferencias
    hist, bin_edges = np.histogram(diffs, bins=100, range=(0, 2.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Buscar coincidencias con picos
    tol = 0.01  # tolerancia 0.01 Hz (resolución 0.01 Hz)
    coincidencias_E = []
    for fp in freq_peaks_E:
        # buscar diferencias cercanas
        cerca = diffs[np.abs(diffs - fp) < tol]
        if len(cerca) > 0:
            coincidencias_E.append((fp, cerca))
    coincidencias_Z = []
    for fp in freq_peaks_Z:
        cerca = diffs[np.abs(diffs - fp) < tol]
        if len(cerca) > 0:
            coincidencias_Z.append((fp, cerca))

    print("\nCoincidencias (pico energía vs diferencias):")
    for fp, c in coincidencias_E:
        print(f"  {fp:.4f} Hz -> diferencias: {c[:5]} ...")
    print("\nCoincidencias (pico enstrofía vs diferencias):")
    for fp, c in coincidencias_Z:
        print(f"  {fp:.4f} Hz -> diferencias: {c[:5]} ...")
else:
    coincidencias_E = []
    coincidencias_Z = []

# ------------------------------------------------------------
# 6. Gráficas
# ------------------------------------------------------------
plt.figure(figsize=(14,10))

# Serie temporal original
plt.subplot(2,2,1)
plt.plot(t, E, 'b-', label='Energía')
plt.plot(t, Z, 'r-', label='Enstrofía')
plt.xlabel('Tiempo (s)')
plt.ylabel('Valor')
plt.title('Evolución temporal')
plt.legend()
plt.grid(True)

# Espectro de energía
plt.subplot(2,2,2)
plt.semilogy(freq_E, power_E, 'b-', linewidth=0.8)
plt.plot(freq_peaks_E, power_E[peaks_E], 'ro', markersize=4)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Potencia')
plt.title('Espectro de energía residual')
plt.xlim(0, 1.0)
plt.grid(True)

# Espectro de enstrofía
plt.subplot(2,2,3)
plt.semilogy(freq_Z, power_Z, 'r-', linewidth=0.8)
plt.plot(freq_peaks_Z, power_Z[peaks_Z], 'bo', markersize=4)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Potencia')
plt.title('Espectro de enstrofía residual')
plt.xlim(0, 1.0)
plt.grid(True)

# Histograma de diferencias de ceros con picos superpuestos
plt.subplot(2,2,4)
if len(diffs) > 0:
    plt.hist(diffs, bins=100, range=(0, 1.0), alpha=0.7, label='Diferencias de ceros')
    for fp in freq_peaks_E:
        plt.axvline(fp, color='b', linestyle='--', linewidth=1, label='Pico energía' if fp==freq_peaks_E[0] else '')
    for fp in freq_peaks_Z:
        plt.axvline(fp, color='r', linestyle=':', linewidth=1, label='Pico enstrofía' if fp==freq_peaks_Z[0] else '')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Conteo')
    plt.title('Diferencias de ceros y picos espectrales')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig('espectro_completo_100s.png', dpi=150)
print("\nGráfica guardada: espectro_completo_100s.png")

# ------------------------------------------------------------
# 7. Correlación cruzada con una serie sintética de ceros (primeros 100 ceros)
# ------------------------------------------------------------
# Construimos una señal sintética: suma de senos con frecuencias = diferencias de ceros
if len(g) > 100:
    g0 = g[:100]
    # Tomamos las diferencias entre ceros consecutivos
    diff_consec = np.diff(g0)
    # Seleccionamos las que están en el rango de interés (0.01 a 1 Hz)
    diff_filt = diff_consec[(diff_consec > 0.01) & (diff_consec < 1.0)]
    # Construir señal sintética en los mismos tiempos t
    sintetica = np.zeros_like(t)
    for d in diff_filt[:20]:  # usar las primeras 20 diferencias para no saturar
        sintetica += np.sin(2 * np.pi * d * t)
    # Correlación con la energía residual
    corr, p_val = pearsonr(E_res, sintetica)
    print(f"\nCorrelación entre energía residual y señal sintética (primeras 20 diferencias): r = {corr:.4f}, p = {p_val:.4e}")
    # Correlación con enstrofía residual
    corrZ, p_valZ = pearsonr(Z_res, sintetica)
    print(f"Correlación entre enstrofía residual y señal sintética: r = {corrZ:.4f}, p = {p_valZ:.4e}")

print("\nAnálisis completado.")
