#!/usr/bin/env python3
# validate_ns3d.py - Validación paralela de los tres modelos NS3D (SI, G3D, G4D)
# Uso: python validate_ns3d.py
# Requiere: numpy, matplotlib, multiprocessing

import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# ================= CONFIGURACIÓN GLOBAL =================
N = 64                 # malla 64^3 (ajustable, ~4 MB por simulación)
Re = 100.0             # número de Reynolds
nu = 1.0 / Re
Tfinal = 6.0           # tiempo final (antes de la inestabilidad numérica)
CFL = 0.3              # factor de estabilidad (más conservador)
Lx, Ly, Lz = 2.0, 2.0, 2.0

dx, dy, dz = Lx/N, Ly/N, Lz/N
# Precalcular vectores de onda para FFT
kx = 2 * np.pi * np.fft.fftfreq(N, dx)
ky = 2 * np.pi * np.fft.fftfreq(N, dy)
kz = 2 * np.pi * np.fft.fftfreq(N, dz)
K2 = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
K2[0,0,0] = 1.0   # evitar división por cero

def run_simulation(model):
    """Ejecuta una simulación para un modelo dado y devuelve (tiempos, energías)"""
    print(f"Iniciando simulación {model} ...")
    start_time = time.time()

    # Parámetro lambda según modelo
    if model == 'SI':
        lamb = 0.0
    elif model == 'G3D':
        lamb = 2.0/9.0
    elif model == 'G4D':
        lamb = 2.0/9.0 + 0.001
    else:
        raise ValueError("Modelo no reconocido")

    # Inicialización de campos (Taylor-Green)
    x = np.linspace(0, Lx, N, endpoint=False)
    y = np.linspace(0, Ly, N, endpoint=False)
    z = np.linspace(0, Lz, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(X)
    p = 0.25 * (np.cos(2*X) + np.cos(2*Y)) * (np.cos(2*Z) + 2.0)
    # Convertir a float32 para ahorrar memoria
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    w = w.astype(np.float32)
    p = p.astype(np.float32)

    # Operadores diferenciales (cierre sobre funciones locales para eficiencia)
    def grad_x(f):
        return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2*dx)
    def grad_y(f):
        return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2*dy)
    def grad_z(f):
        return (np.roll(f, -1, axis=2) - np.roll(f, 1, axis=2)) / (2*dz)
    def laplacian(f):
        lap = (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)) / dx**2
        lap += (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)) / dy**2
        lap += (np.roll(f, -1, axis=2) - 2*f + np.roll(f, 1, axis=2)) / dz**2
        return lap

    # Bucle temporal
    t = 0.0
    step = 0
    times = []
    energy = []

    while t < Tfinal:
        # dt adaptativo con tope de 0.05 para evitar inestabilidades
        max_vel = max(np.max(np.abs(u)), np.max(np.abs(v)), np.max(np.abs(w)))
        if max_vel < 1e-12:
            dt = 0.01
        else:
            dt = CFL / (max_vel * (1/dx + 1/dy + 1/dz))
        dt = min(dt, 0.05, Tfinal - t)

        # Velocidad intermedia (sin presión)
        ux = u * grad_x(u) + v * grad_y(u) + w * grad_z(u)
        vx = u * grad_x(v) + v * grad_y(v) + w * grad_z(v)
        wx = u * grad_x(w) + v * grad_y(w) + w * grad_z(w)

        u_star = u + dt * (-ux + nu * laplacian(u) - lamb * u)
        v_star = v + dt * (-vx + nu * laplacian(v) - lamb * v)
        w_star = w + dt * (-wx + nu * laplacian(w) - lamb * w)

        # Corrección de presión (FFT)
        div_star = grad_x(u_star) + grad_y(v_star) + grad_z(w_star)
        div_hat = np.fft.fftn(div_star)
        phi_hat = -div_hat / (K2 * dt)
        phi_hat[0,0,0] = 0.0
        phi = np.real(np.fft.ifftn(phi_hat)).astype(np.float32)

        u = u_star - dt * grad_x(phi)
        v = v_star - dt * grad_y(phi)
        w = w_star - dt * grad_z(phi)
        p = p + phi

        # Guardar energía cada 10 pasos
        if step % 10 == 0:
            ke = 0.5 * np.mean(u**2 + v**2 + w**2)
            times.append(t)
            energy.append(ke)

        t += dt
        step += 1
        if step % 100 == 0:
            print(f"  {model}: paso {step}, t={t:.2f}, E={ke:.4e}")

    elapsed = time.time() - start_time
    print(f"Simulación {model} completada en {elapsed:.1f} s, pasos={step}")
    return times, energy, lamb

# ================= EJECUCIÓN EN PARALELO =================
if __name__ == '__main__':
    print("Iniciando validación de los tres modelos (SI, G3D, G4D) en paralelo...")
    print(f"Malla {N}^3, Tfinal={Tfinal}, Re={Re}, CFL={CFL}")
    with Pool(processes=3) as pool:
        resultados = pool.map(run_simulation, ['SI', 'G3D', 'G4D'])

    # Graficar resultados
    plt.figure(figsize=(10, 6))
    colores = {'SI': 'red', 'G3D': 'blue', 'G4D': 'purple'}
    estilos = {'SI': '--', 'G3D': '-', 'G4D': '-.'}
    for (times, energy, lamb), modelo in zip(resultados, ['SI', 'G3D', 'G4D']):
        plt.semilogy(times, energy, color=colores[modelo], linestyle=estilos[modelo],
                     linewidth=2, label=f"{modelo} (λ={lamb:.5f})")
        # Línea teórica de decaimiento exponencial (solo para G3D y G4D)
        if lamb > 0:
            t_theo = np.linspace(0, Tfinal, 100)
            E0 = energy[0] if energy else 0.1
            E_theo = E0 * np.exp(-2 * lamb * t_theo)
            plt.semilogy(t_theo, E_theo, color=colores[modelo], linestyle=':',
                         alpha=0.5, label=f"teórico {modelo}")

    plt.xlabel('Tiempo (s)')
    plt.ylabel('Energía cinética E(t)')
    plt.title('Comparación de modelos NS3D: SI vs Gónico 3D vs Gónico 4D')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparativa_NS3D.png', dpi=200)
    print("Gráfica guardada como 'comparativa_NS3D.png'")
    plt.show()
