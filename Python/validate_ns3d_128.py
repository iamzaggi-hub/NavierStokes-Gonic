#!/usr/bin/env python3
# validate_ns3d_128.py - Validación secuencial de modelos NS3D con malla 128³
# Ejecuta SI, G3D, G4D y genera gráfica comparativa + datos CSV

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ================= CONFIGURACIÓN GLOBAL =================
N = 128                  # malla 128³
Re = 100.0
nu = 1.0 / Re
Tfinal = 6.0
CFL = 0.3
Lx, Ly, Lz = 2.0, 2.0, 2.0

dx, dy, dz = Lx/N, Ly/N, Lz/N

# Precalcular vectores de onda para FFT (comunes a todos)
kx = 2 * np.pi * np.fft.fftfreq(N, dx)
ky = 2 * np.pi * np.fft.fftfreq(N, dy)
kz = 2 * np.pi * np.fft.fftfreq(N, dz)
K2 = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
K2[0,0,0] = 1.0   # evitar división por cero

def run_simulation(model):
    """Ejecuta una simulación para un modelo dado y guarda resultados"""
    print(f"\n{'='*50}")
    print(f"Iniciando simulación {model} con malla {N}³")
    if model == 'SI':
        lamb = 0.0
    elif model == 'G3D':
        lamb = 2.0/9.0
    elif model == 'G4D':
        lamb = 2.0/9.0 + 0.001
    else:
        raise ValueError("Modelo no reconocido")
    print(f"λ = {lamb:.6f}")

    # Inicialización de campos (Taylor-Green)
    x = np.linspace(0, Lx, N, endpoint=False)
    y = np.linspace(0, Ly, N, endpoint=False)
    z = np.linspace(0, Lz, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(X)
    p = 0.25 * (np.cos(2*X) + np.cos(2*Y)) * (np.cos(2*Z) + 2.0)
    # float32 para ahorrar memoria y acelerar
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    w = w.astype(np.float32)
    p = p.astype(np.float32)

    # Operadores diferenciales (cierres locales)
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
    start_time = time.time()

    while t < Tfinal:
        # dt adaptativo con tope
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

        # Guardar energía cada cierto número de pasos (para no saturar)
        if step % 20 == 0:
            ke = 0.5 * np.mean(u**2 + v**2 + w**2)
            times.append(t)
            energy.append(ke)

        t += dt
        step += 1
        if step % 200 == 0:
            print(f"  {model}: paso {step:6d}, t={t:.3f}, E={ke:.6e}")

    elapsed = time.time() - start_time
    print(f"Simulación {model} completada en {elapsed:.1f} s, pasos={step}")
    print(f"Energía inicial: {energy[0]:.6e}, final: {energy[-1]:.6e}")

    # Calcular pendiente de decaimiento (en escala logarítmica) para t>0.5
    times_arr = np.array(times)
    energy_arr = np.array(energy)
    mask = times_arr > 0.5
    if np.sum(mask) > 5 and lamb > 0:
        logE = np.log(energy_arr[mask])
        slope, intercept, r_value, p_value, std_err = stats.linregress(times_arr[mask], logE)
        print(f"  Pendiente experimental (log E): {slope:.4f}, teórica: {-2*lamb:.4f}, R²={r_value**2:.4f}")
    else:
        slope = None

    # Guardar datos en CSV
    np.savetxt(f'ns3d_{model}_128.csv', np.column_stack((times, energy)),
               delimiter=',', header='t,energy', comments='')

    return times, energy, lamb, slope

# ================= EJECUCIÓN PRINCIPAL =================
if __name__ == '__main__':
    print(f"Validación NS3D con malla {N}³")
    print(f"Dominio: {Lx}x{Ly}x{Lz}, Re={Re}, Tfinal={Tfinal}, CFL={CFL}")
    print("Ejecutando modelos en secuencia: SI, G3D, G4D")

    resultados = {}
    for modelo in ['SI', 'G3D', 'G4D']:
        t, e, l, s = run_simulation(modelo)
        resultados[modelo] = {'times': t, 'energy': e, 'lambda': l, 'slope': s}

    # Gráfica comparativa
    plt.figure(figsize=(12, 7))
    colores = {'SI': 'red', 'G3D': 'blue', 'G4D': 'purple'}
    estilos = {'SI': '--', 'G3D': '-', 'G4D': '-.'}

    for modelo, data in resultados.items():
        plt.semilogy(data['times'], data['energy'],
                     color=colores[modelo], linestyle=estilos[modelo],
                     linewidth=2, label=f"{modelo} (λ={data['lambda']:.5f})")
        # Línea teórica de decaimiento exponencial (solo para G3D y G4D)
        if data['lambda'] > 0:
            t_theo = np.linspace(0, Tfinal, 100)
            E0 = data['energy'][0]
            E_theo = E0 * np.exp(-2 * data['lambda'] * t_theo)
            plt.semilogy(t_theo, E_theo, color=colores[modelo], linestyle=':',
                         alpha=0.5, label=f"teórico {modelo}")

    plt.xlabel('Tiempo (s)', fontsize=12)
    plt.ylabel('Energía cinética E(t)', fontsize=12)
    plt.title(f'Comparación de modelos NS3D - Malla {N}³', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'comparativa_NS3D_{N}.png', dpi=200)
    print(f"\nGráfica guardada como 'comparativa_NS3D_{N}.png'")

    # Mostrar resumen de pendientes
    print("\n=== Resumen de pendientes de decaimiento (log E) ===")
    for modelo, data in resultados.items():
        if data['lambda'] > 0 and data['slope'] is not None:
            print(f"{modelo}: experimental = {data['slope']:.4f}, teórica = {-2*data['lambda']:.4f}, error = {abs(data['slope'] + 2*data['lambda']):.4f}")
        elif modelo == 'SI':
            print(f"{modelo}: sin decaimiento (λ=0), energía casi constante")

    plt.show()
