#!/usr/bin/env python3
# prueba_huevo.py - Comparación rápida de decaimiento energético para 360°, 380° y 400°
# Ejecutar en el entorno virtual (ns_venv)

import numpy as np
import matplotlib.pyplot as plt
import time

# Parámetros comunes
N = 32               # malla pequeña para rapidez
Re = 100.0
nu = 1.0/Re
Tfinal = 2.0         # tiempo corto
CFL = 0.3
Lx, Ly, Lz = 2.0, 2.0, 2.0
dx, dy, dz = Lx/N, Ly/N, Lz/N

# Precalcular vectores de onda FFT
kx = 2 * np.pi * np.fft.fftfreq(N, dx)
ky = 2 * np.pi * np.fft.fftfreq(N, dy)
kz = 2 * np.pi * np.fft.fftfreq(N, dz)
K2 = kx[:,None,None]**2 + ky[None,:,None]**2 + kz[None,None,:]**2
K2[0,0,0] = 1.0

def run_simulacion(lambda_val, nombre):
    print(f"Ejecutando {nombre} (λ={lambda_val:.6f})...")
    start = time.time()
    # Inicialización Taylor-Green
    x = np.linspace(0, Lx, N, endpoint=False)
    y = np.linspace(0, Ly, N, endpoint=False)
    z = np.linspace(0, Lz, N, endpoint=False)
    X,Y,Z = np.meshgrid(x,y,z, indexing='ij')
    u = np.sin(X)*np.cos(Y)*np.cos(Z)
    v = -np.cos(X)*np.sin(Y)*np.cos(Z)
    w = np.zeros_like(X)
    p = 0.25*(np.cos(2*X)+np.cos(2*Y))*(np.cos(2*Z)+2.0)
    u = u.astype(np.float32); v = v.astype(np.float32); w = w.astype(np.float32); p = p.astype(np.float32)

    def grad_x(f): return (np.roll(f,-1,0)-np.roll(f,1,0))/(2*dx)
    def grad_y(f): return (np.roll(f,-1,1)-np.roll(f,1,1))/(2*dy)
    def grad_z(f): return (np.roll(f,-1,2)-np.roll(f,1,2))/(2*dz)
    def lap(f):
        return ((np.roll(f,-1,0)-2*f+np.roll(f,1,0))/dx**2 +
                (np.roll(f,-1,1)-2*f+np.roll(f,1,1))/dy**2 +
                (np.roll(f,-1,2)-2*f+np.roll(f,1,2))/dz**2)

    t = 0.0
    step = 0
    times = []
    energy = []
    save_every = 5

    while t < Tfinal:
        max_vel = max(np.max(np.abs(u)), np.max(np.abs(v)), np.max(np.abs(w)))
        dt = CFL/(max_vel*(1/dx+1/dy+1/dz)) if max_vel>1e-12 else 0.01
        dt = min(dt, 0.05, Tfinal-t)

        # Términos convectivos
        ux = u*grad_x(u) + v*grad_y(u) + w*grad_z(u)
        vx = u*grad_x(v) + v*grad_y(v) + w*grad_z(v)
        wx = u*grad_x(w) + v*grad_y(w) + w*grad_z(w)

        u_star = u + dt*(-ux + nu*lap(u) - lambda_val*u)
        v_star = v + dt*(-vx + nu*lap(v) - lambda_val*v)
        w_star = w + dt*(-wx + nu*lap(w) - lambda_val*w)

        div_star = grad_x(u_star) + grad_y(v_star) + grad_z(w_star)
        div_hat = np.fft.fftn(div_star)
        phi_hat = -div_hat/(K2*dt)
        phi_hat[0,0,0] = 0.0
        phi = np.real(np.fft.ifftn(phi_hat)).astype(np.float32)

        u = u_star - dt*grad_x(phi)
        v = v_star - dt*grad_y(phi)
        w = w_star - dt*grad_z(phi)
        p = p + phi

        if step % save_every == 0:
            ke = 0.5*np.mean(u**2+v**2+w**2)
            times.append(t)
            energy.append(ke)

        t += dt
        step += 1
        if step % 100 == 0:
            print(f"  {nombre}: paso {step}, t={t:.2f}, E={ke:.4e}")

    elapsed = time.time() - start
    print(f"  {nombre} completada en {elapsed:.1f} s, pasos={step}")
    return np.array(times), np.array(energy)

# Parámetros de los tres modelos
modelos = [
    (0.0,        "360° SI"),
    (1.0/18.0,   "380° Zeta (1/18)"),
    (1.0/9.0,    "400° Gónico (1/9)")
]

resultados = []
for lam, nom in modelos:
    t, e = run_simulacion(lam, nom)
    resultados.append((t, e, lam, nom))

# Gráfica
plt.figure(figsize=(10,6))
colores = ['red', 'orange', 'blue']
estilos = ['--', '-.', '-']
for (t, e, lam, nom), col, est in zip(resultados, colores, estilos):
    plt.semilogy(t, e, color=col, linestyle=est, linewidth=2, label=f"{nom} (λ={lam:.5f})")
    # Línea teórica exponencial (usando el primer valor de energía)
    if lam > 0:
        t_th = np.linspace(0, t[-1], 100)
        e_th = e[0] * np.exp(-2*lam*t_th)
        plt.semilogy(t_th, e_th, color=col, linestyle=':', alpha=0.6)

plt.xlabel('Tiempo (s)')
plt.ylabel('Energía cinética E(t)')
plt.title('Prueba rápida: decaimiento energético para 360°, 380° y 400°')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('prueba_huevo_360_380_400.png', dpi=150)
plt.show()

print("\n=== Resultados finales ===")
for (t, e, lam, nom) in resultados:
    print(f"{nom}: E_inicial={e[0]:.4e}, E_final={e[-1]:.4e}, reducción={100*(1-e[-1]/e[0]):.1f}%")
