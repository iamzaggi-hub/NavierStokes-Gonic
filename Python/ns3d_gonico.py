#!/usr/bin/env python3
# ns3d_gonico.py - Simulación 3D/4D de Navier-Stokes con amortiguamiento gónico
# Ejecutar dentro de un entorno virtual con numpy y matplotlib

import time
import numpy as np
import matplotlib.pyplot as plt

# ================= CONFIGURACIÓN =================
# Elige el modelo: 'SI', 'G3D' o 'G4D'
MODELO = 'G3D'   # Cambiar a 'SI' o 'G4D' para comparar

# Parámetros físicos
Lx, Ly, Lz = 2.0, 2.0, 2.0      # dominio cúbico
Nx, Ny, Nz = 64, 64, 64         # resolución (64^3 = 262k celdas)
Re = 100.0                      # número de Reynolds
nu = 1.0 / Re
Tfinal = 10.0
CFL = 0.4

# Lambda según modelo
if MODELO == 'SI':
    LAMBDA = 0.0
    titulo = "SI (360°) - λ=0"
elif MODELO == 'G3D':
    LAMBDA = 2.0/9.0            # 0.222222...
    titulo = "Gónico 3D - λ=2/9"
elif MODELO == 'G4D':
    LAMBDA = 2.0/9.0 + 0.001    # 0.223222...
    titulo = "Gónico 4D - λ=2/9+0.001"
else:
    raise ValueError("MODELO debe ser 'SI', 'G3D' o 'G4D'")

print(f"Ejecutando {titulo} con λ={LAMBDA:.6f}")
print(f"Malla {Nx}x{Ny}x{Nz}, memoria estimada: {3*Nx*Ny*Nz*4/1e6:.1f} MB")

# ================= INICIALIZACIÓN =================
dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
z = np.linspace(0, Lz, Nz, endpoint=False)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Campo inicial: Taylor-Green
u = np.sin(X) * np.cos(Y) * np.cos(Z)
v = -np.cos(X) * np.sin(Y) * np.cos(Z)
w = np.zeros_like(X)
p = 0.25 * (np.cos(2*X) + np.cos(2*Y)) * (np.cos(2*Z) + 2.0)

# Convertir a float32 para ahorrar memoria
u = u.astype(np.float32)
v = v.astype(np.float32)
w = w.astype(np.float32)
p = p.astype(np.float32)

# Operadores (diferencias centradas)
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

# ================= BUCLE TEMPORAL =================
t = 0.0
step = 0
times = []
energy = []
enstrophy = []

start_time = time.time()

while t < Tfinal:
    # dt adaptativo
    max_vel = max(np.max(np.abs(u)), np.max(np.abs(v)), np.max(np.abs(w)))
    if max_vel < 1e-12:
        dt = 0.01
    else:
        dt = CFL / (max_vel * (1/dx + 1/dy + 1/dz))
    dt = min(dt, 0.1, Tfinal - t)

    # --- Velocidad intermedia (sin presión) ---
    ux = u * grad_x(u) + v * grad_y(u) + w * grad_z(u)
    vx = u * grad_x(v) + v * grad_y(v) + w * grad_z(v)
    wx = u * grad_x(w) + v * grad_y(w) + w * grad_z(w)

    u_star = u + dt * ( -ux + nu * laplacian(u) - LAMBDA * u )
    v_star = v + dt * ( -vx + nu * laplacian(v) - LAMBDA * v )
    w_star = w + dt * ( -wx + nu * laplacian(w) - LAMBDA * w )

    # --- Corrección de presión (FFT, condiciones periódicas) ---
    div_star = grad_x(u_star) + grad_y(v_star) + grad_z(w_star)
    div_hat = np.fft.fftn(div_star)
    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)
    kz = 2 * np.pi * np.fft.fftfreq(Nz, dz)
    K2 = kx[:,None,None]**2 + ky[None,:,None]**2 + kz[None,None,:]**2
    K2[0,0,0] = 1.0
    phi_hat = - div_hat / (K2 * dt)
    phi_hat[0,0,0] = 0.0
    phi = np.real(np.fft.ifftn(phi_hat)).astype(np.float32)

    u = u_star - dt * grad_x(phi)
    v = v_star - dt * grad_y(phi)
    w = w_star - dt * grad_z(phi)
    p = p + phi

    # --- Guardar métricas ---
    if step % 10 == 0:
        ke = 0.5 * np.mean(u**2 + v**2 + w**2)
        vort_x = grad_y(w) - grad_z(v)
        vort_y = grad_z(u) - grad_x(w)
        vort_z = grad_x(v) - grad_y(u)
        enst = np.mean(vort_x**2 + vort_y**2 + vort_z**2)
        energy.append(ke)
        enstrophy.append(enst)
        times.append(t)

    t += dt
    step += 1
    if step % 50 == 0:
        print(f"Paso {step:5d}, t={t:.3f}, E={ke:.6e}")

runtime = time.time() - start_time
print(f"\nTerminado en {runtime:.1f} s, {step} pasos")
print(f"Energía inicial: {energy[0]:.6e}, final: {energy[-1]:.6e}")

# ================= GRÁFICAS =================
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.semilogy(times, energy, 'b-', linewidth=2)
plt.xlabel('Tiempo')
plt.ylabel('Energía cinética')
plt.title(f'{titulo}\nDecaimiento exponencial esperado: pendiente -2λ = {-2*LAMBDA:.4f}')
plt.grid(True)

plt.subplot(1,2,2)
plt.semilogy(times, enstrophy, 'r-', linewidth=2)
plt.xlabel('Tiempo')
plt.ylabel('Enstrofía')
plt.title('Evolución de la vorticidad')
plt.grid(True)

plt.tight_layout()
plt.savefig(f'ns3d_{MODELO}.png', dpi=150)
print(f"Gráfica guardada como ns3d_{MODELO}.png")
