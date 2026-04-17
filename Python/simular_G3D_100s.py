#!/usr/bin/env python3
# simular_G3D_100s.py
# Simulación NS3D G3D (λ=2/9) con malla 64³, Tfinal=100 s, guardado cada 0.1 s
# También guarda enstrofía (vorticidad cuadrada media)

import numpy as np
import time

N = 64
Re = 100.0
nu = 1.0/Re
Tfinal = 100.0
CFL = 0.3
Lx, Ly, Lz = 2.0, 2.0, 2.0
dx, dy, dz = Lx/N, Ly/N, Lz/N
lamb = 2.0/9.0
save_dt = 0.1   # guardar cada 0.1 segundos -> 1000 puntos

# Precalcular FFT
kx = 2 * np.pi * np.fft.fftfreq(N, dx)
ky = 2 * np.pi * np.fft.fftfreq(N, dy)
kz = 2 * np.pi * np.fft.fftfreq(N, dz)
K2 = kx[:,None,None]**2 + ky[None,:,None]**2 + kz[None,None,:]**2
K2[0,0,0] = 1.0

# Inicialización Taylor-Green
x = np.linspace(0, Lx, N, endpoint=False)
y = np.linspace(0, Ly, N, endpoint=False)
z = np.linspace(0, Lz, N, endpoint=False)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
u = np.sin(X) * np.cos(Y) * np.cos(Z)
v = -np.cos(X) * np.sin(Y) * np.cos(Z)
w = np.zeros_like(X)
p = 0.25 * (np.cos(2*X) + np.cos(2*Y)) * (np.cos(2*Z) + 2.0)
u = u.astype(np.float32)
v = v.astype(np.float32)
w = w.astype(np.float32)
p = p.astype(np.float32)

def grad_x(f): return (np.roll(f, -1,0) - np.roll(f, 1,0)) / (2*dx)
def grad_y(f): return (np.roll(f, -1,1) - np.roll(f, 1,1)) / (2*dy)
def grad_z(f): return (np.roll(f, -1,2) - np.roll(f, 1,2)) / (2*dz)
def laplacian(f):
    return ((np.roll(f,-1,0)-2*f+np.roll(f,1,0))/dx**2 +
            (np.roll(f,-1,1)-2*f+np.roll(f,1,1))/dy**2 +
            (np.roll(f,-1,2)-2*f+np.roll(f,1,2))/dz**2)

def enstrophy(u,v,w):
    # Calcula la enstrofía media: (1/2) * promedio de ω²
    wx = grad_y(w) - grad_z(v)
    wy = grad_z(u) - grad_x(w)
    wz = grad_x(v) - grad_y(u)
    return 0.5 * np.mean(wx**2 + wy**2 + wz**2)

t = 0.0
step = 0
next_save = save_dt
times = []
energies = []
enstrophies = []

start_time = time.time()
print(f"Simulación G3D (λ={lamb:.6f}) con malla {N}³, Tfinal={Tfinal}, guardado cada {save_dt} s")
while t < Tfinal:
    max_vel = max(np.max(np.abs(u)), np.max(np.abs(v)), np.max(np.abs(w)))
    dt = CFL / (max_vel * (1/dx+1/dy+1/dz)) if max_vel > 1e-12 else 0.01
    dt = min(dt, 0.05, Tfinal - t)

    ux = u*grad_x(u) + v*grad_y(u) + w*grad_z(u)
    vx = u*grad_x(v) + v*grad_y(v) + w*grad_z(v)
    wx = u*grad_x(w) + v*grad_y(w) + w*grad_z(w)

    u_star = u + dt * (-ux + nu*laplacian(u) - lamb*u)
    v_star = v + dt * (-vx + nu*laplacian(v) - lamb*v)
    w_star = w + dt * (-wx + nu*laplacian(w) - lamb*w)

    div_star = grad_x(u_star) + grad_y(v_star) + grad_z(w_star)
    div_hat = np.fft.fftn(div_star)
    phi_hat = -div_hat / (K2 * dt)
    phi_hat[0,0,0] = 0.0
    phi = np.real(np.fft.ifftn(phi_hat)).astype(np.float32)

    u = u_star - dt * grad_x(phi)
    v = v_star - dt * grad_y(phi)
    w = w_star - dt * grad_z(phi)
    p = p + phi

    if t >= next_save:
        ke = 0.5 * np.mean(u**2 + v**2 + w**2)
        enst = enstrophy(u,v,w)
        times.append(t)
        energies.append(ke)
        enstrophies.append(enst)
        next_save += save_dt
        if len(times) % 100 == 0:
            print(f"t={t:.2f} s, E={ke:.4e}, Enst={enst:.4e}")

    t += dt
    step += 1

runtime = time.time() - start_time
print(f"Terminado en {runtime:.1f} s, pasos={step}")
# Guardar
data = np.column_stack((times, energies, enstrophies))
np.savetxt('energia_enstrofia_G3D_100s.csv', data, delimiter=',', header='t,energy,enstrophy', comments='')
print(f"Guardado energia_enstrofia_G3D_100s.csv con {len(times)} puntos")
