import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. Parámetros (TUS CONSTANTES ORIGINALES)
g = 9.81
nu_si = 0.001
nu_gonic = 0.001
lambd = 1.0
Pfaff = 1.0
Nx, Ny = 128, 128
Lx, Ly = 400.0, 400.0
dx, dy = Lx/Nx, Ly/Ny

def omega_func(sigma, epsilon=0.001):
    return np.exp(-((sigma - 0.5)**2) / (2 * epsilon**2))

# 2. Condición inicial
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
h0 = 1.0 + 0.5 * np.exp(-((X-200)**2 + (Y-200)**2) / 100)
u0 = np.zeros((Nx, Ny))
v0 = np.zeros((Nx, Ny))
state0 = np.concatenate([h0.flatten(), u0.flatten(), v0.flatten()])

# 3. Función del Modelo
def navier_stokes(t, state, is_gonic):
    h = state[0:Nx*Ny].reshape((Nx, Ny))
    u = state[Nx*Ny:2*Nx*Ny].reshape((Nx, Ny))
    v = state[2*Nx*Ny:].reshape((Nx, Ny))

    dh = np.zeros_like(h)
    du = np.zeros_like(u)
    dv = np.zeros_like(v)

    omega = omega_func(0.5) if is_gonic else 0.0

    # Derivadas (Diferencias finitas centrales)
    # Nota: Se omiten bordes para estabilidad simple
    idx_i = slice(1, -1)
    idx_j = slice(1, -1)

    dh[idx_i, idx_j] = -((h[2:, idx_j]*u[2:, idx_j] - h[:-2, idx_j]*u[:-2, idx_j])/(2*dx) +
                         (h[idx_i, 2:]*v[idx_i, 2:] - h[idx_i, :-2]*v[idx_i, :-2])/(2*dy))

    lap_u = (u[2:, idx_j] - 2*u[idx_i, idx_j] + u[:-2, idx_j])/dx**2 + \
            (u[idx_i, 2:] - 2*u[idx_i, idx_j] + u[idx_i, :-2])/dy**2

    lap_v = (v[2:, idx_j] - 2*v[idx_i, idx_j] + v[:-2, idx_j])/dx**2 + \
            (v[idx_i, 2:] - 2*v[idx_i, idx_j] + v[idx_i, :-2])/dy**2

    du[idx_i, idx_j] = -g*(h[2:, idx_j] - h[:-2, idx_j])/(2*dx) + nu_si * lap_u
    dv[idx_i, idx_j] = -g*(h[idx_i, 2:] - h[idx_i, :-2])/(2*dy) + nu_si * lap_v

    if is_gonic:
        du[idx_i, idx_j] -= lambd * Pfaff * omega * u[idx_i, idx_j]
        dv[idx_i, idx_j] -= lambd * Pfaff * omega * v[idx_i, idx_j]

    return np.concatenate([dh.flatten(), du.flatten(), dv.flatten()])

# 4. Resolución y Medición
t_span = (0, 20)
t_eval = np.linspace(0, 20, 40)

print("Resolviendo SI...")
start = time.time()
sol_si = solve_ivp(navier_stokes, t_span, state0, args=(False,), t_eval=t_eval, method='RK45')
t_si = time.time() - start

print("Resolviendo Gónico...")
start = time.time()
sol_gonic = solve_ivp(navier_stokes, t_span, state0, args=(True,), t_eval=t_eval, method='RK45')
t_gonic = time.time() - start

# 5. Resultados
print(f"--- RESULTADOS ---")
print(f"Tiempo SI: {t_si:.4f}s")
print(f"Tiempo Gónico: {t_gonic:.4f}s")
print(f"Pasos SI: {len(sol_si.t)}")
print(f"Pasos Gónico: {len(sol_gonic.t)}")

# 6. Gráfica de salida
plt.figure(figsize=(12, 4))
plt.subplot(131)
e_si = [np.sum(sol_si.y[Nx*Ny:, i]**2) for i in range(len(sol_si.t))]
e_go = [np.sum(sol_gonic.y[Nx*Ny:, i]**2) for i in range(len(sol_gonic.t))]
plt.plot(sol_si.t, e_si, label='SI')
plt.plot(sol_gonic.t, e_go, label='Gónico')
plt.title("Energía Cinética")
plt.legend()

plt.subplot(132)
plt.imshow(sol_si.y[:Nx*Ny, -1].reshape(Nx, Ny), cmap='viridis')
plt.title("SI Final")

plt.subplot(133)
plt.imshow(sol_gonic.y[:Nx*Ny, -1].reshape(Nx, Ny), cmap='viridis')
plt.title("Gónico Final")

plt.tight_layout()
plt.savefig("resultado_ns.png")
print("Gráfica guardada como 'resultado_ns.png'")
