import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. Parámetros de ESTRÉS (Basados en tus originales)
g = 9.81
nu_estres = 0.0001  # 10 veces menos viscosidad para forzar caos
lambd = 1.0
Pfaff = 1.0
Nx, Ny = 128, 128
Lx, Ly = 400.0, 400.0
dx, dy = Lx/Nx, Ly/Ny

def omega_func(sigma, epsilon=0.001):
    return np.exp(-((sigma - 0.5)**2) / (2 * epsilon**2))

# 2. Condición inicial AGRESIVA
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
# Gota con el doble de amplitud (1.0) para inyectar más energía
h0 = 1.0 + 1.0 * np.exp(-((X-200)**2 + (Y-200)**2) / 100)
u0 = np.zeros((Nx, Ny))
v0 = np.zeros((Nx, Ny))
state0 = np.concatenate([h0.flatten(), u0.flatten(), v0.flatten()])

def navier_stokes_stress(t, state, is_gonic):
    h = state[0:Nx*Ny].reshape((Nx, Ny))
    u = state[Nx*Ny:2*Nx*Ny].reshape((Nx, Ny))
    v = state[2*Nx*Ny:].reshape((Nx, Ny))

    dh = np.zeros_like(h); du = np.zeros_like(u); dv = np.zeros_like(v)
    omega = omega_func(0.5) if is_gonic else 0.0

    idx_i = slice(1, -1); idx_j = slice(1, -1)

    # Continuidad
    dh[idx_i, idx_j] = -((h[2:, idx_j]*u[2:, idx_j] - h[:-2, idx_j]*u[:-2, idx_j])/(2*dx) +
                         (h[idx_i, 2:]*v[idx_i, 2:] - h[idx_i, :-2]*v[idx_i, :-2])/(2*dy))

    # Difusión (Viscosidad reducida)
    lap_u = (u[2:, idx_j] - 2*u[idx_i, idx_j] + u[:-2, idx_j])/dx**2 + (u[idx_i, 2:] - 2*u[idx_i, idx_j] + u[idx_i, :-2])/dy**2
    lap_v = (v[2:, idx_j] - 2*v[idx_i, idx_j] + v[:-2, idx_j])/dx**2 + (v[idx_i, 2:] - 2*v[idx_i, idx_j] + v[idx_i, :-2])/dy**2

    du[idx_i, idx_j] = -g*(h[2:, idx_j] - h[:-2, idx_j])/(2*dx) + nu_estres * lap_u
    dv[idx_i, idx_j] = -g*(h[idx_i, 2:] - h[idx_i, :-2])/(2*dy) + nu_estres * lap_v

    if is_gonic:
        du[idx_i, idx_j] -= lambd * Pfaff * omega * u[idx_i, idx_j]
        dv[idx_i, idx_j] -= lambd * Pfaff * omega * v[idx_i, idx_j]

    return np.concatenate([dh.flatten(), du.flatten(), dv.flatten()])

# 3. Simulación
t_span = (0, 30) # Más tiempo para ver si diverge
t_eval = np.linspace(0, 30, 50)

print("Iniciando prueba de estrés SI...")
start = time.time()
sol_si = solve_ivp(navier_stokes_stress, t_span, state0, args=(False,), t_eval=t_eval, method='RK45')
t_si = time.time() - start

print("Iniciando prueba de estrés Gónica...")
start = time.time()
sol_go = solve_ivp(navier_stokes_stress, t_span, state0, args=(True,), t_eval=t_eval, method='RK45')
t_go = time.time() - start

# 4. Gráfica de Estrés
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(sol_si.t, [np.sum(sol_si.y[Nx*Ny:, i]**2) for i in range(len(sol_si.t))], label='SI (Inestable)')
plt.plot(sol_go.t, [np.sum(sol_go.y[Nx*Ny:, i]**2) for i in range(len(sol_go.t))], label='Gónico (Controlado)')
plt.title("Energía en Condiciones Críticas")
plt.legend()

plt.subplot(122)
diff = sol_si.y[:Nx*Ny, -1] - sol_go.y[:Nx*Ny, -1]
plt.imshow(diff.reshape(Nx, Ny), cmap='seismic')
plt.title("Diferencia de Estructura (SI vs Gónico)")
plt.colorbar()

plt.savefig("prueba_estres_zaggi.png")
print(f"Completado. Mejora de tiempo: {round((1 - t_go/t_si)*100, 2)}%")
