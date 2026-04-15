import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. Parámetros del Tsunami (Escala Chile)
g = 9.81
nu = 0.00001        # Viscosidad casi nula (caos extremo)
lambd = 1.5         # Refuerzo del control gónico para desastres
Nx, Ny = 128, 128   # Malla densa
Lx, Ly = 1000.0, 1000.0
dx, dy = Lx/Nx, Ly/Ny

# 2. Perturbación Tectónica (Fosa de Atacama)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
# Desplazamiento vertical masivo (simula el sismo)
h0 = 1.0 + 5.0 * np.exp(-((X-200)**2 + (Y-500)**2) / 800)
u0 = np.zeros((Nx, Ny))
v0 = np.zeros((Nx, Ny))
state0 = np.concatenate([h0.flatten(), u0.flatten(), v0.flatten()])

def omega_gonic(sigma):
    return np.exp(-((sigma - 0.5)**2) / (2 * 0.001**2))

def navier_stokes_tsunami(t, state, is_gonic):
    h = state[0:Nx*Ny].reshape((Nx, Ny))
    u = state[Nx*Ny:2*Nx*Ny].reshape((Nx, Ny))
    v = state[2*Nx*Ny:].reshape((Nx, Ny))

    dh = np.zeros_like(h); du = np.zeros_like(u); dv = np.zeros_like(v)
    omega = omega_gonic(0.5) if is_gonic else 0.0

    i = slice(1, -1); j = slice(1, -1)

    # Continuidad (Masa)
    dh[i, j] = -((h[2:, j]*u[2:, j] - h[:-2, j]*u[:-2, j])/(2*dx) +
                 (h[i, 2:]*v[i, 2:] - h[i, :-2]*v[i, :-2])/(2*dy))

    # Momento con término de corrección gónica
    du[i, j] = -g*(h[2:, j] - h[:-2, j])/(2*dx) + nu * ((u[2:, j] - 2*u[i, j] + u[:-2, j])/dx**2) - (lambd * omega * u[i, j])
    dv[i, j] = -g*(h[i, 2:] - h[i, :-2])/(2*dy) + nu * ((v[i, 2:] - 2*v[i, j] + v[i, :-2])/dy**2) - (lambd * omega * v[i, j])

    return np.concatenate([dh.flatten(), du.flatten(), dv.flatten()])

# 3. Simulación de impacto
t_eval = np.linspace(0, 50, 60)

print("Simulando Tsunami SI (Exigencia Nivel Cuántico)...")
s_si = time.time()
sol_si = solve_ivp(navier_stokes_tsunami, (0, 50), state0, args=(False,), t_eval=t_eval)
t_si = time.time() - s_si

print("Simulando Tsunami Gónico (Exigencia PC Normal)...")
s_go = time.time()
sol_go = solve_ivp(navier_stokes_tsunami, (0, 50), state0, args=(True,), t_eval=t_eval)
t_go = time.time() - s_go

# 4. Visualización del Impacto
plt.figure(figsize=(14, 6))

plt.subplot(121)
plt.imshow(sol_si.y[:Nx*Ny, -1].reshape(Nx, Ny), cmap='magma')
plt.title(f"Modelo SI (Turbulencia Descontrolada)\nTiempo: {round(t_si, 2)}s")

plt.subplot(122)
plt.imshow(sol_go.y[:Nx*Ny, -1].reshape(Nx, Ny), cmap='ocean')
plt.title(f"Modelo Gónico (Flujo Laminar Controlado)\nTiempo: {round(t_go, 2)}s")

plt.savefig("tsunami_chile_gonic.png")
print(f"Éxito. Reducción de carga: {round((1 - t_go/t_si)*100, 2)}%")
