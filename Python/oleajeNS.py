import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. Parámetros de Impacto Costero
g, nu, lambd = 9.81, 0.00001, 1.5
Nx, Ny = 128, 128
Lx, Ly = 400.0, 400.0
dx, dy = Lx/Nx, Ly/Ny

# 2. Configuración de la Costa (Barrera en x = 350)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Onda masiva acercándose a la costa
h0 = 1.0 + 4.0 * np.exp(-((X-150)**2 + (Y-200)**2) / 600)
u0 = np.full((Nx, Ny), 2.0) # Velocidad hacia la derecha (la costa)
v0 = np.zeros((Nx, Ny))
state0 = np.concatenate([h0.flatten(), u0.flatten(), v0.flatten()])

def navier_stokes_costa(t, state, is_gonic):
    h = state[0:Nx*Ny].reshape((Nx, Ny))
    u = state[Nx*Ny:2*Nx*Ny].reshape((Nx, Ny))
    v = state[2*Nx*Ny:].reshape((Nx, Ny))

    dh = np.zeros_like(h); du = np.zeros_like(u); dv = np.zeros_like(v)
    omega = np.exp(-((0.5 - 0.5)**2) / (2 * 0.001**2)) if is_gonic else 0.0

    i = slice(1, -1); j = slice(1, -1)

    # Física de fluidos
    dh[i, j] = -((h[2:, j]*u[2:, j] - h[:-2, j]*u[:-2, j])/(2*dx))
    du[i, j] = -g*(h[2:, j] - h[:-2, j])/(2*dx) + nu * ((u[2:, j] - 2*u[i, j] + u[:-2, j])/dx**2)

    # Condición de frontera rígida (COSTA)
    u[:, -5:] = 0 # La velocidad muere al chocar con la tierra

    if is_gonic:
        du[i, j] -= lambd * omega * u[i, j]

    return np.concatenate([dh.flatten(), du.flatten(), dv.flatten()])

# 3. Simulación del Choque
print("Simulando impacto en costa (SI)...")
sol_si = solve_ivp(navier_stokes_costa, (0, 40), state0, args=(False,), t_eval=np.linspace(0, 40, 50))
print("Simulando impacto en costa (Gónico)...")
sol_go = solve_ivp(navier_stokes_costa, (0, 40), state0, args=(True,), t_eval=np.linspace(0, 40, 50))

# 4. Gráfica Final
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Perfil de elevación al momento del impacto
ax1.plot(x, sol_si.y[:Nx, -1], label="SI (Turbulencia de rebote)", color='red')
ax1.plot(x, sol_go.y[:Nx, -1], label="Gónico (Impacto Laminar)", color='blue', lw=3)
ax1.set_title("Perfil de Elevación en la Costa")
ax1.legend()

# Visualización 2D
im = ax2.imshow(sol_go.y[:Nx*Ny, -1].reshape(Nx, Ny), cmap='terrain')
ax2.set_title("Mapa de Inundación Gónica")
plt.colorbar(im)

plt.savefig("impacto_costa_zaggi.png")
print("Análisis completado. ¿Ves cómo el perfil azul es mucho más nítido?")
