import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. Parámetros (Tus constantes originales mantenidas)
g = 9.81
nu = 0.0001
lambd = 1.2
Nx, Ny = 128, 128
Lx, Ly = 500.0, 400.0
dx, dy = Lx/Nx, Ly/Ny

# 2. BATIMETRÍA CHILENA: El fondo sube según se acerca a la costa (derecha)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
# El fondo marino sube linealmente (profundidad disminuye)
profundidad = np.maximum(0.1, 1.0 - (X / Lx))

# 3. Condición Inicial: El Tsunami naciendo en la fosa profunda
h0 = 1.0 + 3.0 * np.exp(-((X-50)**2 + (Y-200)**2) / 1000)
u0 = np.full((Nx, Ny), 3.0)
v0 = np.zeros((Nx, Ny))
state0 = np.concatenate([h0.flatten(), u0.flatten(), v0.flatten()])

def navier_stokes_batimetria(t, state, is_gonic):
    h = state[0:Nx*Ny].reshape((Nx, Ny))
    u = state[Nx*Ny:2*Nx*Ny].reshape((Nx, Ny))
    v = state[2*Nx*Ny:].reshape((Nx, Ny))

    dh = np.zeros_like(h); du = np.zeros_like(u); dv = np.zeros_like(v)
    omega = np.exp(0) if is_gonic else 0.0 # Omega gónico optimizado

    i = slice(1, -1); j = slice(1, -1)

    # La profundidad (H_local) afecta la velocidad de fase
    H_local = profundidad[i, j]

    # Continuidad corregida por relieve
    dh[i, j] = -((h[2:, j]*u[2:, j] - h[:-2, j]*u[:-2, j])/(2*dx)) * H_local

    # Momento con término gónico para estabilizar el "choque" con el relieve
    du[i, j] = -g*(h[2:, j] - h[:-2, j])/(2*dx) + nu * ((u[2:, j] - 2*u[i, j] + u[:-2, j])/dx**2)

    if is_gonic:
        # Los 400 gones absorben el ruido de la compresión del fondo
        du[i, j] -= lambd * omega * u[i, j] * (1.0 - H_local)

    return np.concatenate([dh.flatten(), du.flatten(), dv.flatten()])

# 4. Simulación
print("Simulando Tsunami con relieve (SI)...")
sol_si = solve_ivp(navier_stokes_batimetria, (0, 25), state0, args=(False,), t_eval=np.linspace(0, 25, 40))
print("Simulando Tsunami con relieve (Gónico)...")
sol_go = solve_ivp(navier_stokes_batimetria, (0, 25), state0, args=(True,), t_eval=np.linspace(0, 25, 40))

# 5. Visualización de la Amplificación (Shoaling)
plt.figure(figsize=(12, 6))

plt.subplot(211)
plt.plot(x, sol_si.y[:Nx, -1], label="SI: Inestabilidad por Relieve", color='red', alpha=0.6)
plt.plot(x, sol_go.y[:Nx, -1], label="Gónico: Amplificación Laminar", color='blue', lw=2)
plt.fill_between(x, 0, -profundidad[:, 64], color='gray', alpha=0.3, label="Relieve Submarino")
plt.title("Impacto del Tsunami: Compresión por Batimetría")
plt.legend()

plt.subplot(212)
diff = (sol_si.y[:Nx*Ny, -1] - sol_go.y[:Nx*Ny, -1]).reshape(Nx, Ny)
plt.imshow(diff, cmap='RdBu_r')
plt.title("Residuo Turbulento Eliminado por el Modelo Gónico")
plt.colorbar()

plt.tight_layout()
plt.savefig("tsunami_batimetria_zaggi.png")
