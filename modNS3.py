import numpy as np
import matplotlib.pyplot as plt
import time

# 1. PARÁMETROS DE LA PRUEBA DEFINITIVA
Nx, Ny = 128, 128 # Malla ligeramente menor para velocidad en 10k pasos
Lx, Ly = 600.0, 600.0
dx, dy = Lx/Nx, Ly/Ny
g, nu = 9.81, 0.01
dt = 0.005
pasos = 10000

# Cambia esto para probar:
# lambd = 1/9  # MODO GÓNICO (Estable)
# lambd = 0    # MODO SI (Probable Colapso)
modo = "GONICO" # Cambia a "SI" para la comparativa
lambd = 1/9 if modo == "GONICO" else 0

# Inicialización
h = 1.0 + 5.0 * np.exp(-((np.linspace(0,Lx,Nx)[:,None]-Lx/2)**2 + (np.linspace(0,Ly,Ny)-Ly/2)**2) / 1000)
u, v = np.zeros((Nx, Ny)), np.zeros((Nx, Ny))
energia_hist = []

print(f"Iniciando Modo {modo} con {pasos} pasos...")

try:
    for n in range(pasos):
        # Derivadas (Diferencias centrales)
        h_x = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) / (2*dx)
        h_y = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2*dy)
        u_x = (np.roll(u, -1, 0) - np.roll(u, 1, 0)) / (2*dx)
        v_y = (np.roll(v, -1, 1) - np.roll(v, 1, 1)) / (2*dy)

        # Navier-Stokes Gónico / SI
        u_new = u + dt * (-g*h_x + nu*((np.roll(u,-1,0)+np.roll(u,1,0)-2*u)/dx**2) - lambd*u)
        v_new = v + dt * (-g*h_y + nu*((np.roll(v,-1,1)+np.roll(v,1,1)-2*v)/dy**2) - lambd*v)
        h_new = h + dt * (-(h*u_x + h*v_y))

        h, u, v = h_new, u_new, v_new

        # Registro de energía cinética media
        e_k = np.mean(0.5 * (u**2 + v**2))
        energia_hist.append(e_k)

        if np.isnan(e_k) or e_k > 1e10:
            print(f"¡COLAPSO DETECTADO en paso {n}! Energía fuera de control.")
            break

        if n % 1000 == 0:
            print(f"Paso {n}: Energía = {e_k:.8f}")

except Exception as e:
    print(f"Error en la simulación: {e}")

# Graficar el comportamiento de la energía
plt.figure(figsize=(10, 5))
plt.plot(energia_hist, label=f"Modo {modo} (λ={lambd})")
plt.title("Evolución de la Energía en el Tiempo")
plt.xlabel("Pasos de tiempo")
plt.ylabel("Energía Cinética Media")
plt.grid(True)
plt.legend()
plt.show()
