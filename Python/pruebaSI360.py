import numpy as np
import matplotlib.pyplot as plt

# CONFIGURACIÓN MODO CAOS (SI 360°)
Nx, Ny = 128, 128
Lx, Ly = 600.0, 600.0
dx, dy = Lx/Nx, Ly/Ny
g, nu = 9.81, 0.01
dt = 0.005
pasos = 10000

# PRUEBA DEL CAOS
lambd = 0  # Eliminamos la estabilidad gónica
modo = "SI_360_CAOS"

# Inicialización
h = 1.0 + 5.0 * np.exp(-((np.linspace(0,Lx,Nx)[:,None]-Lx/2)**2 + (np.linspace(0,Ly,Ny)-Ly/2)**2) / 1000)
u, v = np.zeros((Nx, Ny)), np.zeros((Nx, Ny))
energia_hist = []

print(f"⚠️ Iniciando Modo {modo} (Sin protección gónica)...")

try:
    for n in range(pasos):
        # Derivadas
        h_x = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) / (2*dx)
        h_y = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2*dy)
        u_x = (np.roll(u, -1, 0) - np.roll(u, 1, 0)) / (2*dx)
        v_y = (np.roll(v, -1, 1) - np.roll(v, 1, 1)) / (2*dy)

        # Navier-Stokes SIN Término Gónico
        u_new = u + dt * (-g*h_x + nu*((np.roll(u,-1,0)+np.roll(u,1,0)-2*u)/dx**2) - lambd*u)
        v_new = v + dt * (-g*h_y + nu*((np.roll(v,-1,1)+np.roll(v,1,1)-2*v)/dy**2) - lambd*v)
        h_new = h + dt * (-(h*u_x + h*v_y))

        h, u, v = h_new, u_new, v_new

        e_k = np.mean(0.5 * (u**2 + v**2))
        energia_hist.append(e_k)

        # Detector de colapso
        if np.isnan(e_k) or e_k > 10.0: # Un salto a 10.0 ya es una explosión física
            print(f"\n❌ ¡COLAPSO! El sistema SI divergió en el paso {n}.")
            print(f"Energía final registrada: {e_k}")
            break

        if n % 500 == 0:
            print(f"Paso {n}: Energía = {e_k:.8f} (Inestabilidad creciendo...)")

except Exception as e:
    print(f"Error: {e}")

# Gráfico del desastre
plt.figure(figsize=(10, 5))
plt.plot(energia_hist, color='red', label="Modo SI 360 (λ=0)")
plt.title("Colapso del Sistema Tradicional (Sin estabilidad gónica)")
plt.xlabel("Pasos de tiempo")
plt.ylabel("Energía Cinética Media")
plt.yscale('log') # Escala logarítmica para ver el disparo exponencial
plt.grid(True, which="both", ls="-")
plt.legend()
plt.show()
