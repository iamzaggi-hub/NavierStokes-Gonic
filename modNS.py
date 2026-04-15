import numpy as np
import matplotlib.pyplot as plt
import time

# 1. CONFIGURACIÓN MEJORADA
Nx, Ny = 256, 256
Lx, Ly = 600.0, 600.0
dx, dy = Lx/Nx, Ly/Ny

# Parámetros físicos (Modelo Gónico)
g = 9.81
nu = 0.01      # Viscosidad cinemática
lambd = 1/9    # Valor teórico puro (puedes subirlo a 1.5 si usas dt mayor)
dt = 0.002     # Paso de tiempo reducido para mayor precisión
pasos = 1000

# 2. CONDICIONES INICIALES (Perturbación Gaussiana)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
h = 1.0 + 5.0 * np.exp(-((X-Lx/2)**2 + (Y-Ly/2)**2) / 1000)
u = np.zeros((Nx, Ny))
v = np.zeros((Nx, Ny))

def deriv(f, axis, d):
    """Derivada central de 2do orden con roll vectorizado"""
    if axis == 0: # Eje X
        return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2*d)
    else: # Eje Y
        return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2*d)

def laplaciano(f, d):
    """Laplaciano discreto de 5 puntos"""
    return (np.roll(f, -1, axis=0) + np.roll(f, 1, axis=0) +
            np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1) - 4*f) / (d**2)

# 3. BUCLE DE INTEGRACIÓN (Motor de Navier-Stokes Gónico)
print(f"Ejecutando simulación avanzada en malla {Nx}x{Nx}...")
inicio = time.time()

for n in range(pasos):
    # Gradientes de altura y velocidad
    h_x, h_y = deriv(h, 0, dx), deriv(h, 1, dy)
    u_x, u_y = deriv(u, 0, dx), deriv(u, 1, dy)
    v_x, v_y = deriv(v, 0, dx), deriv(v, 1, dy)

    # Términos de Navier-Stokes Gónico:
    # 1. Advección (u * grad(u))
    # 2. Gradiente de presión (g * grad(h))
    # 3. Difusión (nu * laplaciano)
    # 4. AMORTIGUAMIENTO GÓNICO (-lambd * u) -> El estabilizador

    u_new = u + dt * (-(u*u_x + v*u_y) - g*h_x + nu*laplaciano(u, dx) - lambd*u)
    v_new = v + dt * (-(u*v_x + v*v_y) - g*h_y + nu*laplaciano(v, dy) - lambd*v)

    # Continuidad (Conservación de masa)
    h_new = h + dt * (-(deriv(h*u, 0, dx) + deriv(h*v, 1, dy)))

    # Actualización instantánea
    h, u, v = h_new, u_new, v_new

    if n % 200 == 0:
        print(f"Paso {n}/{pasos} - Energía media: {np.mean(u**2 + v**2):.6f}")

fin = time.time()
print(f"\nSimulación terminada en {fin-inicio:.2f} segundos.")

# 4. VISUALIZACIÓN PROFESIONAL
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(h, cmap='viridis', extent=[0, Lx, 0, Ly])
plt.title(f"Elevación Gónica (t={pasos*dt}s)")
plt.colorbar(label="Altura (m)")

plt.subplot(1, 2, 2)
# Mostramos la magnitud de la velocidad
vel_mag = np.sqrt(u**2 + v**2)
plt.imshow(vel_mag, cmap='magma', extent=[0, Lx, 0, Ly])
plt.title("Campo de Velocidad (Laminar)")
plt.colorbar(label="Magnitud (m/s)")

plt.tight_layout()
plt.show()
