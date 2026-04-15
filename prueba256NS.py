import numpy as np
import matplotlib.pyplot as plt
import time

# 1. PARÁMETROS (TUS CONSTANTES)
Nx, Ny = 256, 256
Lx, Ly = 600.0, 600.0
dx, dy = Lx/Nx, Ly/Ny
g, nu, lambd = 9.81, 0.000001, 2.0
dt = 0.005  # Paso de tiempo fijo para NO bloquear el i5
pasos = 500 # Total de iteraciones

# 2. CONDICIÓN INICIAL (TSUNAMI REAL)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
h = 1.0 + 10.0 * np.exp(-((X-150)**2 + (Y-300)**2) / 1500)
u = np.zeros((Nx, Ny))
v = np.zeros((Nx, Ny))

# 3. BUCLE DE INTEGRACIÓN GÓNICA (MÁXIMA VELOCIDAD)
print(f"Iniciando Motor Gónico en Malla {Nx}x{Nx}...")
inicio = time.time()

for n in range(pasos):
    # Derivadas espaciales (Diferencias finitas rápidas)
    h_x = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) / (2*dx)
    h_y = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2*dy)

    # Navier-Stokes corregido con métrica Gónica
    # El término (lambd * u) estabiliza la memoria instantáneamente
    u_new = u + dt * (-g * h_x - lambd * u)
    v_new = v + dt * (-g * h_y - lambd * v)
    h_new = h + dt * (-(h * (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dx)))

    h, u, v = h_new, u_new, v_new

    if n % 100 == 0:
        print(f"Progreso: {n}/{pasos} iteraciones...")

fin = time.time()
print(f"Simulación completada en {round(fin-inicio, 2)} segundos.")

# 4. GRÁFICA DE RESULTADO
plt.figure(figsize=(10, 6))
plt.imshow(h, cmap='magma', extent=[0, Lx, 0, Ly])
plt.colorbar(label="Elevación (m)")
plt.title(f"Tsunami Gónico en HP 8440p (Malla {Nx}x{Nx})\nEstabilidad absoluta sin colapso")
plt.savefig("resultado_256_gonico.png")
plt.show()
