import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import time

# 1. CONFIGURACIÓN DEL ENTORNO
Nx, Ny = 256, 256
Lx, Ly = 600.0, 600.0
dx, dy = Lx/Nx, Ly/Ny

# Parámetros Gónicos
g, nu, lambd = 9.81, 0.01, 1/9
dt = 0.005
pasos = 400  # Ajusta para una animación más larga

# Inicialización de matrices
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
h = 1.0 + 5.0 * np.exp(-((X-Lx/2)**2 + (Y-Ly/2)**2) / 1000)
u = np.zeros((Nx, Ny))
v = np.zeros((Nx, Ny))

# 2. FUNCIONES DE CÁLCULO VECTORIZADO
def deriv(f, axis):
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2*(dx if axis==0 else dy))

def laplaciano(f):
    return (np.roll(f, -1, axis=0) + np.roll(f, 1, axis=0) +
            np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1) - 4*f) / (dx**2)

# 3. CONFIGURACIÓN DEL WRITER (Requiere ffmpeg instalado)
metadata = dict(title='Tsunami Gónico', artist='ZAGGI_Model')
writer = FFMpegWriter(fps=20, metadata=metadata)

fig, ax = plt.subplots(figsize=(8, 6))
img = ax.imshow(h, cmap='magma', extent=[0, Lx, 0, Ly], animated=True)
plt.colorbar(img, label="Elevación (m)")
ax.set_title("Evolución del Tsunami - Modelo Gónico")

# 4. BUCLE DE SIMULACIÓN Y GRABACIÓN
print("Iniciando grabación del motor gónico...")
inicio = time.time()

with writer.saving(fig, "tsunami_gonico.mp4", dpi=100):
    for n in range(pasos):
        # Navier-Stokes Gónico
        h_x, h_y = deriv(h, 0), deriv(h, 1)
        u_x, u_y = deriv(u, 0), deriv(u, 1)
        v_x, v_y = deriv(v, 0), deriv(v, 1)

        u_new = u + dt * (-(u*u_x + v*u_y) - g*h_x + nu*laplaciano(u) - lambd*u)
        v_new = v + dt * (-(u*v_x + v*v_y) - g*h_y + nu*laplaciano(v) - lambd*v)
        h_new = h + dt * (-(deriv(h*u, 0) + deriv(h*v, 1)))

        h, u, v = h_new, u_new, v_new

        # Actualizar cuadro cada 5 pasos para velocidad de renderizado
        if n % 5 == 0:
            img.set_array(h)
            writer.grab_frame()
            if n % 100 == 0:
                print(f"Progreso: {n}/{pasos} cuadros procesados...")

fin = time.time()
print(f"¡Listo! Archivo 'tsunami_gonico.mp4' creado en {fin-inicio:.2f}s.")
