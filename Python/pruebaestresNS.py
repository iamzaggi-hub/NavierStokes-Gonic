# Script: tsunami_gonic_validation.py
# Versión incluida en el Apéndice A del informe
# Ejecutar en Python 3.10 con NumPy, Matplotlib, imageio
# Optimizado para memoria reducida (float32, arrays in-place)

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import os
import sys

np.random.seed(42)

# Parámetros físicos y numéricos
g = 9.81
A_gonico = 1.0/9.0
A_SI = 0.0
nu = 0.0  # viscosidad numérica (opcional)

# Dominio y malla (inicial)
Nx0, Ny0 = 256, 256
Lx, Ly = 2_000_000.0, 2_000_000.0  # 2000 km en metros
amp = 2.0  # amplitud inicial (m)
sigma = 50_000.0  # 50 km
Tfinal = 7200.0  # 2 horas en segundos
save_interval = 600.0  # guardar cada 600 s
CFL = 0.4
dt_max = 10.0

# Funciones auxiliares
def estimate_memory_bytes(Nx, Ny):
    # estimación simple: 3 campos (h,u,v) float32
    bytes_per = 4
    return 3 * Nx * Ny * bytes_per

def run_simulation(Nx, Ny, A_value):
    dx = Lx / Nx
    dy = Ly / Ny
    x = (np.arange(Nx) + 0.5) * dx - Lx/2
    y = (np.arange(Ny) + 0.5) * dy - Ly/2
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Inicialización
    h = np.ones((Nx, Ny), dtype=np.float32)
    h += amp * np.exp(-((X)**2 + (Y)**2) / (2.0 * sigma**2)).astype(np.float32)
    u = np.zeros((Nx, Ny), dtype=np.float32)
    v = np.zeros((Nx, Ny), dtype=np.float32)

    # Cálculo dt por CFL
    h_max = float(h.max())
    c = np.sqrt(g * h_max)
    dt = min(dt_max, CFL * min(dx, dy) / (c + 1e-6))

    # Cap de pasos
    nsteps = int(np.ceil(Tfinal / dt))
    save_every = max(1, int(np.ceil(save_interval / dt)))

    # Sponge layer (10% del dominio)
    sponge_width_x = int(0.1 * Nx)
    sponge_width_y = int(0.1 * Ny)
    sponge = np.ones((Nx, Ny), dtype=np.float32)
    for i in range(Nx):
        wx = 0.0
        if i < sponge_width_x:
            wx = (sponge_width_x - i) / sponge_width_x
        elif i >= Nx - sponge_width_x:
            wx = (i - (Nx - sponge_width_x - 1)) / sponge_width_x
        for j in range(Ny):
            wy = 0.0
            if j < sponge_width_y:
                wy = (sponge_width_y - j) / sponge_width_y
            elif j >= Ny - sponge_width_y:
                wy = (j - (Ny - sponge_width_y - 1)) / sponge_width_y
            alpha = max(wx, wy)
            if alpha > 0:
                sponge[i,j] = np.exp(- (alpha**2))

    # almacenamiento de energía y tiempos
    times = []
    energy = []
    snapshots = []

    # operadores discretos (centred differences)
    def ddx(f):
        return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dx)
    def ddy(f):
        return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dy)
    def lap(f):
        return (np.roll(f, -1, axis=0) - 2.0*f + np.roll(f, 1, axis=0)) / dx**2 + \
               (np.roll(f, -1, axis=1) - 2.0*f + np.roll(f, 1, axis=1)) / dy**2

    t = 0.0
    step = 0
    start_time = time.time()
    try:
        while t < Tfinal - 1e-9 and step < nsteps:
            # guardar
            if step % save_every == 0:
                times.append(t)
                ke = 0.5 * np.mean(u**2 + v**2)
                energy.append(float(ke))
                snapshots.append(h.copy())

            # RK2 (Heun)
            # paso 1
            h_x = ddx(h * u)
            h_y = ddy(h * v)
            hu = u.copy(); hv = v.copy()

            dh1 = - (h_x + h_y)
            du1 = - g * ddx(h) + nu * lap(u) - A_value * u
            dv1 = - g * ddy(h) + nu * lap(v) - A_value * v

            h1 = h + dt * dh1
            u1 = u + dt * du1
            v1 = v + dt * dv1

            # paso 2
            h_x2 = ddx(h1 * u1)
            h_y2 = ddy(h1 * v1)
            dh2 = - (h_x2 + h_y2)
            du2 = - g * ddx(h1) + nu * lap(u1) - A_value * u1
            dv2 = - g * ddy(h1) + nu * lap(v1) - A_value * v1

            h += 0.5 * dt * (dh1 + dh2)
            u += 0.5 * dt * (du1 + du2)
            v += 0.5 * dt * (dv1 + dv2)

            # aplicar sponge
            h = 1.0 + (h - 1.0) * sponge
            u *= sponge
            v *= sponge

            t += dt
            step += 1

        end_time = time.time()
        exec_time = end_time - start_time

        # guardar último estado
        times.append(t)
        energy.append(float(0.5 * np.mean(u**2 + v**2)))
        snapshots.append(h.copy())

        return {
            'success': True,
            'times': np.array(times),
            'energy': np.array(energy),
            'snapshots': snapshots,
            'h_final': h,
            'exec_time': exec_time,
            'Nx': Nx,
            'Ny': Ny,
            'dt': dt,
            'nsteps': step
        }

    except MemoryError as me:
        return {'success': False, 'error': 'MemoryError', 'message': str(me)}
    except Exception as e:
        return {'success': False, 'error': 'Exception', 'message': str(e)}

# flujo principal: intentar con Nx0,Ny0 y reducir si falla
Nx, Ny = Nx0, Ny0
mem_est = estimate_memory_bytes(Nx, Ny)
report_lines = []
report_lines.append(f"Parametros: Nx={Nx}, Ny={Ny}, Lx={Lx}, Ly={Ly}, amp={amp}, sigma={sigma}, Tfinal={Tfinal}")
report_lines.append(f"Estimacion memoria arrays (bytes): {mem_est}")

results = None
for attempt in range(2):
    try:
        # ejecutar SI (A=0)
        res_SI = run_simulation(Nx, Ny, A_SI)
        if not res_SI['success']:
            raise RuntimeError(res_SI.get('message','unknown'))

        # ejecutar Gónico (A=1/9)
        res_G = run_simulation(Nx, Ny, A_gonico)
        if not res_G['success']:
            raise RuntimeError(res_G.get('message','unknown'))

        results = (res_SI, res_G)
        break
    except RuntimeError as e:
        report_lines.append(f"Intento {attempt+1} fallo: {str(e)}")
        if attempt == 0:
            report_lines.append("Reduciendo resolucion a 128x128 y reintentando...")
            Nx, Ny = Nx//2, Ny//2
            mem_est = estimate_memory_bytes(Nx, Ny)
            report_lines.append(f"Nueva estimacion memoria arrays (bytes): {mem_est}")
            continue
        else:
            report_lines.append("Fallo irreparable en ejecucion.")
            break

# si no hay resultados, escribir informe y salir
if results is None:
    report_lines.append("No se obtuvieron resultados.")
    with open('informe_gonico.txt','w') as f:
        f.write('\n'.join(report_lines))
    print('\n'.join(report_lines))
    sys.exit(1)

res_SI, res_G = results

# generar figura comparativa
plt.figure(figsize=(15,5))
# panel 1: energia
plt.subplot(1,3,1)
plt.plot(res_SI['times']/60.0, res_SI['energy'], label='SI (A=0)', color='red')
plt.plot(res_G['times']/60.0, res_G['energy'], label='Gónico (A=1/9)', color='blue')
plt.xlabel('Tiempo (min)')
plt.ylabel('Energía cinética media')
plt.title('Energía vs Tiempo')
plt.legend()

# panel 2: SI final
plt.subplot(1,3,2)
plt.imshow(res_SI['h_final'].T, origin='lower', cmap='viridis', extent=[-Lx/2,Lx/2,-Ly/2,Ly/2])
plt.title('SI Final (h)')
plt.colorbar(fraction=0.046, pad=0.04)

# panel 3: Gónico final
plt.subplot(1,3,3)
plt.imshow(res_G['h_final'].T, origin='lower', cmap='viridis', extent=[-Lx/2,Lx/2,-Ly/2,Ly/2])
plt.title('Gónico Final (h)')
plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('comparacion_gonico_vs_SI.png', dpi=150)

# informe
def summarize(res, label):
    e0 = float(res['energy'][0])
    epeak = float(np.max(res['energy']))
    efinal = float(res['energy'][-1])
    diss = 100.0 * (epeak - efinal) / (epeak + 1e-12)
    return f"{label}: energia_inicial={e0:.6e}, energia_pico={epeak:.6e}, energia_final={efinal:.6e}, disipacion%={diss:.2f}, dt={res['dt']:.4f}, pasos={res['nsteps']}, tiempo_ejec={res['exec_time']:.2f}s"

report_lines.append('--- Resultados ---')
report_lines.append(summarize(res_SI, 'SI (A=0)'))
report_lines.append(summarize(res_G, 'Gónico (A=1/9)'))
report_lines.append(f"Archivo imagen: comparacion_gonico_vs_SI.png")
report_lines.append(f"Versiones: numpy {np.__version__}, matplotlib {matplotlib.__version__}")

with open('informe_gonico.txt','w') as f:
    f.write('\n'.join(report_lines))

# imprimir resumen
print('\n'.join(report_lines))

