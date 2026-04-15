import numpy as np
import time

def test_fluid():
    nx = 5000  # Puntos de la malla
    u = np.ones(nx)
    u[int(.5/0.01):int(1/0.01+1)] = 2  # Condición inicial
    
    start = time.time()
    for n in range(1000): # 1000 iteraciones
        un = u.copy()
        for i in range(1, nx):
            u[i] = un[i] - un[i] * 0.01 / 0.02 * (un[i] - un[i-1])
    print(f"Tiempo de resolución: {time.time() - start:.4f} segundos")

test_fluid()
