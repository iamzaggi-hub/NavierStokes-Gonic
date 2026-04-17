import numpy as np

# Cargar todos los ceros (suponiendo una columna)
g = np.loadtxt('zeros_100k.txt')  # array de 100k valores

# Mostrar primeros 10 para verificar formato
print("Primeros 10 ceros:", g[:10])

# Valores objetivo (γ ≈ 100 * θ_pf)
objetivos = [50, 182, 347, 546]

# Buscar índices más cercanos
for obj in objetivos:
    idx = np.argmin(np.abs(g - obj))
    gamma_cercano = g[idx]
    print(f"Objetivo γ≈{obj}: índice {idx+1} (1-based), valor γ={gamma_cercano:.6f}, γ/100={gamma_cercano/100:.6f}")

# Extraer primeros 1000 ceros
primeros_1000 = g[:1000]
np.savetxt('primeros_1000_ceros.txt', primeros_1000, fmt='%.6f')
print("\nPrimeros 1000 ceros guardados en 'primeros_1000_ceros.txt'")
