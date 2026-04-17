import numpy as np

# Cargar ceros (suponiendo una columna)
g = np.loadtxt('zeros_100k.txt')  # array de 100k floats
# Índices aproximados (ajustar según búsqueda real)
indices = [10, 79, 260, 700]
theta_pf = [0.50, 1.82, 3.47, 5.46]  # de la tabla anterior
for idx, th in zip(indices, theta_pf):
    gamma = g[idx-1]  # porque Python indexa desde 0
    print(f"n={idx}, γ={gamma:.6f}, γ/100={gamma/100:.6f}, θ_pf={th:.4f}, error={abs(gamma/100 - th):.6f}")
