import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('energia_enstrofia_G3D_100s.csv', delimiter=',', skiprows=1)
t = data[:,0]
E = data[:,1]
enst = data[:,2]

# Evitar división por cero
enst = np.maximum(enst, 1e-15)
Q = np.sqrt(2*E) / np.sqrt(enst)

plt.figure(figsize=(10,6))
plt.plot(t, Q, 'b-', linewidth=1.5)
plt.axhline(2*np.sqrt(2), color='r', linestyle='--', label=r'$2\sqrt{2} \approx 2.828$')
plt.xlabel('Tiempo (s)')
plt.ylabel('Q(t) = sqrt(2E) / sqrt(enst)')
plt.title('Evolución de Q(t) en simulación G3D (λ=2/9)')
plt.legend()
plt.grid(True)
plt.savefig('Q_t_G3D_100s.png')
plt.show()

print(f"Q inicial (t≈{t[0]:.2f} s) = {Q[0]:.6f}")
print(f"Q final   (t≈{t[-1]:.2f} s) = {Q[-1]:.6f}")
print(f"Diferencia respecto a 2√2: {abs(Q[-1] - 2*np.sqrt(2)):.6f}")
