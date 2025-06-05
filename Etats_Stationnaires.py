import numpy as np
import matplotlib.pyplot as plt

# Paramètres de discrétisation
dx = 0.001
x = np.arange(0, 1, dx)
nx = len(x)

# Potentiel : puits carré centré
V0 = 4000 # Profondeur du puits
V = np.zeros(nx)
V[(x > 0.4) & (x < 0.6)] = -V0

# Fonction pour résoudre l'équation de Schrödinger pour une énergie donnée
def solve_schrodinger(E):
    psi = np.zeros(nx)
    psi[1] = 1e-5  # Condition initiale faible pente
    for i in range(1, nx - 1):
        psi[i+1] = 2*psi[i] - psi[i-1] - 2*dx**2 * (E - V[i]) * psi[i]
    return psi

# Critère pour trouver une énergie propre : la valeur de ψ à la frontière doit être proche de 0
def boundary_value(E):
    psi = solve_schrodinger(E)
    return psi[-1]

# Recherche de racines via balayage d'énergie
E_values = np.linspace(-V0, 0, 1000)
boundary = [boundary_value(E) for E in E_values]

# Recherche des changements de signe (indique un zéro potentiel)
eigvals = []
for i in range(len(boundary)-1):
    if boundary[i] * boundary[i+1] < 0:
        eigvals.append((E_values[i] + E_values[i+1]) / 2)

# Affichage des états stationnaires
plt.figure(figsize=(10,5))
for E in eigvals:
    psi = solve_schrodinger(E)
    psi /= np.max(np.abs(psi))  # Normalisation arbitraire
    plt.plot(x, psi + E, label=f"E ≈ {E:.2f}")  # Affiche ψ(x) + E
plt.plot(x, V, 'k--', label="Potentiel")
plt.title("États stationnaires dans un puits fini")
plt.xlabel("x")
plt.ylabel("ψ(x) + E")
plt.legend()
plt.grid()
plt.show()
