import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constantes en unités naturelles (ħ = 1, 2m = 1)
V0 = 4000     # Profondeur du puits
a = 0.05      # Largeur du potentiel
x0 = 0.5      # Centre du potentiel

# Potentiel gaussien
def V(x):
    return -V0 * np.exp(-((x - x0) ** 2) / a**2)

# Équation de Schrödinger
def schrodinger(x, y, E):
    psi, phi = y
    dpsi_dx = phi
    dphi_dx = 2 * (V(x) - E) * psi
    return [dpsi_dx, dphi_dx]

# Résolution pour une énergie donnée
def solve_stationary(E, x_domain):
    y0 = [0, 1e-5]
    sol = solve_ivp(schrodinger, [x_domain[0], x_domain[-1]], y0, t_eval=x_domain, args=(E,), rtol=1e-6)
    return sol.t, sol.y[0]

# Domaine spatial
x = np.linspace(0, 1, 1000)

# Energies testées (niveaux liés)
E_values = [-3900, -3500, -3000, -2500, -2000]

# Tracé
plt.figure(figsize=(10, 6))
for E in E_values:
    x_vals, psi_vals = solve_stationary(E, x)
    psi_vals /= np.max(np.abs(psi_vals))
    plt.plot(x_vals, psi_vals + E, label=f"E = {E}")

# Potentiel redimensionné pour affichage
V_plot = V(x)
V_plot_scaled = V_plot / np.abs(V0) * np.abs(min(E_values))
plt.plot(x, V_plot_scaled, 'k--', label='Potentiel (échelle adaptée)')

plt.title("États stationnaires dans un puits gaussien")
plt.xlabel("x")
plt.ylabel("ψ(x) + E")
plt.legend()
plt.grid()
plt.show()
