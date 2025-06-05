import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import bisect

# ---- CONSTANTES ----
hbar = 1
m = 1

# ---- DOMAINE ----
L = 2.0
nx = 1000
x = np.linspace(0, L, nx)

# ---- POTENTIEL : puits entre a et b ----
V0 = -50
a, b = 0.8, 1.2
V = np.zeros_like(x)
V[(x >= a) & (x <= b)] = V0

# ---- ÉQUATION DE SCHRÖDINGER STATIONNAIRE ----
def schrodinger(x, y, E):
    """ y = [psi, psi'] """
    psi, phi = y
    dpsi = phi
    dphi = 2 * m / hbar**2 * (V_func(x) - E) * psi
    return [dpsi, dphi]

def V_func(xi):
    """ Potentiel V(x) """
    return V0 if a <= xi <= b else 0

def solve_for_E(E, show=False):
    """ Résout l'équation de Schrödinger pour une énergie E et renvoie psi(L) """
    sol = solve_ivp(lambda x, y: schrodinger(x, y, E), [0, L], [0, 1e-5], t_eval=x)
    if show:
        plt.plot(sol.t, sol.y[0], label=f"E = {E:.2f}")
    return sol.y[0, -1]  # psi à la frontière droite

# ---- CHERCHER LES ÉTATS LIES ----
def find_bound_state(E1, E2):
    """ Recherche une valeur de E pour laquelle psi(L) = 0 entre E1 et E2 """
    E_root = bisect(lambda E: solve_for_E(E), E1, E2, xtol=1e-6)
    print(f"État lié trouvé à E = {E_root:.4f}")
    return E_root

# ---- SCAN GROSSIER DES ÉNERGIES POUR TROUVER INTERVALLES SIGNIFICATIFS ----
energies = np.linspace(V0, 0, 300)
sign_changes = []
prev_val = solve_for_E(energies[0])
for i in range(1, len(energies)):
    curr_val = solve_for_E(energies[i])
    if prev_val * curr_val < 0:
        sign_changes.append((energies[i-1], energies[i]))
    prev_val = curr_val

# ---- DESSIN DES ÉTATS LIES ----
plt.figure(figsize=(10,6))
for (E1, E2) in sign_changes:
    E_bound = find_bound_state(E1, E2)
    solve_for_E(E_bound, show=True)

plt.plot(x, V, 'k--', label="Potentiel V(x)")
plt.title("États stationnaires dans un puits de potentiel")
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.legend()
plt.grid()
plt.show()
