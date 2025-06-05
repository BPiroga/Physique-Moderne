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

# ---- POTENTIEL RECTANGULAIRE ----
V0 = -4000  # <-- profondeur du puits
a, b = 0.8, 1.2  # position du puits
V = np.zeros_like(x)
V[(x >= a) & (x <= b)] = V0

# ---- FONCTION POTENTIEL ----
def V_func(xi):
    return V0 if a <= xi <= b else 0

# ---- SCHRODINGER STATIONNAIRE ----
def schrodinger(x, y, E):
    psi, phi = y
    dpsi = phi
    dphi = 2 * m / hbar**2 * (V_func(x) - E) * psi
    return [dpsi, dphi]

# ---- RESOLUTION POUR UNE ENERGIE ----
def solve_for_E(E):
    sol = solve_ivp(lambda x, y: schrodinger(x, y, E), [0, L], [0, 1e-5], t_eval=x)
    return sol.y[0], sol.y[0, -1]  # psi(x), psi(L)

# ---- CHERCHE CHANGEMENTS DE SIGNE ----
def find_sign_changes(energies):
    psiL_prev = solve_for_E(energies[0])[1]
    intervals = []
    for i in range(1, len(energies)):
        psiL = solve_for_E(energies[i])[1]
        if psiL * psiL_prev < 0:
            intervals.append((energies[i-1], energies[i]))
        psiL_prev = psiL
    return intervals

# ---- TROUVE UN ETAT LIE ENTRE E1 ET E2 ----
def find_bound_state(E1, E2):
    E_root = bisect(lambda E: solve_for_E(E)[1], E1, E2, xtol=1e-6)
    psi, _ = solve_for_E(E_root)
    return E_root, psi

# ---- TROUVER ET TRACER LES ÉTATS ----
# On balaye les énergies de -4000 à 0 pour capturer tous les états liés
energies = np.linspace(V0, 0, 1000)
intervals = find_sign_changes(energies)

plt.figure(figsize=(10, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(intervals)))
states = []

for n, (E1, E2) in enumerate(intervals):
    E_n, psi_n = find_bound_state(E1, E2)
    
    # Normaliser
    norm = np.trapz(psi_n**2, x)
    psi_n /= np.sqrt(norm)
    
    # Affichage : placer |ψ|² * facteur + E_n
    scale = 5
    psi_plot = scale * psi_n**2 + E_n
    
    plt.plot(x, psi_plot, label=f"État n={n+1} (E = {E_n:.2f})", color=colors[n])
    states.append((E_n, psi_n))

# ---- Afficher le potentiel ----
plt.plot(x, V, 'k--', label="Potentiel V(x)", linewidth=1.5)

# ---- Mise en forme ----
plt.title("États stationnaires")
plt.xlabel("x")
plt.ylabel("Énergie / Densité de probabilité")
plt.legend()
plt.grid()
plt.ylim(V0 - 500, 50)  # Vue adaptée au puits très profond
plt.tight_layout()
plt.show()
