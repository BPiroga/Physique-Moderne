 import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# === Paramètres physiques et numériques ===
dt = 1E-7
dx = 0.001
nx = int(1 / dx) * 2
nt = 90000
nd = int(nt / 1000) + 1
n_frame = nd
s = dt / dx**2

# === Paramètres du paquet d'onde ===
xc = 0.6
sigma = 0.05
A = 1 / (math.sqrt(sigma * math.sqrt(math.pi)))

# === Paramètres du potentiel (puits) ===
v0 = -4000
x1, x2 = 0.8, 0.9  # bords du puits

# === Choix de l'énergie par l'utilisateur ===
try:
    e = float(input("Entrez la valeur de E/V0 (ex: 5) : "))
except ValueError:
    e = 5  # valeur par défaut
    print("Valeur invalide, E/V0=5 utilisé par défaut.")

E = e * v0
k = math.sqrt(2 * abs(E))

# === Grille d'espace ===
o = np.linspace(0, (nx - 1) * dx, nx)

# === Potentiel : puit de potentiel ===
V = np.zeros(nx)
V[(o >= x1) & (o <= x2)] = v0

# === Initialisation du paquet d'onde ===
cpt = A * np.exp(1j * k * o - ((o - xc) ** 2) / (2 * sigma**2))
re = np.real(cpt)
im = np.imag(cpt)
b = np.zeros(nx)

# === Densité de probabilité ===
densite = np.zeros((nt, nx))
densite[0, :] = np.abs(cpt)**2
final_densite = np.zeros((n_frame, nx))

# === Évolution temporelle (méthode de Crank-Nicolson simplifiée) ===
it = 0
for i in range(1, nt):
    if i % 2 != 0:
        b[1:-1] = im[1:-1]
        im[1:-1] += s * (re[2:] + re[:-2]) - 2 * re[1:-1] * (s + V[1:-1] * dt)
        densite[i, 1:-1] = re[1:-1]**2 + im[1:-1] * b[1:-1]
    else:
        re[1:-1] -= s * (im[2:] + im[:-2]) - 2 * im[1:-1] * (s + V[1:-1] * dt)

    if (i - 1) % 1000 == 0:
        final_densite[it][:] = densite[i][:]
        it += 1

# === Animation ===
def init():
    line.set_data([], [])
    return line,

def animate(j):
    line.set_data(o, final_densite[j, :])
    return line,

fig = plt.figure()
line, = plt.plot([], [])
plt.ylim(0, 1.2 * np.max(final_densite))
plt.xlim(0, 2)
plt.plot(o, (V * 0.1) / abs(v0), label="Potentiel (échelle x0.1)")  # Potentiel redimensionné
plt.title(f"Puits de potentiel avec E/V0 = {e:.2f}")
plt.xlabel("x")
plt.ylabel("Densité de probabilité")
plt.legend()

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=nd, blit=False, interval=100, repeat=False)
plt.show()
