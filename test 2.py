 import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ----- PARAMÈTRES -----
dt = 1E-7
dx = 0.001
nx = int(2 / dx)
nt = 90000
nd = int(nt / 1000) + 1  # Nombre d'images pour l'animation
s = dt / dx**2

# ----- INPUT UTILISATEUR -----
e = float(input("Entrez la valeur de E/V0 (ex: 5) : "))
v0 = -4000  # Profondeur du puits (valeur négative)
E = e * v0
k = math.sqrt(2 * abs(E))

# ----- ESPACE ET POTENTIEL -----
o = np.linspace(0, (nx - 1) * dx, nx)
V = np.zeros(nx)
V[(o >= 0.8) & (o <= 0.9)] = v0  # Puits de potentiel carré de largeur 0.1

# ----- CONDITION INITIALE : PAQUET D’ONDE -----
xc = 0.6       # Centre du paquet
sigma = 0.05   # Largeur du paquet
A = 1 / (math.sqrt(sigma * math.sqrt(math.pi)))
cpt = A * np.exp(1j * k * o - ((o - xc) ** 2) / (2 * sigma**2))

# ----- INITIALISATION -----
densite = np.zeros((nt, nx))
final_densite = np.zeros((nd, nx))
re = np.real(cpt)
im = np.imag(cpt)
b = np.zeros(nx)
densite[0, :] = np.abs(cpt) ** 2

# ----- PROPAGATION DE L'ÉQUATION DE SCHRÖDINGER -----
it = 0
for i in range(1, nt):
    if i % 2 != 0:
        b[1:-1] = im[1:-1]
        im[1:-1] += s * (re[2:] + re[:-2]) - 2 * re[1:-1] * (s + V[1:-1] * dt)
        densite[i, 1:-1] = re[1:-1]**2 + im[1:-1] * b[1:-1]
    else:
        re[1:-1] -= s * (im[2:] + im[:-2]) - 2 * im[1:-1] * (s + V[1:-1] * dt)

    if (i - 1) % 1000 == 0:
        final_densite[it, :] = densite[i, :]
        it += 1

# ----- ANIMATION -----
def init():
    line.set_data([], [])
    return line,

def animate(j):
    ydata = final_densite[j, :]
    line.set_data(o, ydata)
    return line,

# ----- FIGURE -----
fig = plt.figure()
line, = plt.plot([], [], lw=2, label="Densité de probabilité")
plt.plot(o, V, label="Puits de potentiel", color="black", linestyle="--")

plt.ylim(v0 * 1.2, np.max(densite) * 1.2)  # Affiche aussi le puits en profondeur
plt.xlim(0, 2)
plt.xlabel("x")
plt.ylabel("Énergie & Densité")
plt.title(f"Propagation du paquet d'ondes avec E/V0 = {e}")
plt.legend()

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=nd, interval=100, blit=False)
plt.show()

