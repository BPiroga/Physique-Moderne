import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ----- PARAMÈTRES -----
dt = 1E-7
dx = 0.001
nx = int(2 / dx)
nt = 90000
nd = int(nt / 1000) + 1  # Nombre d'images pour l'animation
s = dt / dx**2

# ----- PARAMÈTRES INITIAUX -----
xc = 0.6       # Centre du paquet d'onde
sigma = 0.05   # Largeur du paquet

# ----- INPUT UTILISATEUR -----
e = 5  # Modifier ici si besoin
v0 = -4000
E = e * v0
k = np.sqrt(2 * abs(E))

# ----- ESPACE ET POTENTIEL GAUSSIEN -----
x = np.linspace(0, 2, nx)
x0 = 1.0   # Centre du puits
a = 0.05   # Largeur du puits
V = v0 * np.exp(-((x - x0)**2) / (2 * a**2))  # Potentiel gaussien

# ----- CONDITION INITIALE : PAQUET D’ONDE -----
A = 1 / (np.sqrt(sigma * np.sqrt(np.pi)))
psi = A * np.exp(1j * k * x - ((x - xc) ** 2) / (2 * sigma**2))

# ----- INITIALISATION -----
densite = np.zeros((nt, nx))
final_densite = np.zeros((nd, nx))
re = np.real(psi)
im = np.imag(psi)
b = np.zeros(nx)
densite[0, :] = np.abs(psi) ** 2

# ----- PROPAGATION -----
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
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
V_plot = (V / abs(v0)) * np.max(densite)  # Échelle du puits visible
ax.plot(x, V_plot, label="Potentiel (échelle ajustée)", color="black", linestyle="--")
ax.set_xlim(0, 2)
ax.set_ylim(-np.max(densite), np.max(densite) * 1.2)
ax.set_xlabel("x")
ax.set_ylabel("Densité de probabilité")
ax.set_title(f"Propagation du paquet d'ondes dans un potentiel gaussien (E/V0 = {e})")
ax.legend()

def init():
    line.set_data([], [])
    return line,

def animate(j):
    ydata = final_densite[j, :]
    line.set_data(x, ydata)
    return line,

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=nd, interval=100, blit=False)
plt.show()
