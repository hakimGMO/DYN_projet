import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

ak = 0.004
k0 = 0.2
k1 = 0.222
n = 2
p = 5

def ComK(K, S, bk):
    return ak + ((bk * K**n) / (k0**n + K**n)) - (K / (1 + K + S))

def ComS(K, S, bs):
    return (bs / (1 + (K/k1)**p)) - (S / (1 + K + S))

def equations(vars, bk, bs):
    k, s = vars
    return [ComK(k, s, bk), ComS(k, s, bs)]

def plot_nullclines(bk, bs, ax, title, initial_guesses):
    K_vals = np.linspace(0, 0.5, 100)  # Creer 100 valeur en tre 0 et 0.5
    S_vals = np.linspace(0, 6, 100)
    
    K, S = np.meshgrid(K_vals, S_vals)  # Il fait un tableau par valeur de K pour une valeur de S, il y a 100 meme tableau K avec 100 lignes de 0 à 0.45 et 100 tableau avec 100 lignes d'une valeur different pour chaque tableau S

    # Calcul des nullclines
    dK_dt = ComK(K, S, bk)
    dS_dt = ComS(K, S, bs)

    # intersection_indices = np.argwhere(np.isclose(dK_dt, 0, atol=1e-4) & np.isclose(dS_dt, 0, atol=1e-3))

    # intersections = [(K[i, j], S[i, j]) for i, j in intersection_indices]

    # Tracer les nullclines
    ax.contour(K, S, dK_dt, levels=[0], colors='blue', linewidths=2)
    ax.contour(K, S, dS_dt, levels=[0], colors='green', linewidths=2)

    # Tracer les vecteurs
    K_vecteur = np.linspace(0, 0.5, 10)
    S_vecteur = np.linspace(0, 6, 10)
    K_vec, S_vec = np.meshgrid(K_vecteur, S_vecteur)
    dK_dtvec = ComK(K_vec, S_vec, bk)
    dS_dtvec = ComS(K_vec, S_vec, bs)
    ax.quiver(K_vec, S_vec, dK_dtvec, dS_dtvec, scale=2, color='gray', scale_units='xy', angles='xy', width=0.004)

    # for (k, s) in intersections:
    #     ax.plot(k, s, 'ro')

    # Tracer les intersections basées sur les estimations initiales
    for guess in initial_guesses:
        intersection = fsolve(equations, guess, args=(bk, bs))
        ax.plot(intersection[0], intersection[1], 'ro')

    ax.set_xlabel('ComK')
    ax.set_ylabel('ComS')
    ax.set_title(title)
    ax.grid(True)

# Créer une figure avec une grille de sous-figures
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Estimations initiales pour chaque sous-figure
initial_guesses_A = [[0.01, 2.0]]
initial_guesses_B = [[0.2, 2.0]]
initial_guesses_C = [[0.01, 2.0], [0.1, 2.2], [0.16, 1.8]]
initial_guesses_D = [[0.029, 4.1], [0.08, 4.2], [0.12, 3.8]]

# Tracer les nullclines dans chaque sous-figure avec les titres personnalisés et les estimations initiales
plot_nullclines(bk=0.08, bs=0.68, ax=axs[0, 0], title='A: Vegetative monostable (bk = 0.08, bs = 0.68)', initial_guesses=initial_guesses_A)
plot_nullclines(bk=0.12, bs=0.92, ax=axs[0, 1], title='B: Competent monostable (bk = 0.12, bs = 0.92)', initial_guesses=initial_guesses_B)
plot_nullclines(bk=0.14, bs=0.68, ax=axs[1, 0], title='C: Bistable (bk = 0.14, bs = 0.68)', initial_guesses=initial_guesses_C)
plot_nullclines(bk=0.08, bs=0.8, ax=axs[1, 1], title='D: Excitable (bk = 0.08, bs = 0.8)', initial_guesses=initial_guesses_D)

# Ajuster les espaces entre les sous-figures
plt.tight_layout()

# Sauvegarder la figure
plt.savefig('Fig_S3_combined.png', dpi=300)

# Afficher la figure
plt.show()
