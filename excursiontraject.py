import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

ak = 0.004
k0 = 0.2
k1 = 0.222
n = 2
p = 5


def ComK(K, S, bk):
    return ak + ((bk * K ** n) / (k0 ** n + K ** n)) - (K / (1 + K + S))


def ComS(K, S, bs):
    return (bs / (1 + (K / k1) ** p)) - (S / (1 + K + S))


# Nouvelle fonction pour le système d'équations différentielles
def system(state, t, bk, bs):
    K, S = state
    return [ComK(K, S, bk), ComS(K, S, bs)]


def plot_nullclines(bk, bs, filename):
    K_vals = np.linspace(0, 0.5, 100)
    S_vals = np.linspace(0, 6, 100)
    K, S = np.meshgrid(K_vals, S_vals)

    dK_dt = ComK(K, S, bk)
    dS_dt = ComS(K, S, bs)

    intersection_indices = np.argwhere(np.isclose(dK_dt, 0, atol=1e-4) & np.isclose(dS_dt, 0, atol=1e-3))
    intersections = [(K[i, j], S[i, j]) for i, j in intersection_indices]

    plt.figure(figsize=(8, 6))
    plt.contour(K, S, dK_dt, levels=[0], colors='blue', linewidths=2)
    plt.contour(K, S, dS_dt, levels=[0], colors='green', linewidths=2)

    K_vecteur = np.linspace(0, 0.5, 10)
    S_vecteur = np.linspace(0, 6, 10)
    K_vec, S_vec = np.meshgrid(K_vecteur, S_vecteur)
    dK_dtvec = ComK(K_vec, S_vec, bk)
    dS_dtvec = ComS(K_vec, S_vec, bs)
    plt.quiver(K_vec, S_vec, dK_dtvec, dS_dtvec, scale=2, color='gray', scale_units='xy', angles='xy', width=0.004)

    # Ajout des trajectoires
    t = np.linspace(0, 1000, 10000)
    trajectoires_initiales = [
        [0.1, 4.0],
        [0.15, 4.5],
        [0.2, 3.5],
    ]

    # Tracer plusieurs trajectoires en rose
    for init_cond in trajectoires_initiales:
        solution = odeint(system, init_cond, t, args=(bk, bs))
        plt.plot(solution[:, 0], solution[:, 1], 'pink', alpha=0.6)

    # Trajectoire principale en violet
    solution = odeint(system, [0.15, 4.0], t, args=(bk, bs))
    plt.plot(solution[:, 0], solution[:, 1], 'purple', linewidth=2, alpha=0.8)

    for (k, s) in intersections:
        plt.plot(k, s, 'ro')

    plt.xlabel('ComK')
    plt.ylabel('ComS')
    plt.title('Nullclines pour bk = {}, bs = {}'.format(bk, bs))
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.show()


# Appel des fonctions avec les différents paramètres
plot_nullclines(bk=0.08, bs=0.68, filename='Fig_S3_etat_vegetative_monostable.png')
plot_nullclines(bk=0.12, bs=0.92, filename='Fig_S3_etat_competent_monostable.png')
plot_nullclines(bk=0.14, bs=0.68, filename='Fig_S3_etat_bistable.png')
plot_nullclines(bk=0.08, bs=0.80, filename='Fig_S3_etat_excitable.png')