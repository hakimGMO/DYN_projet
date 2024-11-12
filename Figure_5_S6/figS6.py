# On reprend la figure 4 en changeant quelques point dont la def de ComK 

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Paramètres du système
ak = 0.004
k0 = 0.2
k1 = 0.222
n = 2
p = 5
gamma_k = 0.96
kg = 0.2
bg = 0.2
gamma_s = 0.1
dt = 0.1
temps_max = 200

# Fonction pour calculer les dérivées
def ComK(K, S, bk, gamma_k):
    return ak + ((bk * K ** n) / (k0 ** n + K ** n)) - (gamma_k * K / (1 + K + S))

def ComS(K, S, bs):
    return (bs / (1 + (K / k1) ** p)) + ((bg * K ** n) / (kg ** n + K ** n)) - (S / (1 + K + S))

def equations(vars): # Sert pour affichage des intersections
    k, s = vars
    return [ComK(k, s, BK,gamma_k), ComS(k, s, BS)]

# Estimations initiales pour trouver les intersections
initial_guesses = [
    [0.035,4.4],
    [0.045, 5.3],
    [0.15, 4]
]


def simulate_system(bk, bs, k0, s0, t_end, dt):
    t = np.arange(0, t_end, dt)
    n = len(t)
    k = np.zeros(n)
    s = np.zeros(n)
    k[0] = k0
    s[0] = s0
    noise_amplitude = 0.01

    for i in range(1, n):
        dk_dt = ComK(k[i-1], s[i-1], bk, gamma_k)
        ds_dt = ComS(k[i-1], s[i-1], bs)
        noise_s = np.random.normal(0, noise_amplitude)
        k[i] = k[i-1] + dk_dt * dt
        s[i] = s[i-1] + ds_dt * dt + noise_s * np.sqrt(dt)

    return k, s, t


# Paramètres de la simulation
BK = 0.08
BS = 0.80
K0 = 0.05
S0 = 4.5
T_END = 400
DT = 0.2

def plot_nullclines_and_vector_field():
    k_vals = np.linspace(0, 0.45, 1000)
    s_vals = np.linspace(0, 8, 1000)
    k_grid, s_grid = np.meshgrid(k_vals, s_vals)
    dk_dt = ComK(k_grid, s_grid, BK, gamma_k)
    ds_dt = ComS(k_grid, s_grid, BS)

    plt.figure(figsize=(8, 6))
    plt.contour(k_grid, s_grid, dk_dt, levels=[0], colors='blue', linewidths=2)
    plt.contour(k_grid, s_grid, ds_dt, levels=[0], colors='green', linewidths=2)

    k_vector = np.linspace(0, 0.45, 10)
    s_vector = np.linspace(0, 8, 10)
    k_vec, s_vec = np.meshgrid(k_vector, s_vector)
    dk_dt_vec = ComK(k_vec, s_vec, BK, gamma_k)
    ds_dt_vec = ComS(k_vec, s_vec, BS)
    plt.quiver(k_vec, s_vec, dk_dt_vec, ds_dt_vec, scale=2, color='gray', scale_units='xy', angles='xy', width=0.004)

    # Calcule quand ComK et ComS sont egaux dans les environs des trois positions donner (guess)
    intersection_list = []
    for guess in initial_guesses:
        intersection = fsolve(equations, guess)
        intersection_list.append(intersection)
    
    
    for intersection in intersection_list:
        plt.plot(intersection[0] , intersection[1], 'ro') # Affiche les points en rouge
    
    
    
    # Simuler le système
    k, s, t = simulate_system( BK, BS, K0, S0, T_END, DT)
    plt.plot(k, s, color='purple', linewidth=1)

    plt.xlabel('[ComK] (arb. units)')
    plt.ylabel('[ComS] (arb. units)')

    plt.savefig('FigS6.png', dpi=300)
    plt.show()
plot_nullclines_and_vector_field()
