import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Paramètres du système
ak = 0.004
k0 = 0.2
k1 = 0.222
n = 2
p = 5

def ComK(K, S, bk):
    return ak + ((bk * (K)**n) / (k0**n + (K)**n)) - (K / (1 + K + S))

def ComS(K, S, bs):
    return (bs / (1 + ((K)/k1)**p)) - (S / (1 + K + S))

def simulate_system(bk, bs, k0, s0, t_end, dt):
    t = np.arange(0, t_end, dt)
    n = len(t)
    k = np.zeros(n)
    s = np.zeros(n)
    k[0] = k0
    s[0] = s0
    noise_amplitude = 0.5 # Variable de bruit

    for i in range(1, n):
        dk_dt = ComK(k[i-1], s[i-1], bk)
        ds_dt = ComS(k[i-1], s[i-1], bs)
        noise_s = np.random.normal(0, noise_amplitude) # Bruit 
        k[i] = k[i-1] + dk_dt * dt
        s[i] = s[i-1] + ds_dt * dt + noise_s * np.sqrt(dt) # Bruit de Wiener

    return k, s, t

def equations(vars): # Sert pour affichage des intersections
    k, s = vars
    return [ComK(k, s, BK), ComS(k, s, BS)]

# Estimations initiales pour trouver les intersections
initial_guesses = [
    [0.029, 4.1],
    [0.08, 4.2],
    [0.12, 3.8]
]

def plot_nullclines_and_trajectory(k, s, t_end, dt, mult_K): # Affichage de la figure 4a, avec nullcline, champs de vecteur et simulation
    
    k_vals = np.linspace(0, 0.45*mult_K, 1000) # Prend 1000 valeur entre 0 et 4.5
    s_vals = np.linspace(0, 8, 1000)
    
    k_grid, s_grid = np.meshgrid(k_vals / mult_K, s_vals) # Diviser par mult_K pour que le calcul soit fait dans les bonnes unités (4.5 --> 0.45)
    
    dk_dt = ComK(k_grid, s_grid, BK) # Tableau numpy de l'ensemble des donnes de dk et ds par rapport a dt
    ds_dt = ComS(k_grid, s_grid, BS)

    plt.figure(figsize=(8, 6)) # Plot la figure 
    plt.contour(k_grid * mult_K, s_grid, dk_dt, levels=[0], colors='blue', linewidths=2)
    plt.contour(k_grid * mult_K, s_grid, ds_dt, levels=[0], colors='green', linewidths=2)

    # Champs de vecteur 
    k_vector = np.linspace(0, 0.45*mult_K, 10) # Cette fois ci que 10 pour ne pas trop avoir de fleche pour un plus belle visibilité 
    s_vector = np.linspace(0, 8, 10)
    k_vec, s_vec = np.meshgrid(k_vector / mult_K, s_vector)
    dk_dt_vec = ComK(k_vec, s_vec, BK)
    ds_dt_vec = ComS(k_vec, s_vec, BS)
    plt.quiver(k_vec * mult_K, s_vec, dk_dt_vec * mult_K, ds_dt_vec, scale=2, color='gray', scale_units='xy', angles='xy', width=0.004)

    # Afficher des simulations des mouvements de ComK et ComS dans le temps
    
    plt.plot(k * mult_K, s, color='purple', linewidth=1)# Une trace qui resort en violet 
    
    # Plusieurs simulations en rose transparentes en fond 
    num_simulations = 10
    for _ in range(num_simulations):
        k_sim, s_sim, _ = simulate_system(BK, BS, K0, S0, T_END, DT)
        plt.plot(k_sim * mult_K, s_sim, color='pink', alpha=0.3, linewidth=1)

   
    # Calcule quand ComK et ComS sont egaux dans les environs des trois positions donner (guess)
    intersection_list = []
    for guess in initial_guesses:
        intersection = fsolve(equations, guess)
        intersection_list.append(intersection)
    
    
    for intersection in intersection_list:
        plt.plot(intersection[0] * 10, intersection[1], 'ro') # Affiche les points en rouge

    # Parametre d'affichage 
    plt.xlabel('ComK cocnentration (échelle x10)')
    plt.ylabel('ComS cocnentration')
    plt.ylim(0, 8)
    plt.savefig('figure_4a', dpi=300)
    plt.show()



def plot_time_series(k, s, t_end, dt): # Affichage de la figure 4b, une simulation dans le temps du comportement de ComK et ComS
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(0, t_end, dt), k * 10, label='ComK Concentration', color='blue', linewidth=2)
    plt.plot(np.arange(0, t_end, dt), s, label='ComS Concentration', color='green', linewidth=2)
    plt.xlabel('Time (h)')
    plt.ylabel('Protein levels (a.u.)')
    plt.legend()
    plt.savefig('figure_4b.png', dpi=300)
    plt.show()

###########################################################################
# Main 

# Paramètres de la simulation
BK = 0.08
BS = 0.80
K0 = 0.01
S0 = 5
T_END = 120
DT = 0.2

k, s, t = simulate_system(BK, BS, K0, S0, T_END, DT)
plot_nullclines_and_trajectory(k, s, T_END, DT, 10)
plot_time_series(k, s, T_END, DT)

print(f"Valeur maximale de ComS atteinte : S_max = {max(s):.3f}")
