import numpy as np
import matplotlib.pyplot as plt


ak = 0.004  
bk = 0.07  
bs = 0.8   
bg = 0.02   
n, p = 2, 5 
k0 = 0.2
k1 = 0.222
kg = 0.15   
gamma_k = 0.08
gamma_s = 0.1
dt = 0.1   
temps_max = 80  

# Fonction pour calculer les dérivées
def ComKComS(K, S, ak, bk, bs, bg, n, p, k0, k1, kg, gamma_k, gamma_s):
    dK_dt = ak + (bk * K**n) / (k0**n + K**n) - gamma_k * K / (1 + K + S)
    dS_dt = bs / (1 + (K / k1)**p) + (bg * K**n) / (kg**n + K**n) - S / (1 + K + S)
    return dK_dt, dS_dt

# Temps de simulation
t = np.arange(0, temps_max, dt)

# Initialisation des listes pour stocker les valeurs de K et S
K_values = [0.1]
S_values = [0.01]

# Simulation avec la méthode d'Euler
for i in range(1, len(t)):
    K = K_values[-1]
    S = S_values[-1]
    dK_dt, dS_dt = ComKComS(K, S, ak, bk, bs, bg, n, p, k0, k1, kg, gamma_k, gamma_s)
    
    # Mise à jour des valeurs de K et S
    K_new = K + dK_dt * dt
    S_new = S + dS_dt * dt
    K_values.append(K_new)
    S_values.append(S_new)

# Ajout de bruit pour simuler des données expérimentales
K_values_noisy = np.array(K_values) + np.random.normal(0, 0.01, len(K_values))
S_values_noisy = np.array(S_values) + np.random.normal(0, 0.05, len(S_values))

# Graphique
fig, ax1 = plt.subplots(figsize=(8, 5))

# Courbe pour la fluorescence c/p
ax1.plot(t, K_values_noisy, color='red', label="c/p fluorescence (ComK)", linewidth=1)
ax1.set_xlabel("Temps (h)")
ax1.set_ylabel("c/p fluorescence (a.u.)", color="red")
ax1.tick_params(axis="y", labelcolor="red")

# Deuxième axe y pour la fluorescence yfp
ax2 = ax1.twinx()
ax2.plot(t, S_values_noisy, color='green', label="yfp fluorescence (ComS)", linewidth=1)
ax2.set_ylabel("yfp fluorescence (a.u.)", color="green")
ax2.tick_params(axis="y", labelcolor="green")

# Titre du graphique
plt.title("Dynamique des fluorescences c/p et yfp en fonction du temps avec rétroaction (méthode d'Euler)")

# Affichage du graphique
plt.show()






