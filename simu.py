import numpy as np
import matplotlib.pyplot as plt

# Paramètres des nullclines
ak = 0.004
k0 = 0.2
k1 = 0.222
n = 2
p = 5

def ComK(K, S, bk):
    """Calcul du taux de changement pour K en fonction des paramètres."""
    return ak + ((bk * K**n) / (k0**n + K**n)) - (K / (1 + K + S))

def ComS(K, S, bs):
    """Calcul du taux de changement pour S en fonction des paramètres."""
    return (bs / (1 + (K / k1)**p)) - (S / (1 + K + S))


bk = 0.08  
bs = 0.80  
K0 = 0.01   
S0 = 5    

# Vecteur de temps
dt = 0.03
t_init = 0
t_end = 120
t = np.arange(t_init, t_end, dt)
N = len(t)

# Initialisation des vecteurs pour K et S
K = np.zeros(N)
S = np.zeros(N)

# Conditions initiales
K[0] = K0
S[0] = S0

# Amplitude du bruit gaussien
noise_amplitude = 0.4  


for i in range(1, N):
    
    dK_dt = ComK(K[i-1], S[i-1], bk)
    dS_dt = ComS(K[i-1], S[i-1], bs)
    
    # Ajout de bruit gaussien aux dérivées
    noise_S = np.random.normal(0, noise_amplitude)
    K[i] = K[i-1] + dK_dt * dt 
    S[i] = S[i-1] + dS_dt * dt + noise_S * np.sqrt(dt)

K_visual = K *10
min_S = np.min(S)
print(f"La valeur minimale de S est : {min_S:.3f}")


# Affichage des concentrations de K et S au fil du temps
plt.figure(figsize=(10, 6))
plt.plot(t, K_visual, label='ComK', color='blue',linewidth=4)
plt.plot(t, S, label='ComS', color='green',linewidth=4)
plt.xlabel('Time')
plt.ylabel('Protein levels')
plt.legend()
plt.grid(True)
plt.savefig('simu2.png', dpi=300)
plt.show()  # Affiche la figure
