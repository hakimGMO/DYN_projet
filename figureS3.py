import numpy as np
import matplotlib.pyplot as plt


ak = 0.004
k0 = 0.2
k1 = 0.222
n = 2
p = 5


def ComK(K, S, bk):
    return ak + ((bk * K**n) / (k0**n + K**n)) - (K / (1 + K + S))

def ComS(K, S, bs):
    return (bs / (1 + (K/k1)**p)) - (S / (1 + K + S))

def plot_nullclines(bk, bs, filename):
    K_vals = np.linspace(0, 0.5, 1000) # Creer 100 valeur en tre 0 et 0.5
    S_vals = np.linspace(0, 6, 1000)
    K, S = np.meshgrid(K_vals, S_vals) # Il fait un tableau par valeur de K pour une valeur de S, il y a 100 meme tableau K avec 100 lignes de 0 à 0.45 et 100 tableau avec 100 lignes d'une valeur different pour chaque tableau S

    # Calcul des nullclines
    dK_dt  = ComK(K, S, bk)
    dS_dt  = ComS(K, S, bs)

    intersection_indices = np.argwhere(np.isclose(dK_dt, 0, atol=1e-4) & np.isclose(dS_dt, 0, atol=1e-3))
    
    intersections = [(K[i, j], S[i, j]) for i, j in intersection_indices]
    
    # Tracer les nullclines
    plt.figure(figsize=(8, 6))  # Taille de la figure
    plt.contour(K, S, dK_dt, levels=[0], colors='blue', linewidths=2)
    plt.contour(K, S, dS_dt, levels=[0], colors='green', linewidths=2)

    for (k, s) in intersections:
        plt.plot(k, s, 'ro')
    
    plt.xlabel('ComK')
    plt.ylabel('ComS')
    plt.title('Nullclines pour bk = {}, bs = {}'.format(bk, bs))
    plt.grid(True)
    
    # Sauvegarder la figure
    plt.savefig(filename, dpi=300)
    plt.close()  


plot_nullclines(bk=0.08, bs=0.68, filename='Fig_S3_etat_vegetative_monostable.png')
plot_nullclines(bk=0.12, bs=0.92, filename='Fig_S3_etat_competent_monostable.png')
plot_nullclines(bk=0.14, bs=0.68, filename='Fig_S3_etat_bistable.png')
plot_nullclines(bk=0.08, bs=0.8, filename='Fig_S3_etat_excitable.png')
