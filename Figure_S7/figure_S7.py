import numpy as np
import matplotlib.pyplot as plt

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
    noise_amplitude = 0.6

    for i in range(1, n):
        dk_dt = ComK(k[i-1], s[i-1], bk)
        ds_dt = ComS(k[i-1], s[i-1], bs)
        noise_s = np.random.normal(0, noise_amplitude)
        k[i] = k[i-1] + dk_dt * dt
        s[i] = s[i-1] + ds_dt * dt + noise_s * np.sqrt(dt)

    return k, s, t

def calculate_excursion_times(k, t, threshold=0.4):
    excursion_times = []
    in_excursion = False
    start_time = 0

    for i in range(len(k)):
        if k[i] * 10 > threshold:
            if not in_excursion:
                in_excursion = True
                start_time = t[i]
        else:
            if in_excursion:
                in_excursion = False
                end_time = t[i]
                duration = end_time - start_time
                if duration >= 10:
                    excursion_times.append(duration)
                    break

    return excursion_times

# Paramètres de la simulation
BK = 0.08
BS = 0.80
K0 = 0.01
S0 = 5
T_END = 120
DT = 0.2

NUM_SIMULATIONS = 2  # Nombre de séries temporelles à afficher

plt.figure(figsize=(15, 10))

# Générer et afficher plusieurs séries temporelles
for i in range(NUM_SIMULATIONS):
    k, s, t = simulate_system(BK, BS, K0, S0, T_END, DT)
    plt.subplot(3, 2, i + 1)  # Organiser les sous-figures en 3x2
    plt.plot(t, k * 10, label='ComK Concentration (x10)', color='blue', linewidth=1.5)
    plt.plot(t, s, label='ComS Concentration', color='green', linewidth=1.5)

    # Remplir l'arrière-plan en rouge lorsque k dépasse 0.4
    threshold = 0.8
    for j in range(len(k) - 1):
        if k[j] * 10 > threshold and k[j + 1] * 10 > threshold:
            plt.fill_between(t[j:j+2], 0, 1, color='red', alpha=0.3)

    plt.xlabel('Temps')
    plt.ylabel('Niveaux de protéines')
    plt.title(f'Simulation de ComK et ComS - Série {i + 1}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig('FigS7_time_series.png', dpi=300)
plt.show()

NUM_SIMULATIONS = 1000

all_excursion_times = []
threshold = 0.8
# Exécuter plusieurs simulations pour collecter les temps d'excursion
for _ in range(NUM_SIMULATIONS):
    k, s, t = simulate_system(BK, BS, K0, S0, T_END, DT)
    excursion_times = calculate_excursion_times(k, t, threshold)
    all_excursion_times.extend(excursion_times)

# Calculer le compte cumulatif
all_excursion_times.sort()
cumulative_count = np.arange(1, len(all_excursion_times) + 1) / len(all_excursion_times)

# Tracer la figure S7 avec l'histogramme
plt.figure(figsize=(12, 6))

# Histogramme des temps d'excursion
plt.subplot(1, 2, 1)
plt.hist(all_excursion_times, bins=15, color='skyblue', edgecolor='black')
plt.xlabel("Temps d'excursion (unités de temps)")
plt.ylabel("Fréquence")
plt.title("Histogramme des temps d'excursion")
plt.grid(True)

# Courbe cumulative des temps d'excursion
plt.subplot(1, 2, 2)
plt.plot(all_excursion_times, cumulative_count, marker='o', linestyle='-', color='purple')
plt.xlabel("Temps d'excursion (unités de temps)")
plt.ylabel("Taux cumulatif des excursions")
plt.title("Distribution cumulative des temps d'excursion")
plt.grid(True)

plt.tight_layout()
plt.savefig('FigS7_excursion_times_histogram.png', dpi=300)
plt.show()
