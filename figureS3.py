import numpy as np
import matplotlib.pyplot as plt

ak = 0.004
bk = 0.07
bs = 0.82
k0 = 0.2
k1 = 0.222
n = 2
p = 5

def dK_dt(K, S):
    return ak + (bk * K**n) / (k0**n + K**n) - (K / (1 + K + S))

def dS_dt(K, S):
    return bs / (1 + (K/k1)**p) - (S / (1 + K + S))


for K in np.arange(0, 8, 0.0001):
    for S in np.arange(0,6,0.00001):
        if dK_dt(K,S) == 0:
            print(K,S)
    