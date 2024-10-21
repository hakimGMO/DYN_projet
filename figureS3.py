import numpy as np
import matplotlib.pyplot as plt

ak = 0.004
bk = 0.07
bs = 0.82
k0 = 0.2
k1 = 0.222
n = 2
p = 5

def comK(K, S):
    return ak + (bk * K**n) / (k0**n + K**n) - K / (1 + K + S)

def comS(K, S):
    return bs / (1 + (K/k1)**p) - S / (1 + K + S)
