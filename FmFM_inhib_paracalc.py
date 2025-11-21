import numpy as np

beta_mem = 0.9
beta_inh = 0.6

def coeffs(d):
    r = beta_inh / beta_mem
    S = sum((r**k) for k in range(d))
    A = beta_mem**d
    B = beta_mem**d * S + beta_inh**d
    return A, B

ds = [1, 5, 12]                 # short, best, long
P = np.array([0.2, 1.1, 0.2])   # desired voltages

M = []
for d in ds:
    A, B = coeffs(d)
    M.append([A, B, 1.0])
M = np.array(M)

w1, w_inh, w2 = np.linalg.solve(M, P)
print(w1, w_inh, w2)