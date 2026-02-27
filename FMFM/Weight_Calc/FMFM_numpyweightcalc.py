import numpy as np

beta_mem = 0.9   # same as FMFMNeuronInhib(beta_mem=0.6, ...)
beta_inh = 0.9

def coeffs(d, beta_mem, beta_inh):
    r = beta_inh / beta_mem
    # geometric sum S_d = 1 + r + ... + r^{d-1}
    if abs(r - 1.0) < 1e-9:
        S_d = d
    else:
        S_d = (1 - r**d) / (1 - r)

    A_d = beta_mem**d
    B_d = beta_mem**d * S_d + beta_inh**d
    return A_d, B_d

# choose delays:
d_short = 4
d_best  = 5   # <-- desired best delay
d_long  = 6

# desired voltages at echo time
V_short = 0.9
V_best  = 1.1
V_long  = 0.9

# build system
ds = [d_short, d_best, d_long]
Vs = [V_short, V_best, V_long]

M = []
for d in ds:
    A_d, B_d = coeffs(d, beta_mem, beta_inh)
    M.append([A_d, B_d, 1.0])

M = np.array(M)
V = np.array(Vs)

# solve for [w1, w_inh, w2]
w1, w_inh, w2 = np.linalg.solve(M, V)
print("Solved weights:")
print("  w1   =", w1)
print("  w_inh=", w_inh)
print("  w2   =", w2)
