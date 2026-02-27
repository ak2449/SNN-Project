import numpy as np

def compute_Ad_Bd(beta_mem, beta_inh, d):
    # A_d = β_mem^d
    A_d = beta_mem ** d
    
    # r = β_inh / β_mem
    r = beta_inh / beta_mem
    
    # S_d = sum_{k=0}^{d-1} r^k
    if abs(r - 1.0) < 1e-12:
        S_d = d
    else:
        S_d = (1 - r**d) / (1 - r)
    
    # B_d = β_mem^d * S_d + β_inh^d
    B_d = (beta_mem**d) * S_d + (beta_inh**d)
    
    return A_d, B_d


def solve_weights(beta_mem, beta_inh, d0=5, theta=1.0, margin=0.2):
    # delays to constrain
    d_vals = [d0, d0-1, d0+1]
    
    # targets
    V_targets = [theta, theta - margin, theta - margin]
    
    # Build linear system M * w = V
    M = []
    V = []
    
    for d, Vd in zip(d_vals, V_targets):
        A_d, B_d = compute_Ad_Bd(beta_mem, beta_inh, d)
        M.append([A_d, B_d, 1.0])  # coefficients of [w1, w_inh, w2]
        V.append(Vd)
    
    M = np.array(M)
    V = np.array(V)
    
    # Solve for weights
    w1, w_inh, w2 = np.linalg.solve(M, V)
    return w1, w_inh, w2


# -------------------------
# Example usage
# -------------------------
beta_mem = 0.6
beta_inh = 0.4
d0 = 5
theta = 1.0
margin = 0.15

w1, w_inh, w2 = solve_weights(beta_mem, beta_inh, d0=d0, theta=theta, margin=margin)

print("Computed weights:")
print("w_FM1 (w1)      =", w1)
print("w_inh           =", w_inh)
print("w_FM3 (w2)      =", w2)
