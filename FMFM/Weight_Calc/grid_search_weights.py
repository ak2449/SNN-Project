import numpy as np
import pandas as pd

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

# choose delays (fixed)
d_short = 2
d_best  = 5
d_long  = 8

# Grid search ranges
beta_mem_values = np.linspace(0.01, 0.99, 10)
beta_inh_values = np.linspace(0.01, 0.99, 10)

# Try different desired voltages to find feasible solutions
voltage_configs = [
    {'V_short': 0.9, 'V_best': 1.1, 'V_long': 0.9},   # Original
    {'V_short': 0.5, 'V_best': 1.1, 'V_long': 0.5},   # Lower voltages
    {'V_short': 0.3, 'V_best': 1.1, 'V_long': 0.3},   # Even lower
    {'V_short': 0.1, 'V_best': 1.1, 'V_long': 0.1},   # Minimal
]

best_config = None
best_results = []

for config in voltage_configs:
    V_short = config['V_short']
    V_best = config['V_best']
    V_long = config['V_long']
    
    results = []
    
    for beta_mem in beta_mem_values:
        for beta_inh in beta_inh_values:
            try:
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
                weights = np.linalg.solve(M, V)
                w1, w_inh, w2 = weights
                
                results.append({
                    'beta_mem': beta_mem,
                    'beta_inh': beta_inh,
                    'w1': w1,
                    'w_inh': w_inh,
                    'w2': w2
                })
            except np.linalg.LinAlgError:
                # Skip singular matrices
                continue
    
    # Filter for valid weight combinations
    df_test = pd.DataFrame(results)
    df_filtered = df_test[(df_test['w1'] >= 0) & (df_test['w2'] >= 0) & (df_test['w_inh'] <= 0)]
    
    if len(df_filtered) > 0:
        best_config = config
        best_results = results
        print(f"✓ Found {len(df_filtered)} valid combinations with V_short={V_short}, V_best={V_best}, V_long={V_long}\n")
        break
    else:
        print(f"✗ No valid combinations with V_short={V_short}, V_best={V_best}, V_long={V_long}")

if best_config is None:
    print("\nNo feasible voltage configuration found. Using last attempted configuration.")
    V_short = voltage_configs[-1]['V_short']
    V_best = voltage_configs[-1]['V_best']
    V_long = voltage_configs[-1]['V_long']
    best_results = results

results = best_results

# Create DataFrame for easy viewing and analysis
df = pd.DataFrame(results)

# Filter out invalid weight combinations:
# - Keep positive w1
# - Keep negative w2 (inhibitory weights are typically negative)
# - w_inh can be positive or negative (adjust constraint based on results)
df_filtered = df[(df['w1'] >= 0) & (df['w2'] >= 0)]

print("Grid Search Results (Filtered):")
print("=" * 80)
print(df_filtered.to_string(index=False))
print("=" * 80)


# Print some statistics
print("\nStatistics:")
print(f"Total valid combinations: {len(df)}")
print(f"Valid filtered combinations: {len(df_filtered)}")
if len(df_filtered) > 0:
    print(f"\nw1 range: [{df_filtered['w1'].min():.6f}, {df_filtered['w1'].max():.6f}]")
    print(f"w_inh range: [{df_filtered['w_inh'].min():.6f}, {df_filtered['w_inh'].max():.6f}]")
    print(f"w2 range: [{df_filtered['w2'].min():.6f}, {df_filtered['w2'].max():.6f}]")
else:
    print("\nNo valid combinations found with the specified constraints.")
