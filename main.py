# =========================
# Standard Python 3 Libraries
# =========================

import math
import numpy as np
import matplotlib.pyplot as plt

# Plot defaults for clarity and print-readiness
plt.rcParams.update({
    "figure.figsize": (12, 5),
    "axes.grid": True,
    "font.size": 11
})

# ============================================================
# 1. Install dependencies
# ============================================================
# Note: The line below is for Jupyter/Colab environments. 
# For a local script, install these via your terminal: pip install mpmath pandas tqdm matplotlib
# !pip install mpmath pandas tqdm matplotlib

# ============================================================
# 2. Imports & precision setup
# ============================================================
import mpmath as mp
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

mp.mp.dps = 50  # 50 digits precision

# ============================================================
# 3. Settings
# ============================================================
t_min = 0
t_max = 1000
dt = 0.1  # resolution (10,001 samples)

ts = np.arange(t_min, t_max + dt, dt)

# ============================================================
# 4. Compute zeta magnitudes
# ============================================================
magnitudes = []

print("Computing ζ(1/2 + it) ... This may take 1–2 minutes.")

for t in tqdm(ts):
    z = mp.zeta(0.5 + t * 1j)
    magnitudes.append(float(abs(z)))

# ============================================================
# 5. Save raw results
# ============================================================
df = pd.DataFrame({
    "t": ts,
    "zeta_abs": magnitudes
})

# Note: Update this path if running locally
raw_path = "zeta_samples_0_1000_dt0_1.csv"
# Original Colab path: "/content/zeta_samples_0_1000_dt0_1.csv"
df.to_csv(raw_path, index=False)

print(f"\nSaved full sample CSV to: {raw_path}")

# ============================================================
# 6. Plot |ζ(1/2 + it)|
# ============================================================
plt.figure(figsize=(14, 5))
plt.plot(ts, magnitudes, linewidth=0.8)
plt.title("|ζ(1/2 + it)| from t=0 to t=1000 (dt=0.1)")
plt.xlabel("t")
plt.ylabel("|ζ|")
plt.grid(True)
plt.show()

# ============================================================
# 7. Entropy analysis by intervals
# ============================================================
interval_size = 10  # adjust if needed
entropy_results = []

print("\nComputing entropy per interval...")

for start in tqdm(range(0, 1000, interval_size)):
    end = start + interval_size
    mask = (df["t"] >= start) & (df["t"] < end)
    segment = df[mask]["zeta_abs"].values

    if len(segment) == 0:
        continue

    # Normalize to probabilities
    p = segment / np.sum(segment)
    p = p[p > 0]  # avoid log(0)

    # Shannon entropy
    H = -np.sum(p * np.log(p))

    entropy_results.append((start, end, H))

# Save entropy table
entropy_df = pd.DataFrame(entropy_results, columns=["t_start", "t_end", "entropy"])

# Note: Update this path if running locally
entropy_path = "zeta_entropy_intervals.csv"
# Original Colab path: "/content/zeta_entropy_intervals.csv"
entropy_df.to_csv(entropy_path, index=False)

print(f"\nSaved entropy interval CSV to: {entropy_path}")

print(entropy_df.head())
