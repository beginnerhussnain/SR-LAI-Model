import sys
import numpy as np

# ==========================================================
# IMPORT
# ==========================================================

REPO_PATH = r"D:\Code Playground\VS Code Repository\SR_LAI Model\pypro4sail"

if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

import pypro4sail

# Compatibility alias for old repository imports
sys.modules["pyPro4Sail"] = pypro4sail

from pypro4sail import cost_functions


# ==========================================================
# OBSERVED M3M PIXEL
# ==========================================================

observed_reflectance = np.array([
    0.016830701,
    0.02323949,
    0.011688411,
    0.10592813,
    0.2220798
], dtype=np.float64)


# ==========================================================
# WAVELENGTHS
# ==========================================================

wavelengths = [
    450,
    560,
    650,
    730,
    860
]


# ==========================================================
# ONLY OPTIMIZE LAI
# ==========================================================

ObjParam = [
    "LAI"
]


# ==========================================================
# FIXED PARAMETERS
# ==========================================================

FixedValues = [
    1.5,      # N_leaf
    40.0,     # Cab
    10.0,     # Car
    0.0,      # Cbrown
    0.015,    # Cw
    0.009,    # Cm
    0.0,      # Ant
    0.1,      # hotspot
    57.0      # leaf_angle
]


# ==========================================================
# GEOMETRY
# ==========================================================

n_obs = 1

vza = [0.0]
sza = [30.0]
psi = [0.0]


# ==========================================================
# DIFFUSE SKY
# ==========================================================

skyl = np.array([
    [0.2, 0.2, 0.2, 0.2, 0.2]
])


# ==========================================================
# SOIL
# ==========================================================

rsoil = np.array([
    0.15,
    0.15,
    0.15,
    0.15,
    0.15
])


# ==========================================================
# LAI SCALE
# ==========================================================

scale = [
    (0.0, 8.0)
]


# ==========================================================
# OBSERVATION ARRAY
# ==========================================================

rho_canopy = observed_reflectance.reshape(1, -1)


# ==========================================================
# TEST LAI = 4
# ==========================================================

lai = 4.0

scaled_lai = lai / 8.0

x0 = np.array([
    scaled_lai
])


print("=" * 60)
print("TESTING cost_prosail()")
print("=" * 60)

print("\nTesting LAI:", lai)

print("\nCalling cost_prosail...")

mse = cost_functions.cost_prosail(
    x0,
    ObjParam,
    FixedValues,
    n_obs,
    rho_canopy,
    vza,
    sza,
    psi,
    skyl,
    rsoil,
    wavelengths,
    scale
)

print("\nSUCCESS!")

print("MSE:", mse)