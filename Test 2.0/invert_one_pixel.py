import sys
import numpy as np
from scipy.optimize import differential_evolution, minimize


# ==========================================================
# 1. IMPORT pyPro4Sail
# ==========================================================

REPO_PATH = r"D:\Code Playground\VS Code Repository\SR_LAI Model\pypro4sail"

if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

import pypro4sail

# The repository contains some historical imports using
# "pyPro4Sail". Create an alias here.
# We are NOT modifying the repository.
sys.modules["pyPro4Sail"] = pypro4sail

from pypro4sail import cost_functions


# ==========================================================
# 2. OBSERVED M3M REFLECTANCE
# ==========================================================

# Selected high-vegetation pixel from load_bands.py
#
# Row = 6223
# Column = 7506
# NDVI = 0.9
#
# Blue     = 0.016830701
# Green    = 0.02323949
# Red      = 0.011688411
# RedEdge  = 0.10592813
# NIR      = 0.2220798

observed_reflectance = np.array([
    0.016830701,
    0.02323949,
    0.011688411,
    0.10592813,
    0.2220798
], dtype=np.float64)


# ==========================================================
# 3. M3M BAND WAVELENGTHS
# ==========================================================

# Approximate center wavelengths used for this test.
#
# Blue     = 450 nm
# Green    = 560 nm
# Red      = 650 nm
# RedEdge  = 730 nm
# NIR      = 860 nm

wavelengths = [
    450,
    560,
    650,
    730,
    860
]


# ==========================================================
# 4. INVERSION SETTINGS
# ==========================================================

# For this first test we retrieve ONLY LAI.

ObjParam = [
    "LAI"
]


# ==========================================================
# 5. FIXED PROSAIL PARAMETERS
# ==========================================================

# Exact parameter order from:
#
# four_sail_jacobian.params_prosail
#
# ('N_leaf', 'Cab', 'Car', 'Cbrown', 'Cw',
#  'Cm', 'Ant', 'LAI', 'hotspot', 'leaf_angle')
#
# LAI is optimized, therefore it is NOT included here.

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
# 6. OBSERVATION GEOMETRY
# ==========================================================

n_obs = 1

# View zenith angle
vza = [0.0]

# Assumed solar zenith angle
sza = [30.0]

# Relative view-sun azimuth angle
psi = [0.0]


# ==========================================================
# 7. DIFFUSE SKY FRACTION
# ==========================================================

skyl = np.array([
    [0.2, 0.2, 0.2, 0.2, 0.2]
], dtype=np.float64)


# ==========================================================
# 8. SOIL REFLECTANCE
# ==========================================================

# Temporary neutral soil assumption.
#
# IMPORTANT:
# This is NOT measured soil reflectance.
# It is only being used for this initial inversion test.

rsoil = np.array([
    0.15,
    0.15,
    0.15,
    0.15,
    0.15
], dtype=np.float64)


# ==========================================================
# 9. LAI SCALING
# ==========================================================

# cost_prosail() expects optimized variables scaled
# between 0 and 1.
#
# We allow:
#
# LAI = 0 ... 8
#
# Therefore:
#
# actual_LAI = scaled_LAI * 8

scale = [
    (0.0, 8.0)
]


# ==========================================================
# 10. OBSERVED REFLECTANCE FORMAT
# ==========================================================

# cost_prosail() expects:
#
# observations × wavelengths
#
# Therefore:
#
# (5,) -> (1,5)

rho_canopy = observed_reflectance.reshape(1, -1)


# ==========================================================
# 11. OBJECTIVE FUNCTION
# ==========================================================

def objective(x):

    mse = cost_functions.cost_prosail(
        x,
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

    return float(mse)


# ==========================================================
# 12. HEADER
# ==========================================================

print("=" * 70)
print("PROSAIL SINGLE-PIXEL LAI INVERSION")
print("=" * 70)

print("\nSelected pixel:")
print("Row    : 6223")
print("Column : 7506")


# ==========================================================
# 13. OBSERVED REFLECTANCE
# ==========================================================

print("\nObserved M3M reflectance:")

for wl, refl in zip(wavelengths, observed_reflectance):

    print(
        f"{wl:4d} nm : {refl:.8f}"
    )


# ==========================================================
# 14. TEST DIFFERENT LAI VALUES
# ==========================================================

print("\n" + "=" * 70)
print("TESTING DIFFERENT LAI VALUES")
print("=" * 70)

print("\nLAI        MSE")
print("-" * 30)

for lai in np.arange(0.0, 8.1, 0.5):

    scaled_lai = lai / 8.0

    x = np.array([
        scaled_lai
    ])

    mse = objective(x)

    print(
        f"{lai:4.1f}       {mse:.10f}"
    )


# ==========================================================
# 15. GLOBAL OPTIMIZATION
# ==========================================================

print("\n" + "=" * 70)
print("OPTIMIZING LAI")
print("=" * 70)

print("\nRunning Differential Evolution...")


result = differential_evolution(
    objective,

    bounds=[
        (0.0, 1.0)
    ],

    seed=42,

    tol=1e-8,

    polish=True
)


# ==========================================================
# 16. CONVERT SCALED LAI
# ==========================================================

optimized_scaled_lai = float(result.x[0])

optimized_lai = (
    optimized_scaled_lai * 8.0
)


# ==========================================================
# 17. GLOBAL OPTIMIZATION RESULT
# ==========================================================

print("\n" + "=" * 70)
print("GLOBAL OPTIMIZATION RESULT")
print("=" * 70)

print(
    f"\nScaled LAI : {optimized_scaled_lai:.8f}"
)

print(
    f"Estimated LAI : {optimized_lai:.8f}"
)

print(
    f"MSE : {result.fun:.12f}"
)

print(
    f"Success : {result.success}"
)

print(
    f"Message : {result.message}"
)


# ==========================================================
# 18. REFINED OPTIMIZATION
# ==========================================================

print("\n" + "=" * 70)
print("REFINING RESULT WITH L-BFGS-B")
print("=" * 70)


refined = minimize(
    objective,

    x0=result.x,

    bounds=[
        (0.0, 1.0)
    ],

    method="L-BFGS-B"
)


# ==========================================================
# 19. FINAL LAI
# ==========================================================

refined_scaled_lai = float(refined.x[0])

refined_lai = (
    refined_scaled_lai * 8.0
)


# ==========================================================
# 20. FINAL RESULT
# ==========================================================

print("\n" + "=" * 70)
print("FINAL LAI ESTIMATION")
print("=" * 70)

print(
    f"\nEstimated LAI : {refined_lai:.8f}"
)

print(
    f"Scaled LAI    : {refined_scaled_lai:.8f}"
)

print(
    f"MSE           : {refined.fun:.12f}"
)

print(
    f"Success        : {refined.success}"
)

print(
    f"Message        : {refined.message}"
)


# ==========================================================
# 21. SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
Pixel:
    Row    = 6223
    Column = 7506

Observed bands:
    Blue     = {observed_reflectance[0]:.8f}
    Green    = {observed_reflectance[1]:.8f}
    Red      = {observed_reflectance[2]:.8f}
    RedEdge  = {observed_reflectance[3]:.8f}
    NIR      = {observed_reflectance[4]:.8f}

Estimated LAI:
    {refined_lai:.6f}

Final MSE:
    {refined.fun:.10f}
""")

print("=" * 70)
print("INVERSION COMPLETE")
print("=" * 70)