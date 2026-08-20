import sys
import numpy as np

REPO_PATH = r"D:\Code Playground\VS Code Repository\SR_LAI Model\pypro4sail"

if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

import pypro4sail

# Compatibility alias for old repository imports
sys.modules["pyPro4Sail"] = pypro4sail

from pypro4sail import prospect


print("=" * 60)
print("PROSPECT TEST")
print("=" * 60)

wl, rho, tau = prospect.prospectd(
    1.5,     # N_leaf
    40.0,    # Cab
    10.0,    # Car
    0.0,     # Cbrown
    0.015,   # Cw
    0.009,   # Cm
    0.0      # Ant
)

print("Number of wavelengths:", len(wl))
print("First wavelengths:", wl[:10])
print("First reflectance:", rho[:10])
print("First transmittance:", tau[:10])

print("\nPROSPECT TEST SUCCESS")