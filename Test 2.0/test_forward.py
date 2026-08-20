import sys
import numpy as np

# ==========================================================
# ADD pyPro4Sail REPOSITORY TO PYTHON PATH
# ==========================================================

sys.path.insert(
    0,
    r"D:\Code Playground\VS Code Repository\SR_LAI Model\pypro4sail"
)

from pypro4sail import pypro4sail as p4s


# ==========================================================
# PROSAIL PARAMETERS
# ==========================================================

N = 1.5
chloro = 40.0
caroten = 10.0
brown = 0.0
EWT = 0.015
LMA = 0.009
Ant = 0.0

LAI = 3.0

hot_spot = 0.1

solar_zenith = 30.0
solar_azimuth = 180.0

view_zenith = 0.0
view_azimuth = 0.0

LIDF = 57.0


# ==========================================================
# RUN PROSAIL
# ==========================================================

wl, rho_canopy = p4s.run(
    N,
    chloro,
    caroten,
    brown,
    EWT,
    LMA,
    Ant,
    LAI,
    hot_spot,
    solar_zenith,
    solar_azimuth,
    view_zenith,
    view_azimuth,
    LIDF
)


# ==========================================================
# PRINT RESULTS
# ==========================================================

print("=" * 60)
print("PROSAIL FORWARD MODEL TEST")
print("=" * 60)

print("Number of wavelengths:", len(wl))

print("Wavelength range:")
print("Minimum:", np.min(wl), "nm")
print("Maximum:", np.max(wl), "nm")

print("\nReflectance range:")
print("Minimum:", np.min(rho_canopy))
print("Maximum:", np.max(rho_canopy))

print("\nFirst 10 wavelengths:")
print(wl[:10])

print("\nFirst 10 reflectance values:")
print(rho_canopy[:10])


# ==========================================================
# CHECK SOME COMMON WAVELENGTHS
# ==========================================================

print("\n" + "=" * 60)
print("SAMPLE SPECTRAL VALUES")
print("=" * 60)

wavelengths_to_check = [
    450,
    500,
    550,
    650,
    700,
    750,
    800,
    850
]

for target in wavelengths_to_check:

    index = np.argmin(np.abs(wl - target))

    print(
        f"{wl[index]:.1f} nm -> "
        f"reflectance = {rho_canopy[index]:.6f}"
    )