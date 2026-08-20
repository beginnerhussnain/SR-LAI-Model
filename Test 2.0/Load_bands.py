import rasterio
import numpy as np
import os


# ==========================================================
# SETTINGS
# ==========================================================

folder = r"D:\Code Playground\VS Code Repository\SR_LAI Model\pypro4sail\inputs"

files = {
    "Blue": "M3_MS_Ortho_Blue.tif",
    "Green": "M3_MS_Ortho_Green.tif",
    "Red": "M3_MS_Ortho_Red.tif",
    "RedEdge": "M3_MS_Ortho_RedEdge.tif",
    "NIR": "M3_MS_Ortho_NIR.tif"
}


# ==========================================================
# 1. LOAD ALL FIVE BANDS
# ==========================================================

bands = []

reference_shape = None
reference_crs = None
reference_transform = None

for band_name, filename in files.items():

    path = os.path.join(folder, filename)

    print(f"\nLoading {band_name}: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with rasterio.open(path) as src:

        data = src.read(1).astype(np.float32)

        print("Shape:", data.shape)
        print("Declared NoData:", src.nodata)
        print("CRS:", src.crs)

        # --------------------------------------------------
        # Check that all bands have the same spatial setup
        # --------------------------------------------------

        if reference_shape is None:

            reference_shape = data.shape
            reference_crs = src.crs
            reference_transform = src.transform

        else:

            if data.shape != reference_shape:
                raise ValueError(
                    f"{band_name} has a different image shape!"
                )

            if src.crs != reference_crs:
                raise ValueError(
                    f"{band_name} has a different CRS!"
                )

            if src.transform != reference_transform:
                raise ValueError(
                    f"{band_name} has a different spatial transform!"
                )

        # --------------------------------------------------
        # Replace NoData with NaN
        # --------------------------------------------------

        if src.nodata is not None:
            data[data == src.nodata] = np.nan

        # --------------------------------------------------
        # Statistics of valid pixels
        # --------------------------------------------------

        valid = data[np.isfinite(data)]

        print("Valid pixels:", len(valid))
        print("Min:", np.min(valid))
        print("Max:", np.max(valid))
        print("Mean:", np.mean(valid))
        print("Median:", np.median(valid))

        bands.append(data)


# ==========================================================
# 2. STACK THE FIVE BANDS
# ==========================================================

# Band order:
#
# 0 = Blue
# 1 = Green
# 2 = Red
# 3 = Red Edge
# 4 = NIR

multispectral = np.stack(bands, axis=0)

print("\n" + "=" * 50)
print("MULTISPECTRAL IMAGE")
print("=" * 50)

print("Final shape:", multispectral.shape)
print("Band order:", list(files.keys()))


# ==========================================================
# 3. CREATE COMMON VALID PIXEL MASK
# ==========================================================

valid_mask = np.all(np.isfinite(multispectral), axis=0)

valid_count = np.sum(valid_mask)
total_count = valid_mask.size

print("\nValid pixels in all five bands:", valid_count)
print("Total pixels:", total_count)

print(
    "Valid percentage:",
    100.0 * valid_count / total_count,
    "%"
)


# ==========================================================
# 4. EXTRACT INDIVIDUAL BANDS
# ==========================================================

blue = multispectral[0]
green = multispectral[1]
red = multispectral[2]
rededge = multispectral[3]
nir = multispectral[4]


# ==========================================================
# 5. CALCULATE NDVI
# ==========================================================

# NDVI = (NIR - Red) / (NIR + Red)

denominator = nir + red

# Avoid division by zero
ndvi = np.full_like(nir, np.nan, dtype=np.float32)

valid_ndvi = valid_mask & (denominator != 0)

ndvi[valid_ndvi] = (
    (nir[valid_ndvi] - red[valid_ndvi])
    / denominator[valid_ndvi]
)


# ==========================================================
# 6. NDVI STATISTICS
# ==========================================================

valid_ndvi_values = ndvi[np.isfinite(ndvi)]

print("\n" + "=" * 50)
print("NDVI STATISTICS")
print("=" * 50)

print("Minimum NDVI:", np.min(valid_ndvi_values))
print("Maximum NDVI:", np.max(valid_ndvi_values))
print("Mean NDVI:", np.mean(valid_ndvi_values))
print("Median NDVI:", np.median(valid_ndvi_values))


# ==========================================================
# 7. SELECT A REALISTIC VEGETATION TEST PIXEL
# ==========================================================

# We don't search through all 111 million pixels.
# Instead, search within a small window around the
# center of the image.

height, width = valid_mask.shape

center_row = height // 2
center_col = width // 2

# Size of search window
window_size = 500

row_min = max(0, center_row - window_size)
row_max = min(height, center_row + window_size)

col_min = max(0, center_col - window_size)
col_max = min(width, center_col + window_size)

# Extract small windows
ndvi_window = ndvi[row_min:row_max, col_min:col_max]
red_window = red[row_min:row_max, col_min:col_max]
nir_window = nir[row_min:row_max, col_min:col_max]

# Find realistic vegetation pixels
vegetation_mask = (
    np.isfinite(ndvi_window)
    & (ndvi_window >= 0.70)
    & (ndvi_window <= 0.90)
    & (red_window > 0.005)
    & (nir_window > red_window)
)

if not np.any(vegetation_mask):
    raise ValueError(
        "No suitable vegetation pixels found "
        "in the center search window."
    )

# Find the pixel with the highest NDVI
# among the realistic vegetation pixels
candidate_ndvi = np.where(
    vegetation_mask,
    ndvi_window,
    np.nan
)

local_row, local_col = np.unravel_index(
    np.nanargmax(candidate_ndvi),
    candidate_ndvi.shape
)

# Convert local coordinates back to full-image coordinates
row = row_min + local_row
col = col_min + local_col

# ==========================================================
# 8. EXTRACT 5-BAND REFLECTANCE FROM TEST PIXEL
# ==========================================================

observed = multispectral[:, row, col]


# ==========================================================
# 9. PRINT TEST PIXEL
# ==========================================================

print("\n" + "=" * 50)
print("HIGH-VEGETATION TEST PIXEL")
print("=" * 50)

print("Row:", row)
print("Column:", col)

print("\nNDVI:", ndvi[row, col])

print("\nObserved reflectance:")

print("Blue    :", observed[0])
print("Green   :", observed[1])
print("Red     :", observed[2])
print("RedEdge :", observed[3])
print("NIR     :", observed[4])

print("\nSpectrum:")
print(observed)


# ==========================================================
# 10. CHECK REFLECTANCE RANGE
# ==========================================================

print("\n" + "=" * 50)
print("TEST PIXEL CHECK")
print("=" * 50)

if np.all((observed >= 0) & (observed <= 1)):

    print("✓ All five values are between 0 and 1.")
    print("✓ Values are consistent with reflectance.")

else:

    print("⚠ Some values are outside the 0-1 reflectance range.")


# ==========================================================
# END
# ==========================================================

print("\n" + "=" * 50)
print("LOAD AND TEST COMPLETE")
print("=" * 50)