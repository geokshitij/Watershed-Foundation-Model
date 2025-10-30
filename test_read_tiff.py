import tifffile
import sys

# --- CHANGE THIS to one of the files that produced an error ---
# Example from your logs:
FILE_TO_TEST = "/scratch/kdahal3/DEM_CONUS/clipped_huc12_dems/190102101108.tif"
# -------------------------------------------------------------

print(f"--- Attempting to read a single TIFF file ---")
print(f"File: {FILE_TO_TEST}\n")

try:
    image_data = tifffile.imread(FILE_TO_TEST)
    print("SUCCESS: The file was read successfully!")
    print(f"Image Shape: {image_data.shape}")
    print(f"Data Type: {image_data.dtype}")
except Exception as e:
    print("--- ERROR ---")
    print(f"Failed to read the file.")
    print(f"The specific error was: {e}")
    print("\nThis suggests the issue is with the file itself or the core reading libraries (tifffile/imagecodecs).")
    sys.exit(1)

print("\n--- Test Complete ---")
