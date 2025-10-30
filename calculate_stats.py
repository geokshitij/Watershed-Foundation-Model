import numpy as np
import tiffile
import os
from tqdm import tqdm

def calculate_stats(root_dir):
    # Define the NoData value that we need to ignore.
    nodata_value = -9999.0

    all_pixels = []
    # Use a representative subset (e.g., 5000 images) for speed.
    image_files = [f for f in os.listdir(root_dir) if f.endswith(('.tiff', '.tif'))][:5000]
    print(f"Calculating stats based on {len(image_files)} images, ignoring NoData value of {nodata_value}...")

    for filename in tqdm(image_files):
        img_path = os.path.join(root_dir, filename)
        try:
            image = tiffile.imread(img_path).astype(np.float32)
            # CRITICAL: Filter out the NoData values before appending.
            valid_pixels = image[image != nodata_value]
            all_pixels.append(valid_pixels)
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    pixel_array = np.concatenate(all_pixels)
    mean = np.mean(pixel_array)
    std = np.std(pixel_array)
    return mean, std

# --- USAGE ---
dataset_path = "/scratch/kdahal3/DEM_CONUS/clipped_huc12_dems"
mean, std = calculate_stats(dataset_path)
print("\n--- Results (Ignoring NoData) ---")
print(f"Calculated Mean: {mean}")
print(f"Calculated Std: {std}")
print("-----------------------------------")
print("Use these values in the UI for your next training run.")