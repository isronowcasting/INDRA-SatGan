import h5py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# --- Configuration: Point this to your data ---
DATA_DIR = "/home/sac/data_67/Nowcasting/data/required_south"
CSV_PATH = "/home/sac/data_67/Nowcasting/data/input_sequences_south.csv"
NUM_FILES_TO_INSPECT = 5  # We'll check a few files to be sure

# -------------------------------------------------

def inspect_file(filepath):
    """Opens a single HDF5 file and prints its statistics."""
    if not os.path.exists(filepath):
        print(f"  - File not found: {filepath}")
        return

    with h5py.File(filepath, "r") as f:
        # Check if the dataset exists
        if "precipitationCal" not in f:
            print(f"  - 'precipitationCal' dataset not found in {filepath}")
            return
            
        data = f["precipitationCal"][:]
        
        # --- The Core Inspection ---
        print(f"  - Min value:      {np.min(data):.4f}")
        print(f"  - Max value:      {np.max(data):.4f}")
        print(f"  - Mean value:     {np.mean(data):.4f}")
        print(f"  - Std Dev:        {np.std(data):.4f}")
        
        # Visualize the distribution with a histogram
        plt.figure(figsize=(10, 6))
        plt.hist(data.flatten(), bins=50)
        plt.title(f"Histogram of Pixel Values\nFile: {os.path.basename(filepath)}")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.5)
        
        # Save the plot
        plot_filename = f"inspection_{os.path.basename(filepath)}.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"  - Saved histogram to: {plot_filename}")


if __name__ == "__main__":
    print("--- Starting Data Inspection ---")
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        exit()
        
    # Get a list of all unique files from the CSV
    all_files_df = pd.read_csv(CSV_PATH, header=None)
    all_files_flat = all_files_df.values.flatten()
    unique_files = pd.unique(all_files_flat[~pd.isna(all_files_flat)])

    if len(unique_files) == 0:
        print("No files found in the CSV.")
        exit()
        
    # Select a random sample of files to inspect
    files_to_check = np.random.choice(unique_files, size=min(NUM_FILES_TO_INSPECT, len(unique_files)), replace=False)

    for i, filename in enumerate(files_to_check):
        print(f"\nInspecting file {i+1}/{len(files_to_check)}: {filename}")
        full_path = os.path.join(DATA_DIR, filename)
        inspect_file(full_path)

    print("\n--- Inspection Complete ---")