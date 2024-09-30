import numpy as np
import os

def combine_npz_files(folder_path: str, output_file: str):
    """
    Combine all .npz files in a specified folder into a single .npz file
    and delete the original files.

    Parameters:
    - folder_path (str): Path to the folder containing .npz files.
    - output_file (str): Name of the output combined .npz file.
    """
    # List all .npz files in the folder
    npz_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npz')]
    print(f"Merging {len(npz_files)} files...")

    # Initialize an empty dictionary to store combined data
    combined_data = {}

    # Iterate through each .npz file
    for npz_file in npz_files:
        # Load the current .npz file
        with np.load(npz_file, allow_pickle=True) as data:
            # Iterate over each key in the file
            for key in data.files:
                if key not in combined_data:
                    combined_data[key] = []
                # Append the data from the current batch
                combined_data[key].append(data[key])

    # Concatenate all data arrays for each key
    for key in combined_data:
        combined_data[key] = np.concatenate(combined_data[key])

    # Save the combined data to a new .npz file
    np.savez_compressed(output_file, **combined_data)

    # Delete the original .npz files
    for npz_file in npz_files:
        os.remove(npz_file)

    print(f"Combined data saved to {output_file} and original files deleted.")
