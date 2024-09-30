import os
from dataProcess import Data_Processing
import numpy as np
from time import time
from Merge_npz import combine_npz_files

start = time()

# Directory containing the root files
root_dir = "rootfiles"

# Get a list of all .root files in the directory
root_files = [f for f in os.listdir(root_dir) if f.endswith('.root')]

for root_file in root_files:
    print(f"Processing {root_file}...")
    
    # Construct the full file path
    full_path = os.path.join(root_dir, root_file)
    
    data_processor = Data_Processing(full_path)

    num_events = data_processor.get_num_events()

    ideal_events = np.zeros(num_events)
    # Create an array of ideal events
    for event in range(num_events):
        good_event = data_processor.find_ideal_events(event)
        if good_event:
            ideal_events[event] = event
    ideal_events = ideal_events[ideal_events != 0]

    print(f"There are this many ideal events in {root_file}: {len(ideal_events)}")

    def batch_generator(data, batch_size):
        """Yield successive batches from the data."""
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    # Use a generator to avoid loading all events into memory at once
    batch_size = 100000
    batch_gen = batch_generator(ideal_events, batch_size)

    # Loop through the generator
    for i, batch_events in enumerate(batch_gen, 1):
        # Process the batch with make_Hitmatrix
        Truth_elementID_mup, Truth_elementID_mum, Truth_values_drift_mup, Truth_values_drift_mum, hit_matrix = data_processor.make_Hitmatrix(batch_events)

        # Save the batch results to the npz file using compression
        with open(f'Hit_Data/Hit_Info_{os.path.splitext(root_file)[0]}_batch_{i}.npz', 'wb') as f:
            np.savez_compressed(f, Truth_elementID_mup=Truth_elementID_mup, 
                                      Truth_elementID_mum=Truth_elementID_mum, 
                                      Truth_values_drift_mup=Truth_values_drift_mup, 
                                      Truth_values_drift_mum=Truth_values_drift_mum, 
                                      hit_matrix=hit_matrix, 
                                      ideal_events=batch_events)

        print(f"Batch {i} saved with {len(batch_events)} events for {root_file}.")

# Merge and delete files using merge_npz.py:
# Define the folder path and output file name
folder_path = 'Hit_Data'
output_file = 'Hit_Info.npz'
combine_npz_files(folder_path, output_file)

stop = time()

print(f"This code took this long to complete: {stop - start}")
