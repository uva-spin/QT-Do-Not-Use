import uproot
import numpy as np
import os
from sklearn.ensemble import IsolationForest

# Initialize iForest
iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)

input_dir = "path_to_input_root_files"
output_dir = "path_to_save_good_events"

os.makedirs(output_dir, exist_ok=True)

def process_root_file(file_path):
    with uproot.open(file_path) as file:
        tree = file["tree_name"]
        hit_matrix = tree.arrays(["branch_name_containing_hit_matrix"], library="np")

    X = hit_matrix.flatten().reshape(1, -1) 

    y_pred = iso_forest.fit_predict(X)

    # 1 for good events
    is_good_event = y_pred[0] == 1

    return is_good_event

# Iterate over all ROOT files in the input directory
for root_file in os.listdir(input_dir):
    if root_file.endswith(".root"):
        file_path = os.path.join(input_dir, root_file)
        
        # Process the file to determine if it contains a good event
        if process_root_file(file_path):
            # If it's a good event, save the file to the output directory
            output_path = os.path.join(output_dir, root_file)
            os.rename(file_path, output_path)
            print(f"Saved good event: {root_file}")
        else:
            # If it's a bad event, delete or skip the file
            print(f"Discarded bad event: {root_file}")
            # os.remove(file_path) # Don't remove just yet, need to see if outputted events are actually good

print("Batch processing complete.")
