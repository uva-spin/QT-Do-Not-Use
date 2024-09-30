import os
import shutil

def copy_root_files(source_dir, dest_dir):
    """
    Copies 'merged_trackQA_v2.root' files from specified subdirectories
    to a destination directory, renaming them based on their parent folder name.
    
    Args:
    source_dir (str): The path to the source directory containing subfolders.
    dest_dir (str): The path to the destination directory where files will be copied.
    """
    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Iterate through the subdirectories in the source directory
    for folder in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder)

        # Check if it's a directory and matches the desired pattern
        if os.path.isdir(folder_path) and folder.startswith("DY_Target_27M"):
            # Construct the path to the root file
            root_file_path = os.path.join(folder_path, "merged_trackQA_v2.root")
            
            # Check if the root file exists
            if os.path.isfile(root_file_path):
                # Rename the file based on the folder name
                new_file_name = f"{folder}.root"
                new_file_path = os.path.join(dest_dir, new_file_name)

                # Copy the file to the destination directory
                shutil.copy(root_file_path, new_file_path)
                print(f"Copied: {root_file_path} to {new_file_path}")
            else:
                print(f"File not found: {root_file_path}")

# Example usage:
# copy_root_files('/project/ptgroup/spinquest/MC_merge_files', 'rootfiles')
