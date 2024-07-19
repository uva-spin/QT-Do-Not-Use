import os
import ROOT
import subprocess
import uproot
import numpy as np
from numba import njit,prange

def split_files_in_subdirectories(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith("A.root") and "merged_track" in file:
                print(subdir)
                input_root_file_path = os.path.join(subdir, file)
                input_root_file = ROOT.TFile.Open(input_root_file_path)
                tree = input_root_file.Get("QA_ana")
                entries = tree.GetEntries()

                name = os.path.basename(subdir)

                train_file_path = os.path.join(subdir, f"{name}_QA_train.root")
                val_file_path = os.path.join(subdir, f"{name}_QA_val.root")

                train_root_file = ROOT.TFile(train_file_path, "RECREATE")
                train_tree = tree.CloneTree(0)

                # Loop through the specified entry range and copy them to the output tree
                for i in range(int(0.8 * entries)):
                    tree.GetEntry(i)
                    train_tree.Fill()
                train_root_file.Write()
                train_root_file.Close()

                val_root_file = ROOT.TFile(val_file_path, "RECREATE")
                val_tree = tree.CloneTree(0)

                for i in range(int(0.8 * entries), int( entries)):
                    tree.GetEntry(i)
                    val_tree.Fill()
                val_root_file.Write()
                val_root_file.Close()

                input_root_file.Close()
            if file.endswith("2.root") and "merged_track" in file:
                print(subdir)
                input_root_file_path = os.path.join(subdir, file)
                input_root_file = ROOT.TFile.Open(input_root_file_path)
                tree = input_root_file.Get("QA_ana")
                entries = tree.GetEntries()

                name = os.path.basename(subdir)

                train_file_path = os.path.join(subdir, f"{name}_QA2_train.root")
                val_file_path = os.path.join(subdir, f"{name}_QA2_val.root")

                train_root_file = ROOT.TFile(train_file_path, "RECREATE")
                train_tree = tree.CloneTree(0)

                # Loop through the specified entry range and copy them to the output tree
                for i in range(int(0.8 * entries)):
                    tree.GetEntry(i)
                    train_tree.Fill()
                train_root_file.Write()
                train_root_file.Close()

                val_root_file = ROOT.TFile(val_file_path, "RECREATE")
                val_tree = tree.CloneTree(0)

                for i in range(int(0.8 * entries), int( entries)):
                    tree.GetEntry(i)
                    val_tree.Fill()
                val_root_file.Write()
                val_root_file.Close()

                input_root_file.Close()

root_directory = "MC/"
split_files_in_subdirectories(root_directory)




commands = [
    "hadd Merged_MC/All_Vertex_Train_QA.root MC/All_*/*QA_train*",
    "hadd Merged_MC/All_Vertex_Val_QA.root MC/All_*/*QA_val*",
    "hadd Merged_MC/All_Vertex_Train_QA_v2.root MC/All_*/*QA2_train*",
    "hadd Merged_MC/All_Vertex_Val_QA_v2.root MC/All_*/*QA2_val*",
    "hadd Merged_MC/Z_Train_QA.root MC/Z_*/*QA_train*",
    "hadd Merged_MC/Z_Val_QA.root MC/Z_*/*QA_val*",
    "hadd Merged_MC/Z_Train_QA_v2.root MC/Z_*/*QA2_train*",
    "hadd Merged_MC/Z_Val_QA_v2.root MC/Z_*/*QA2_val*",
    "hadd Merged_MC/Target_Train_QA.root MC/Target_*/*QA_train*",
    "hadd Merged_MC/Target_Val_QA.root MC/Target_*/*QA_val*",
    "hadd Merged_MC/Target_Train_QA_v2.root MC/Target_*/*QA2_train*",
    "hadd Merged_MC/Target_Val_QA_v2.root MC/Target_*/*QA2_val*",
    "hadd Merged_MC/Dump_Train_QA.root MC/Dump_*/*QA_train*",
    "hadd Merged_MC/Dump_Val_QA.root MC/Dump_*/*QA_val*",
    "hadd Merged_MC/Dump_Train_QA_v2.root MC/Dump_*/*QA2_train*",
    "hadd Merged_MC/Dump_Val_QA_v2.root MC/Dump_*/*QA2_val*"
]

for command in commands:
    subprocess.run(command, shell=True)

@njit()
def fill_indexes_parallel(indexes_to_fill, mass, mass_bins, entries):
    bin_allocation = int(len(indexes_to_fill) / (len(mass_bins) - 1))
    offset = 0
    for k in range(len(mass_bins) - 1):
        if entries[k] > 0:
            in_bin_mask = (mass > mass_bins[k]) & (mass < mass_bins[k + 1])
            indexes_in_bin = np.where(in_bin_mask)[0]
            num_indexes_in_bin = len(indexes_in_bin)
            if num_indexes_in_bin > 0:
                selected_indexes = np.random.choice(indexes_in_bin, size=bin_allocation, replace=True)
                indexes_to_fill[offset:offset+len(selected_indexes)] = selected_indexes
            offset += bin_allocation
            
# Specify the directory containing root files
directory_path = 'Merged_MC'

# Get a list of all root files in the directory
root_files = [f for f in os.listdir(directory_path) if (f.endswith('.root'))]

# Loop through each root file
for root_file_name in root_files:

    input_root_file = ROOT.TFile.Open(os.path.join(directory_path, root_file_name))
    targettree = uproot.open(os.path.join(directory_path, root_file_name)+':QA_ana')
    px_all=targettree["gpx"].arrays(library="np")["gpx"]
    py_all=targettree["gpy"].arrays(library="np")["gpy"]
    pz_all=targettree["gpz"].arrays(library="np")["gpz"]

    px = np.zeros((len(px_all),2))
    py = np.zeros((len(px_all),2))
    pz = np.zeros((len(px_all),2))
    for i in range(len(px_all)):
        px[i][0]=px_all[i][0]
        px[i][1]=px_all[i][1]
        py[i][0]=py_all[i][0]
        py[i][1]=py_all[i][1]
        pz[i][0]=pz_all[i][0]
        pz[i][1]=pz_all[i][1]

    # Constants
    m_mu = 0.1057

    #print("Mass deviaton:")
    E_1 = np.sqrt(px[:,0]**2 + py[:,0]**2 + pz[:,0]**2 + m_mu**2)
    E_2 = np.sqrt(px[:,1]**2 + py[:,1]**2 + pz[:,1]**2 + m_mu**2)

    # Given equation
    mass = np.sqrt((E_1 + E_2)**2 - (px[:,0] + px[:,1])**2 - (py[:,0] + py[:,1])**2 - (pz[:,0] + pz[:,1])**2)

    entries, mass_bins = np.histogram(mass,bins=np.linspace(2., 9, 71))
    indexes_to_fill = np.zeros(70*np.max(entries), dtype=np.int64)

    # Call the parallelized function
    fill_indexes_parallel(indexes_to_fill, mass, mass_bins, entries)
    sorted_indexes = np.sort(indexes_to_fill)

    # Find unique values and count occurrences
    unique_values, counts = np.unique(sorted_indexes, return_counts=True)

    # Check if the file was opened successfully
    if not input_root_file or input_root_file.IsZombie():
        print("Failed to open file")
        sys.exit(1)  # Use sys.exit instead of exit

    # Access the TTree
    tree = input_root_file.Get("QA_ana")

    # Check if the tree was found
    if not tree:
        print("Failed to find the TTree")
        input_root_file.Close()
        sys.exit(1)  # Use sys.exit instead of exit

    # Create an output ROOT file to save the filtered data
    output_file_path = "Resampled/"+root_file_name
    output_root_file = ROOT.TFile(output_file_path, "RECREATE")

    # Create a new TTree in thae output file with the same structure
    output_tree = tree.CloneTree(0)  # 0 means no entries initially
    i = 0
    # Loop through the specified entry range and copy them to the output tree
    # Loop through the specified entry range and copy them to the output tree
    for i in range(len(unique_values)):
        tree.GetEntry(unique_values[i])
        for j in range(counts[i]):
            output_tree.Fill()


    # Write the output TTree to the output file
    output_root_file.Write()

    # Close both input and output ROOT files
    input_root_file.Close()
    output_root_file.Close()
