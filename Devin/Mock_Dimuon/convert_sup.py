import uproot
import ROOT
import numpy as np
import sys

# Check if an input file is provided
if len(sys.argv) != 2:
    print("Usage: python3 script.py inputfile.root")
    sys.exit(1)

# Input and output file paths
input_file_path = sys.argv[1]
output_file_path = "output.root"

# Open the input ROOT file with uproot
with uproot.open(input_file_path) as file:
    tree = file["QA_ana"]

# Get the total number of events in the tree
num_events = tree.num_entries

# Create the output ROOT file with PyROOT
output_file = ROOT.TFile(output_file_path, "RECREATE")
output_tree = ROOT.TTree("QA_ana_filtered", "Filtered QA_ana tree")

# Define branches for the output tree
spillID = np.zeros(1, dtype=int)
eventID = np.zeros(1, dtype=int)
fpgaTriggers = np.zeros(5, dtype=int)
nimTriggers = np.zeros(5, dtype=int)
rfIntensities = np.full(33, 10000, dtype=int)
elementID = ROOT.std.vector('int')()
detectorID = ROOT.std.vector('int')()
driftDistance = ROOT.std.vector('double')()
tdcTime = ROOT.std.vector('double')()
hitID = ROOT.std.vector('int')()

output_tree.Branch("spillID", spillID, "spillID/I")
output_tree.Branch("eventID", eventID, "eventID/I")
output_tree.Branch("fpgaTriggers", fpgaTriggers, "fpgaTriggers[5]/I")
output_tree.Branch("nimTriggers", nimTriggers, "nimTriggers[5]/I")
output_tree.Branch("rfIntensities", rfIntensities, "rfIntensities[33]/I")
output_tree.Branch("elementID", elementID)
output_tree.Branch("detectorID", detectorID)
output_tree.Branch("driftDistance", driftDistance)
output_tree.Branch("tdcTime", tdcTime)
output_tree.Branch("hitID", hitID)

# Iterate over all events in the tree
for event_index, event_data in enumerate(tree.iterate(["n_tracks", "detectorID", "elementID", "driftDistance"], library="np", step_size=1)):
    spillID[0] = event_index  # Dummy data for spillID
    eventID[0] = event_index  # Set the event index as eventID
    fpgaTriggers[:] = [1, 1, 1, 1, 1]
    nimTriggers[:] = [1, 1, 1, 1, 1]
    rfIntensities[:] = np.full(33, 10000, dtype=int)

    # Clear vectors for this event
    elementID.clear()
    detectorID.clear()
    driftDistance.clear()
    tdcTime.clear()
    hitID.clear()

    # Get the number of hits in the current event
    nHits = event_data["nHits"][0]

    # Loop over the hits in the event
    for hit_index in range(nHits):
        elementID.push_back(int(event_data["elementID"][0][hit_index]))
        detectorID.push_back(int(event_data["detectorID"][0][hit_index]))
        driftDistance.push_back(float(event_data["driftDistance"][0][hit_index]))
        tdcTime.push_back(0.0)  # Dummy value for tdcTime
        hitID.push_back(hit_index)  # Index each hit

    # Fill the tree with the current event's data
    output_tree.Fill()

# Write and close the output file
output_file.Write()
output_file.Close()

print(f"Filtered data saved to {output_file_path}")
