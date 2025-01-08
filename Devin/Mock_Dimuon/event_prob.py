import ROOT
import argparse
import numpy as np

# RUS file goals of optimized file: Universality, compressibility, no external librairies, flexibility to scale, speed of reading and writting

# Setting the file compression level: output_file.SetCompressionLevel(9) sets the compression level to 5 (9 is the highest).
# Using LZMA compression: ROOT.kLZMA is specified to use LZMA compression, which is known for its high compression ratio.
# Setting the tree compression level: new_tree.SetCompressionLevel(5) ensures that the TTree itself uses the highest compression level to speed ratio.
# Adjusting the basket size and auto flush: new_tree.SetAutoFlush(2000) and new_tree.SetBasketSize("*", 64000) are used to optimize the tree filling process and compression.



# Parameters for injection
NUM_TRACKS = 5  # Number of tracks to inject (1-100)
PROB_MEAN = 0.9  # Mean probability for keeping hits
PROB_WIDTH = 0.1  # Width of the probability distribution

def inject_tracks(file1, file2, output_file, num_tracks, prob_mean, prob_width):
    if num_tracks < 1 or num_tracks > 100:
        raise ValueError("The number of tracks to inject must be between 1 and 100.")
    if not (0 <= prob_mean <= 1):
        raise ValueError("The mean probability must be between 0 and 1.")
    if prob_width < 0:
        raise ValueError("The width of the probability distribution must be non-negative.")

    # Open the input files
    f1 = ROOT.TFile.Open(file1, "READ")
    f2 = ROOT.TFile.Open(file2, "READ")

    # Access the trees
    tree1 = f1.Get("tree")
    tree2 = f2.Get("tree")

    # Create output file with compression settings
    fout = ROOT.TFile.Open(output_file, "RECREATE", "", ROOT.kLZMA)  # LZMA with max compression
    fout.SetCompressionLevel(9)

    # Create output tree
    output_tree = ROOT.TTree("tree", "Tree with injected tracks and additional data")
    output_tree.SetAutoFlush(0)

    # Prepare variables and branches
    runID = ROOT.Int_t(0)
    spillID = ROOT.Int_t(0)
    eventID = ROOT.Int_t(0)
    fpgaTriggers = np.zeros(5, dtype=int)
    nimTriggers = np.zeros(5, dtype=int)
    rfIntensities = np.full(33, 10000, dtype=int)

    elementIDs = ROOT.std.vector("int")()
    detectorIDs = ROOT.std.vector("int")()
    driftDistances = ROOT.std.vector("double")()
    tdcTimes = ROOT.std.vector("double")()
    hitsInTime = ROOT.std.vector("bool")()

    triggerElementIDs = ROOT.std.vector("int")()
    triggerDetectorIDs = ROOT.std.vector("int")()
    triggerDriftDistances = ROOT.std.vector("double")()
    triggerTdcTimes = ROOT.std.vector("double")()
    triggerHitsInTime = ROOT.std.vector("bool")()

    output_tree.Branch("runID", runID, "runID/I")
    output_tree.Branch("spillID", spillID, "spillID/I")
    output_tree.Branch("eventID", eventID, "eventID/I")
    output_tree.Branch("fpgaTriggers", fpgaTriggers, "fpgaTriggers[5]/I")
    output_tree.Branch("nimTriggers", nimTriggers, "nimTriggers[5]/I")
    output_tree.Branch("rfIntensities", rfIntensities, "rfIntensities[33]/I")

    output_tree.Branch("elementIDs", elementIDs)
    output_tree.Branch("detectorIDs", detectorIDs)
    output_tree.Branch("driftDistances", driftDistances)
    output_tree.Branch("tdcTimes", tdcTimes)
    output_tree.Branch("hitsInTime", hitsInTime)

    output_tree.Branch("triggerElementIDs", triggerElementIDs)
    output_tree.Branch("triggerDetectorIDs", triggerDetectorIDs)
    output_tree.Branch("triggerDriftDistances", triggerDriftDistances)
    output_tree.Branch("triggerTdcTimes", triggerTdcTimes)
    output_tree.Branch("triggerHitsInTime", triggerHitsInTime)

    # Loop over events in tree1 and inject tracks from tree2
    num_events_tree2 = tree2.GetEntries()
    tree2_index = 0

    for entry1 in tree1:
        if tree2_index >= num_events_tree2:
            break  # Stop when the second file is exhausted

        # Reset vectors
        elementIDs.clear()
        detectorIDs.clear()
        driftDistances.clear()
        tdcTimes.clear()
        hitsInTime.clear()

        triggerElementIDs.clear()
        triggerDetectorIDs.clear()
        triggerDriftDistances.clear()
        triggerTdcTimes.clear()
        triggerHitsInTime.clear()

        # Populate event-specific metadata
        runID = entry1.runID
        spillID = entry1.spillID
        eventID = entry1.eventID
        fpgaTriggers[:] = entry1.fpgaTriggers
        nimTriggers[:] = entry1.nimTriggers
        rfIntensities[:] = entry1.rfIntensities

        # Add hits from tree1
        for elem in entry1.elementIDs:
            elementIDs.push_back(elem)
        for det in entry1.detectorIDs:
            detectorIDs.push_back(det)
        for dist in entry1.driftDistances:
            driftDistances.push_back(dist)
        for tdc in entry1.tdcTimes:
            tdcTimes.push_back(tdc)
        for hit in entry1.hitsInTime:
            hitsInTime.push_back(hit)

        # Inject tracks from tree2
        for _ in range(num_tracks):
            if tree2_index >= num_events_tree2:
                break  # Stop when the second file is exhausted

            tree2.GetEntry(tree2_index)
            tree2_index += 1

            # Generate random probability for this track
            probability = np.clip(np.random.normal(prob_mean, prob_width), 0, 1)

            for elem, det, dist, tdc, hit in zip(tree2.elementIDs, tree2.detectorIDs, 
                                                 tree2.driftDistances, tree2.tdcTimes, 
                                                 tree2.hitsInTime):
                # Determine whether to keep the hit based on detector ID and probability
                keep_hit = np.random.random() < probability * (1 - det / 100)
                if keep_hit:
                    elementIDs.push_back(elem)
                    detectorIDs.push_back(det)
                    driftDistances.push_back(dist)
                    tdcTimes.push_back(tdc)
                    hitsInTime.push_back(hit)

            for elem, det, dist, tdc, hit in zip(tree2.triggerElementIDs, tree2.triggerDetectorIDs,
                                                 tree2.triggerDriftDistances, tree2.triggerTdcTimes,
                                                 tree2.triggerHitsInTime):
                triggerElementIDs.push_back(elem)
                triggerDetectorIDs.push_back(det)
                triggerDriftDistances.push_back(dist)
                triggerTdcTimes.push_back(tdc)
                triggerHitsInTime.push_back(hit)

        # Fill the combined tree
        output_tree.Fill()

    # Write the output file
    fout.Write()
    fout.Close()
    f1.Close()
    f2.Close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject tracks with a hit-keeping probability from one ROOT file into another.")
    parser.add_argument("file1", type=str, help="Path to the first ROOT file.")
    parser.add_argument("file2", type=str, help="Path to the second ROOT file.")
    parser.add_argument("--output", type=str, default="prob_tracks.root",
                        help="Name of the output ROOT file.")
    args = parser.parse_args()

    inject_tracks(args.file1, args.file2, args.output, NUM_TRACKS, PROB_MEAN, PROB_WIDTH)
