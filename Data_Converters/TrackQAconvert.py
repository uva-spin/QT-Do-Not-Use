import ROOT
import numpy as np
import argparse
from pathlib import Path

class TrackQAConverter:
    def __init__(self, input_file, output_file, compression_level=5):
        """
        Initialize the TrackQA converter
        
        Args:
            input_file (str): Path to input ROOT file
            output_file (str): Path to output ROOT file
            compression_level (int): Compression level for output file (default: 5)
        """
        self.input_file_path = input_file
        self.output_file_path = output_file
        self.compression_level = compression_level
        
        # Initialize arrays
        self.triggerbits = np.zeros(10, dtype='int32')
        self.detectorID = np.zeros(500, dtype='int32')
        self.elementID = np.zeros(500, dtype='int32')
        self.driftDistance = np.zeros(500, dtype='float32')
        self.tdcTime = np.zeros(500, dtype='float32')
        self.nHits = np.zeros(1, dtype='int32')
        
        # Initialize vectors
        self.hitID = ROOT.std.vector('int')()
        self.elementID_vec = ROOT.std.vector('int')()
        self.detectorID_vec = ROOT.std.vector('int')()
        self.driftDistance_vec = ROOT.std.vector('double')()
        self.tdcTime_vec = ROOT.std.vector('double')()
        self.hitInTime_vec = ROOT.std.vector('bool')()
        
        # Initialize other arrays
        self.runID = np.zeros(1, dtype=int)
        self.eventID = np.zeros(1, dtype=int)
        self.spillID = np.zeros(1, dtype=int)
        self.rfID = np.zeros(1, dtype=int)
        self.rfIntensity = np.zeros(33, dtype=int)
        self.fpgaTrigger = np.zeros(5, dtype=int)
        self.nimTrigger = np.zeros(5, dtype=int)

    def setup_input_tree(self, source_tree):
        """Set up the input tree branch addresses"""
        source_tree.SetBranchAddress("triggerbits", self.triggerbits)
        source_tree.SetBranchAddress("detectorID", self.detectorID)
        source_tree.SetBranchAddress("elementID", self.elementID)
        source_tree.SetBranchAddress("driftDistance", self.driftDistance)
        source_tree.SetBranchAddress("nHits", self.nHits)

    def create_output_branches(self, new_tree):
        """Create branches in the output tree"""
        # Create scalar branches
        new_tree.Branch("runID", self.runID, "runID/I")
        new_tree.Branch("spillID", self.spillID, "spillID/I")
        new_tree.Branch("eventID", self.eventID, "eventID/I")
        new_tree.Branch("rfFID", self.rfID, "rfID/I")
        new_tree.Branch("rfIntensity", self.rfIntensity, "rfIntensity[33]/I")
        new_tree.Branch("fpgaTrigger", self.fpgaTrigger, "fpgaTrigger[5]/I")
        new_tree.Branch("nimTrigger", self.nimTrigger, "nimTrigger[5]/I")

        # Create vector branches
        new_tree.Branch("hitID", self.hitID)
        new_tree.Branch("elementID", self.elementID_vec)
        new_tree.Branch("detectorID", self.detectorID_vec)
        new_tree.Branch("driftDistance", self.driftDistance_vec)
        new_tree.Branch("tdcTime", self.tdcTime_vec)
        new_tree.Branch("hitInTime", self.hitInTime_vec)

    def process_triggers(self):
        """Process trigger bits"""
        self.nimTrigger[:] = 0
        self.fpgaTrigger[:] = 0
        
        for j in range(10):
            current_trigger = self.triggerbits[j]
            if j < 5:
                for bit in range(5):
                    if current_trigger & (1 << bit):
                        self.nimTrigger[bit] = 1
            elif 5 <= j < 10:
                for bit in range(5):
                    if current_trigger & (1 << bit):
                        self.fpgaTrigger[bit] = 1

    def clear_vectors(self):
        """Clear all vectors before processing new entry"""
        self.hitID.clear()
        self.elementID_vec.clear()
        self.detectorID_vec.clear()
        self.driftDistance_vec.clear()
        self.tdcTime_vec.clear()
        self.hitInTime_vec.clear()

    def process_hits(self):
        """Process hits for current entry"""
        for k in range(self.nHits[0]):
            self.hitID.push_back(k)
            self.elementID_vec.push_back(int(self.elementID[k]))
            self.driftDistance_vec.push_back(float(self.driftDistance[k]))
            self.tdcTime_vec.push_back(0.0)
            self.detectorID_vec.push_back(int(self.detectorID[k]))
            self.hitInTime_vec.push_back(1)

    def convert(self):
        """Perform the conversion"""
        # Open input file
        source_file = ROOT.TFile.Open(self.input_file_path)
        if not source_file or source_file.IsZombie():
            raise RuntimeError(f"Could not open input file: {self.input_file_path}")
        
        source_tree = source_file.Get("QA_ana")
        if not source_tree:
            raise RuntimeError("Could not find QA_ana tree in input file")

        # Create output file
        output_file = ROOT.TFile(self.output_file_path, "RECREATE", "", ROOT.kLZMA)
        output_file.SetCompressionLevel(self.compression_level)
        new_tree = ROOT.TTree("tree", "Tree with vector branches")
        new_tree.SetAutoFlush(2500)

        # Setup trees
        self.setup_input_tree(source_tree)
        self.create_output_branches(new_tree)

        # Process entries
        n_entries = source_tree.GetEntries()
        for i in range(n_entries):
            source_tree.GetEntry(i)
            self.clear_vectors()
            self.process_triggers()
            self.process_hits()
            new_tree.Fill()

        # Save and close
        output_file.Write()
        output_file.Close()
        source_file.Close()

def main():
    parser = argparse.ArgumentParser(description='Convert TrackQA ROOT file format')
    parser.add_argument('input_file', help='Input ROOT file path')
    parser.add_argument('output_file', help='Output ROOT file path')
    parser.add_argument('--compression', type=int, default=5,
                      help='Compression level for output file (default: 5)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' does not exist")
        return 1
    
    try:
        converter = TrackQAConverter(args.input_file, args.output_file, args.compression)
        converter.convert()
        print(f"Successfully converted {args.input_file} to {args.output_file}")
        return 0
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())

