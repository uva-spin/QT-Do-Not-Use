import uproot
import ROOT
import numpy as np
import argparse

class SRAWConverter:
    def __init__(self, input_file, output_file):
        self.input_file_path = input_file
        self.output_file_path = output_file
        
        # Initialize scalar variables
        self.runID = np.zeros(1, dtype=int)
        self.eventID = np.zeros(1, dtype=int)
        self.spillID = np.zeros(1, dtype=int)
        self.rfID = np.zeros(1, dtype=int)
        self.rfIntensity = np.zeros(33, dtype=int)
        self.fpgaTrigger = np.zeros(5, dtype=int)
        self.nimTrigger = np.zeros(5, dtype=int)
        
        # Initialize vector variables
        self.hitID = ROOT.std.vector('int')()
        self.elementID = ROOT.std.vector('int')()
        self.detectorID = ROOT.std.vector('int')()
        self.driftDistance = ROOT.std.vector('double')()
        self.tdcTime = ROOT.std.vector('double')()
        self.hitInTime = ROOT.std.vector('bool')()
        
        # Define branches to read
        self.branches = [
            "rawEvent/fRunID", "rawEvent/fEventID", "rawEvent/fSpillID", "rawEvent/fTriggerBits",
            "rawEvent/fRFID", "rawEvent/fIntensity[33]",
            "rawEvent/fAllHits/fAllHits.index", "rawEvent/fAllHits/fAllHits.detectorID",
            "rawEvent/fAllHits/fAllHits.elementID", "rawEvent/fAllHits/fAllHits.tdcTime",
            "rawEvent/fAllHits/fAllHits.driftDistance"
        ]

    def setup_output_tree(self):
        """Set up the output ROOT file and tree structure"""
        self.output_file = ROOT.TFile(self.output_file_path, "RECREATE", "", ROOT.kLZMA)
        self.output_file.SetCompressionLevel(5)
        self.new_tree = ROOT.TTree("tree", "Optimized Data Tree")
        self.new_tree.SetAutoFlush(2500)
        self.new_tree.SetBasketSize("*", 64000)
        
        # Create branches for scalar variables
        self.new_tree.Branch("runID", self.runID, "runID/I")
        self.new_tree.Branch("spillID", self.spillID, "spillID/I")
        self.new_tree.Branch("eventID", self.eventID, "eventID/I")
        self.new_tree.Branch("rfFID", self.rfID, "rfID/I")
        self.new_tree.Branch("rfIntensity", self.rfIntensity, "rfIntensity[33]/I")
        self.new_tree.Branch("fpgaTrigger", self.fpgaTrigger, "fpgaTrigger[5]/I")
        self.new_tree.Branch("nimTrigger", self.nimTrigger, "nimTrigger[5]/I")
        
        # Create branches for vector variables
        self.new_tree.Branch("hitID", self.hitID)
        self.new_tree.Branch("elementID", self.elementID)
        self.new_tree.Branch("detectorID", self.detectorID)
        self.new_tree.Branch("driftDistance", self.driftDistance)
        self.new_tree.Branch("tdcTime", self.tdcTime)
        self.new_tree.Branch("hitInTime", self.hitInTime)

    def process_trigger_bits(self, trigger_bits):
        """Process trigger bits for NIM and FPGA"""
        self.nimTrigger.fill(0)  # Reset triggers
        self.fpgaTrigger.fill(0)
        
        for bit in range(5):  # NIM triggers
            if trigger_bits & (1 << bit):
                self.nimTrigger[bit] = 1
                
        for bit in range(5, 10):  # FPGA triggers
            if trigger_bits & (1 << bit):
                self.fpgaTrigger[bit - 5] = 1

    def clear_vectors(self):
        """Clear all vector branches"""
        self.hitID.clear()
        self.elementID.clear()
        self.detectorID.clear()
        self.driftDistance.clear()
        self.tdcTime.clear()
        self.hitInTime.clear()

    def convert(self):
        """Main conversion method"""
        try:
            # Open input file
            input_file = uproot.open(self.input_file_path)
            tree = input_file["save"]
            
            # Setup output
            self.setup_output_tree()
            
            # Process data
            for data in tree.iterate(self.branches, library="np"):
                for i in range(len(data["rawEvent/fRunID"])):
                    # Fill scalar branches
                    self.runID[0] = data["rawEvent/fRunID"][i]
                    self.eventID[0] = data["rawEvent/fEventID"][i]
                    self.spillID[0] = data["rawEvent/fSpillID"][i]
                    self.rfID[0] = data["rawEvent/fRFID"][i]
                    self.rfIntensity[:] = data["rawEvent/fIntensity[33]"][i]
                    
                    # Process triggers
                    self.process_trigger_bits(data["rawEvent/fTriggerBits"][i])
                    
                    # Clear and fill vector branches
                    self.clear_vectors()
                    
                    hits_count = len(data["rawEvent/fAllHits/fAllHits.index"][i])
                    for j in range(hits_count):
                        self.hitID.push_back(int(data["rawEvent/fAllHits/fAllHits.index"][i][j]))
                        self.elementID.push_back(int(data["rawEvent/fAllHits/fAllHits.elementID"][i][j]))
                        self.detectorID.push_back(int(data["rawEvent/fAllHits/fAllHits.detectorID"][i][j]))
                        self.driftDistance.push_back(float(data["rawEvent/fAllHits/fAllHits.driftDistance"][i][j]))
                        self.tdcTime.push_back(float(data["rawEvent/fAllHits/fAllHits.tdcTime"][i][j]))
                        self.hitInTime.push_back(1)
                    
                    self.new_tree.Fill()
            
            # Save and close
            self.new_tree.Write()
            self.output_file.Close()
            print(f"Conversion completed successfully. Output saved to {self.output_file_path}")
            
        except Exception as e:
            print(f"Error during conversion: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Convert ROOT file format')
    parser.add_argument('input_file', help='Input ROOT file path')
    parser.add_argument('output_file', help='Output ROOT file path')
    
    args = parser.parse_args()
    
    converter = SRAWConverter(args.input_file, args.output_file)
    converter.convert()

if __name__ == "__main__":
    main()
