import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from time import time
from tqdm import tqdm
from typing import List, Tuple, Optional
# from Generate_Background import Background

class DataProcessing:
    """
    This program processes data from QA_V2 for SpinQuest and UVA.
    It reads branches from a ROOT file, filters ideal events, and organizes data into
    a hit matrix and ground truth labels. The goal is to preprocess the data for use in a DNN.
    """

    def __init__(self, rootfile: str) -> None:
        """
        Initializes the DataProcessing class.

        Args:
            rootfile (str): Path to the ROOT file containing the data.
        """
        self.rootfile = rootfile
        print("Reading the ROOT file...")
        self.TTree = uproot.open(self.rootfile + ":QA_ana")

        # Define detector and drift chamber order
        self.detectors_order = np.array([
            'D0U_ele', 'D0Up_ele', 'D0X_ele', 'D0Xp_ele', 'D0V_ele', 'D0Vp_ele',
            'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN',
            'D2V_ele', 'D2Vp_ele', 'D2Xp_ele', 'D2X_ele', 'D2U_ele', 'D2Up_ele',
            'D3pVp_ele', 'D3pV_ele', 'D3pXp_ele', 'D3pX_ele', 'D3pUp_ele', 'D3pU_ele',
            'D3mVp_ele', 'D3mV_ele', 'D3mXp_ele', 'D3mX_ele', 'D3mUp_ele', 'D3mU_ele',
            'H1B_ele', 'H1T_ele', 'H1L_ele', 'H1R_ele', 'H2L_ele', 'H2R_ele',
            'H2B_ele', 'H2T_ele', 'H3B_ele', 'H3T_ele', 'H4Y1L_ele', 'H4Y1R_ele',
            'H4Y2L_ele', 'H4Y2R_ele', 'H4B_ele', 'H4T_ele', 'P1Y1_ele', 'P1Y2_ele',
            'P1X1_ele', 'P1X2_ele', 'P2X1_ele', 'P2X2_ele', 'P2Y1_ele', 'P2Y2_ele',
            'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN'
        ])

        self.drift_order = np.array([
            'D0U_drift', 'D0Up_drift', 'D0X_drift', 'D0Xp_drift', 'D0V_drift', 'D0Vp_drift',
            'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN',
            'D2V_drift', 'D2Vp_drift', 'D2Xp_drift', 'D2X_drift', 'D2U_drift', 'D2Up_drift',
            'D3pVp_drift', 'D3pV_drift', 'D3pXp_drift', 'D3pX_drift', 'D3pUp_drift', 'D3pU_drift',
            'D3mVp_drift', 'D3mV_drift', 'D3mXp_drift', 'D3mX_drift', 'D3mUp_drift', 'D3mU_drift'
        ])

        self.useful_info = np.array(['n_tracks', 'elementID', 'detectorID', 'pid'])

        # Remove 'NaN' from arrays to only load valid branch names
        self.detectors = self.detectors_order[self.detectors_order != 'NaN']
        self.drifts = self.drift_order[self.drift_order != 'NaN']

        # Load branches as a dictionary of numpy arrays
        self.data = self.load_branches()
        self.num_events = len(self.data['n_tracks'])

    def load_branches(self) -> dict:
        """
        Loads the specified branches from the ROOT file.

        Returns:
            dict: A dictionary containing the loaded branches as numpy arrays.
        """
        branch_names = np.concatenate((self.detectors, self.drifts, self.useful_info))
        return self.TTree.arrays(branch_names, library="np")

    def get_num_events(self) -> int:
        """
        Returns the number of events in the file.

        Returns:
            int: The number of events.
        """
        count = self.num_events
        print(f"We have {count} events in this file!")
        return count

    def get_branch_info(self, branch: str, event: int) -> Optional[np.ndarray]:
        """
        Retrieves data for a specific branch and event.

        Args:
            branch (str): The name of the branch.
            event (int): The event index.

        Returns:
            Optional[np.ndarray]: The data for the specified branch and event, or None if the branch is unused.
        """
        if branch == 'NaN':
            return None
        return self.data[branch][event]

    def find_ideal_events(self, event: int) -> bool:
        """
        Determines if an event is ideal based on the number of hits in each station.

        Args:
            event (int): The event index.

        Returns:
            bool: True if the event is ideal, False otherwise.
        """
        n_track = self.data['n_tracks'][event]
        detector_id = self.data['detectorID'][event]
        detector_id = detector_id[detector_id <= 31]

        hits_per_station = n_track * 6

        # Count hits in each station
        st1_count = np.sum(detector_id <= 6)
        st2_count = np.sum((detector_id >= 13) & (detector_id <= 18))
        st3p_count = np.sum((detector_id >= 19) & (detector_id <= 24))
        st3m_count = np.sum((detector_id >= 25) & (detector_id <= 31))

        # Check conditions for an ideal event
        if not (hits_per_station == st1_count == st2_count):
            return False
        return (hits_per_station == st3p_count) or (hits_per_station == st3m_count)

    
    def make_hit_matrix(
        self,
        ideal_events: List[int],
        quality_metric: float = 0.5  # Quality metric (value between 0 and 1)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates separate hit matrices for muon-plus and muon-minus with strict right-side truncation for background simulation.
        Args:
            ideal_events: Valid event indices for processing
            quality_metric: A value between 0 and 1 representing the fraction of detectors to include (cutoff point)
        Returns:
            Tuple containing separate hit matrices for muon-plus and muon-minus,
            along with their respective truth labels and drift values.
        """
        num_events = len(ideal_events)
        
        # Separate hit matrices for mu+ and mu-
        hit_matrix_mup = np.zeros((num_events, 62, 201), dtype=bool)
        hit_matrix_mum = np.zeros((num_events, 62, 201), dtype=bool)
        
        # Truth arrays for element ID and drift values
        truth_elementID_mup = np.zeros((num_events, 62), dtype=np.uint8)
        truth_elementID_mum = np.zeros((num_events, 62), dtype=np.uint8)
        truth_values_drift_mup = np.zeros((num_events, 62))
        truth_values_drift_mum = np.zeros((num_events, 62))

        # Calculate the cutoff detector index based on the quality metric
        total_detectors = len(self.detectors_order[self.detectors_order != 'NaN'])  # Total valid detectors
        cutoff_detector = int(quality_metric * total_detectors)  # Cutoff detector index

        for i, event_idx in enumerate(ideal_events):
            event = int(event_idx)
            pid = self.data['pid'][event]
            n_track = self.data['n_tracks'][event]

            for track in range(n_track):
                for j, detector in enumerate(self.detectors_order):
                    if detector == 'NaN':
                        continue

                    # Skip detectors beyond the cutoff index
                    if j >= cutoff_detector:
                        continue

                    hit_info = self.data[detector][event]
                    drift_variable = self.drift_order[j] if j < len(self.drift_order) else None
                    hit = hit_info[track]

                    # Only include hits within the valid range
                    if 0 <= hit < 201:  # Ensure hit is within the valid range
                        if pid[track] > 0:  # Positive muon
                            truth_elementID_mup[i, j] = hit
                            hit_matrix_mup[i, j, hit] = True
                            if drift_variable:
                                truth_values_drift_mup[i, j] = self.data[drift_variable][event][track]
                        else:  # Negative muon
                            truth_elementID_mum[i, j] = hit
                            hit_matrix_mum[i, j, hit] = True
                            if drift_variable:
                                truth_values_drift_mum[i, j] = self.data[drift_variable][event][track]

        return (
            truth_elementID_mup,
            truth_elementID_mum,
            truth_values_drift_mup,
            truth_values_drift_mum,
            hit_matrix_mup,
            hit_matrix_mum
        )

def load_and_filter_events(root_file: str, max_events: int = 50000) -> np.ndarray:
    """
    Loads the ROOT file, filters ideal events, and returns selected event indices.

    Args:
        root_file (str): Path to the ROOT file.
        max_events (int): Maximum number of events to process.

    Returns:
        np.ndarray: Array of selected ideal event indices.
    """
    print("Loading ROOT file and filtering ideal events...")
    dp = DataProcessing(root_file)
    num_events = dp.get_num_events()

    # Filter ideal events
    ideal_events = [event for event in range(num_events) if dp.find_ideal_events(event)]
    if not ideal_events:
        raise ValueError("No ideal events found in the dataset.")

    # Limit to max_events
    return np.array(ideal_events[:max_events])

def plot_hits(ax, hit_matrix, color: str, label: str):
    """
    Plots individual hits on the given axis.

    Args:
        ax: Matplotlib axis object.
        hit_matrix (np.ndarray): 2D array of hit data.
        color (str): Color for the hits.
        label (str): Label for the legend.
    """
    y, x = np.where(hit_matrix.T > 0)  # Transpose for correct orientation
    ax.scatter(x, y, color=color, label=label, marker='_', s=100, alpha=0.8)

def plot_heatmap(ax, hit_matrix, cmap: str, alpha: float = 0.7):
    """
    Plots a heatmap on the given axis.

    Args:
        ax: Matplotlib axis object.
        hit_matrix (np.ndarray): 2D array of hit data.
        cmap (str): Colormap for the heatmap.
        alpha (float): Transparency level.
    """
    ax.imshow(hit_matrix.T, cmap=cmap, interpolation='none', origin='upper', alpha=alpha)

def plot_heatmap(ax, hit_matrix, cmap: str, alpha: float = 0.7):
    """
    Plots a heatmap on the given axis.
    Args:
        ax: Matplotlib axis object.
        hit_matrix (np.ndarray): 2D array of hit data.
        cmap (str): Colormap for the heatmap.
        alpha (float): Transparency level.
    """
    ax.imshow(hit_matrix.T, cmap=cmap, interpolation='none', origin='upper', alpha=alpha)

def visualize_tracks(hit_matrix_mup, hit_matrix_mum, plot_mode: str = "heatmap"):
    """
    Visualizes the hit matrices for muon plus and muon minus tracks.
    Args:
        hit_matrix_mup (np.ndarray): 3D hit matrix for muon plus (num_events, 62, 201).
        hit_matrix_mum (np.ndarray): 3D hit matrix for muon minus (num_events, 62, 201).
        plot_mode (str): Visualization mode ("hits" or "heatmap").
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Sum over events to get 2D hit matrices
    hit_matrix_mup_2d = np.sum(hit_matrix_mup, axis=0)  # Shape (62, 201)
    hit_matrix_mum_2d = np.sum(hit_matrix_mum, axis=0)  # Shape (62, 201)

    if plot_mode == "heatmap":
        # Heatmap mode
        plot_heatmap(ax, hit_matrix_mup_2d, cmap='Reds', alpha=0.6)
        plot_heatmap(ax, hit_matrix_mum_2d, cmap='Blues', alpha=0.6)

        # Create proxy artists for the legend
        legend_elements = [
            Patch(facecolor=plt.cm.Reds(100), label='Muon Plus', alpha=0.6),
            Patch(facecolor=plt.cm.Blues(100), label='Muon Minus', alpha=0.6)
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12, title="Track Type")

    elif plot_mode == "hits":
        # Hits mode: Plot individual points
        plot_hits(ax, hit_matrix_mup_2d, color='blue', label='Muon Plus')
        plot_hits(ax, hit_matrix_mum_2d, color='red', label='Muon Minus')

        # Add legend for hits mode
        ax.legend(loc='upper right', fontsize=12)

    # Customize plot
    ax.set_xlabel("Detector ID", fontsize=12)
    ax.set_ylabel("Element ID", fontsize=12)
    ax.set_title("Overlay of Muon Tracks", fontsize=14)
    ax.invert_yaxis()  # Flip y-axis for visualization consistency
    ax.set_aspect(0.1)  # Maintain aspect ratio
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)  # Add grid

    plt.tight_layout()
    plt.savefig("Dimuon_Sim.jpeg")  # Save before showing to avoid blank image
    plt.show()
    
    
if __name__ == "__main__":
    # Configuration
    root_file = "Dimuon_target_100K.root"
    max_events = 50000  # Number of events to process
    plot_mode = "heatmap"  # Options: "hits" or "heatmap"
    
    dp = DataProcessing(root_file)

    # Get total number of events in the file
    num_events = dp.get_num_events()

    # # Create an array of ideal events
    # ideal_events = np.zeros(num_events)
    # for event in range(num_events):
    #     if dp.find_ideal_events(event):
    #         ideal_events[event] = event

    # # Filter out zeros (non-ideal events)
    # ideal_events = ideal_events[ideal_events != 0]
    
    ideal_events = [event for event in range(dp.get_num_events()) if dp.find_ideal_events(event)]

    # Select up to `max_events` ideal events
    selected_events = ideal_events[:max_events]
    print(f"There are {len(selected_events)} selected ideal events.")

    # If no ideal events are found, exit early
    if len(selected_events) == 0:
        print("No ideal events found. Exiting.")
        exit()

    # Start timer
    start_time = time()

    # Generate hit matrices using only selected ideal events
    (
        truth_elementID_mup,  
        truth_elementID_mum,  
        truth_values_drift_mup,  
        truth_values_drift_mum,  
        hit_matrix_mup,  
        hit_matrix_mum   
    ) = dp.make_hit_matrix(selected_events)

    # Visualize results using the provided hit matrices
    visualize_tracks(hit_matrix_mup, hit_matrix_mum, plot_mode)

    # Print execution time
    print(f"Execution time: {time() - start_time:.2f} seconds")




# # Example
# start = time()
# #Reads in root file
# root_file = "/home/devin/Documents/Big_Data/Dimuon_Mock/Dimuon_target_100K.root"
# data_processor = DataProcessing(root_file)

# #Set number of events
# num_events = data_processor.get_num_events()

# ideal_events = np.zeros(num_events)
# # Create an array of ideal events
# for event in range(num_events):
#     good_event = data_processor.find_ideal_events(event)
#     if good_event:
#         ideal_events[event] = event
# ideal_events = ideal_events[ideal_events != 0]

# print(f"There are this many ideal events: {len(ideal_events)}")



# #Make hitmatrix
# Truth_elementID_mup, Truth_elementID_mum, Truth_values_drift_mup, Truth_values_drift_mum, hit_matrix = data_processor.make_Hitmatrix(ideal_events)

# np.savez('/home/devin/Documents/Big_Data/Dimuon_Mock/Hit_Info.npz', Truth_elementID_mup, Truth_elementID_mum, Truth_values_drift_mup, Truth_values_drift_mum,hit_matrix,ideal_events)

# #print(ideal_events)


# stop = time()
# print(stop-start)


# # Initialize the plot for combining all scatter plots
# plt.figure(figsize=(10, 6))
# detID = np.arange(1,63)
# n = 1000
# # Loop over all the events with a progress bar using tqdm
# for event in tqdm(ideal_events[:n], desc="Processing Events"):
#     event = int(event)

#     # Get the Truth event index for the current event
#     Truth_event = np.where(ideal_events == event)


#     # Scatter plots for Mup and Mum, overlaying them onto the same figure
#     plt.scatter(detID, Truth_elementID_mup[Truth_event], marker='_', color='r', alpha=0.5)
#     plt.scatter(detID, Truth_elementID_mum[Truth_event], marker='_', color='g', alpha=0.5)

# # Add labels, title, and formatting
# plt.xlim(0, 64)
# plt.ylim(0, 201)
# plt.title("Combined Truth Events")
# plt.xlabel("DetectorID")
# plt.ylabel("ElementID")
# plt.legend(['Mup', 'Mum'])

# # Save and show the combined plot
# plt.savefig(r"/home/devin/Documents/Big_Data/Dimuon_Mock/Combined_Truth_Events.jpeg")
# plt.show()

# stop = time()
# print(f"Processing time: {stop - start} seconds")