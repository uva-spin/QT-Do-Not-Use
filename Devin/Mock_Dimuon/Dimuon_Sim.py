"""
This code is used to simulate a dimuon event in the target.
It reads in a ROOT file, filters ideal events, and organizes data into
a hit matrix and ground truth labels. The goal is to preprocess the data for use in a DNN. In this case,
we are implementing "cuts" to the hit matrix to create background that we can use for training.

Further addition to the background is made in Generate_Background.py, which adds slightly more sophisticated background simulation.

Keep in mind that a MAJORITY of the noise we see in data events is due to partial muon tracks (ones that don't go through the entire detector, up to some cut-off).
"""

import numpy as np
import uproot
import matplotlib.pyplot as plt
import seaborn as sns
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
        
        self.detector_groups = [
        {'label': 'Station 1', 'detectors': [
            {'name': 'D0V', 'elements': 201, 'id': 5},
            {'name': 'D0Vp', 'elements': 201, 'id': 6},
            {'name': 'D0Xp', 'elements': 160, 'id': 4},
            {'name': 'D0X', 'elements': 160, 'id': 3},
            {'name': 'D0U', 'elements': 201, 'id': 1},
            {'name': 'D0Up', 'elements': 201, 'id': 2}
        ]},
        {'label': 'Hodo', 'detectors': [
            {'name': 'H1L', 'elements': 20, 'id': 33},
            {'name': 'H1R', 'elements': 20, 'id': 34},
            {'name': 'H1B', 'elements': 23, 'id': 31},
            {'name': 'H1T', 'elements': 23, 'id': 32}
        ]},
        {'label': 'DP-1', 'detectors': [
            {'name': 'DP1TL', 'elements': 80, 'id': 55},
            {'name': 'DP1TR', 'elements': 80, 'id': 56},
            {'name': 'DP1BL', 'elements': 80, 'id': 57},
            {'name': 'DP1BR', 'elements': 80, 'id': 58}
        ]},
        {'label': 'Station 2', 'detectors': [
            {'name': 'D2V', 'elements': 128, 'id': 13},
            {'name': 'D2Vp', 'elements': 128, 'id': 14},
            {'name': 'D2Xp', 'elements': 112, 'id': 15},
            {'name': 'D2X', 'elements': 112, 'id': 16},
            {'name': 'D2U', 'elements': 128, 'id': 17},
            {'name': 'D2Up', 'elements': 128, 'id': 18}
        ]},
        {'label': 'Hodo', 'detectors': [
            {'name': 'H2R', 'elements': 19, 'id': 36},
            {'name': 'H2L', 'elements': 19, 'id': 35},
            {'name': 'H2T', 'elements': 16, 'id': 38},
            {'name': 'H2B', 'elements': 16, 'id': 37}
        ]},
        {'label': 'DP-2', 'detectors': [
            {'name': 'DP2TL', 'elements': 48, 'id': 59},
            {'name': 'DP2TR', 'elements': 48, 'id': 60},
            {'name': 'DP2BL', 'elements': 48, 'id': 61},
            {'name': 'DP2BR', 'elements': 48, 'id': 62}
        ]},
        {'label': 'Station 3+', 'detectors': [
            {'name': 'D3pVp', 'elements': 134, 'id': 19},
            {'name': 'D3pV', 'elements': 134, 'id': 20},
            {'name': 'D3pXp', 'elements': 116, 'id': 21},
            {'name': 'D3pX', 'elements': 116, 'id': 22},
            {'name': 'D3pUp', 'elements': 134, 'id': 23},
            {'name': 'D3pU', 'elements': 134, 'id': 24}
        ]},
        {'label': 'Station 3-', 'detectors': [
            {'name': 'D3mVp', 'elements': 134, 'id': 25},
            {'name': 'D3mV', 'elements': 134, 'id': 26},
            {'name': 'D3mXp', 'elements': 116, 'id': 27},
            {'name': 'D3mX', 'elements': 116, 'id': 28},
            {'name': 'D3mUp', 'elements': 134, 'id': 29},
            {'name': 'D3mU', 'elements': 134, 'id': 30}
        ]},
        {'label': 'Hodo', 'detectors': [
            {'name': 'H3T', 'elements': 16, 'id': 40},
            {'name': 'H3B', 'elements': 16, 'id': 39}
        ]},
        {'label': 'Prop', 'detectors': [
            {'name': 'P1Y1', 'elements': 72, 'id': 47},
            {'name': 'P1Y2', 'elements': 72, 'id': 48}
        ]},
        {'label': 'Hodo', 'detectors': [
            {'name': 'H4Y1R', 'elements': 16, 'id': 42},
            {'name': 'H4Y1L', 'elements': 16, 'id': 41}
        ]},
        {'label': 'Prop', 'detectors': [
            {'name': 'P1X1', 'elements': 72, 'id': 49},
            {'name': 'P1X2', 'elements': 72, 'id': 50}
        ]},
        {'label': 'Hodo', 'detectors': [
            {'name': 'H4Y2R', 'elements': 16, 'id': 44},
            {'name': 'H4Y2L', 'elements': 16, 'id': 43},
            {'name': 'H4T', 'elements': 16, 'id': 46},
            {'name': 'H4B', 'elements': 16, 'id': 45}
        ]},
        {'label': 'Prop', 'detectors': [
            {'name': 'P2X1', 'elements': 72, 'id': 51},
            {'name': 'P2X2', 'elements': 72, 'id': 52},
            {'name': 'P2Y1', 'elements': 72, 'id': 53},
            {'name': 'P2Y2', 'elements': 72, 'id': 54}
        ]}
        ]

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

        self.detectors = self.detectors_order[self.detectors_order != 'NaN']
        self.drifts = self.drift_order[self.drift_order != 'NaN']

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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates separate hit matrices for muon-plus and muon-minus with strict right-side truncation for background simulation.
        Args:
            ideal_events: Valid event indices for processing
        Returns:
            Tuple containing separate hit matrices for muon-plus and muon-minus,
            along with their respective truth labels and drift values.
        """
        num_events = len(ideal_events)
        
        # Separate hit matrices for mu+ and mu-
        # We're just creating empty hit matrices for now to populate later
        hit_matrix_mup = np.zeros((num_events, 62, 201), dtype=bool)
        hit_matrix_mum = np.zeros((num_events, 62, 201), dtype=bool)
        
        # Truth arrays for element ID and drift values
        # We're just creating empty truth arrays for now to populate later
        truth_elementID_mup = np.zeros((num_events, 62), dtype=np.uint8)
        truth_elementID_mum = np.zeros((num_events, 62), dtype=np.uint8)
        truth_values_drift_mup = np.zeros((num_events, 62))
        truth_values_drift_mum = np.zeros((num_events, 62))

        # Here, we're constructing the hit matrices from the event files
        for i, event_idx in enumerate(ideal_events):
            event = int(event_idx)
            pid = self.data['pid'][event]
            n_track = self.data['n_tracks'][event]

            for track in range(n_track):
                for j, detector in enumerate(self.detectors_order):
                    if detector == 'NaN':
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
            hit_matrix_mup, ### This is the hit matrix for muon plus
            hit_matrix_mum ### This is the hit matrix for muon minus
            ### These two matrices are our main interest right now
        )

    def load_and_filter_events(self,root_file: str, max_events: int = 50000) -> np.ndarray:
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

        ideal_events = [event for event in range(num_events) if dp.find_ideal_events(event)]
        if not ideal_events:
            raise ValueError("No ideal events found in the dataset.")

        return np.array(ideal_events[:max_events])

    def plot_hits(self, ax, hit_matrix, color: str, label: str):
        """
        Plots individual hits using matplotlib

        Args:
            ax: Matplotlib axis object.
            hit_matrix (np.ndarray): 2D array of hit data.
            color (str): Color for the hits.
            label (str): Label for the legend.
        """
        y, x = np.where(hit_matrix.T > 0)  
        ax.scatter(x, y, color=color, label=label, marker='_', s=100, alpha=0.8)

    def plot_heatmap(self, ax, hit_matrix, cmap: str, alpha: float = 0.7):
        """
        Plots a heatmap of hit matrix using matplotlib

        Args:
            ax: Matplotlib axis object.
            hit_matrix (np.ndarray): 2D array of hit data.
            cmap (str): Colormap for the heatmap.
            alpha (float): Transparency level.
        """
        heatmap = ax.imshow(hit_matrix.T, cmap=plt.cm.get_cmap(cmap), 
                            alpha=alpha, aspect='auto', origin='upper')
        return heatmap

    def visualize_tracks(self, hit_matrix_mup, hit_matrix_mum, plot_mode: str = "heatmap"):
        """
        Visualizes the hit matrices for muon plus and muon minus tracks using matplotlib.
        
        Args:
            hit_matrix_mup (np.ndarray): 3D hit matrix for muon plus (num_events, 62, 201).
            hit_matrix_mum (np.ndarray): 3D hit matrix for muon minus (num_events, 62, 201).
            plot_mode (str): Visualization mode ("hits" or "heatmap").
        """
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Sum over events to get 2D hit matrices
        hit_matrix_mup_2d = np.sum(hit_matrix_mup, axis=0)  # Shape (62, 201)
        hit_matrix_mum_2d = np.sum(hit_matrix_mum, axis=0)  # Shape (62, 201)

        if plot_mode == "heatmap":
            # Heatmap mode
            mup_heatmap = self.plot_heatmap(ax, hit_matrix_mup_2d, cmap='Reds', alpha=0.9)
            mum_heatmap = self.plot_heatmap(ax, hit_matrix_mum_2d, cmap='Blues', alpha=0.4)

            legend_elements = [
                plt.Line2D([0], [0], marker='s', color='w', 
                        markerfacecolor='red', markersize=15, alpha=0.9, 
                        label='Muon Plus'),
                plt.Line2D([0], [0], marker='s', color='w', 
                        markerfacecolor='blue', markersize=15, alpha=0.9, 
                        label='Muon Minus')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=12, 
                    title="Track Type", title_fontsize=14, frameon=True, 
                    fancybox=True, shadow=True)

        elif plot_mode == "hits":
            # Hits mode: Plot individual points
            self.plot_hits(ax, hit_matrix_mup_2d, color='red', label='Muon Plus')
            self.plot_hits(ax, hit_matrix_mum_2d, color='blue', label='Muon Minus')
            ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)

        all_detectors = []
        for group in self.detector_groups:
            for detector in group['detectors']:
                all_detectors.append(detector)

        tick_positions = np.arange(len(all_detectors))  # Positions based on the order
        tick_labels = [detector['name'] for detector in all_detectors]  # Labels based on detector names

        separators = []
        current_position = 0
        for group_idx, group in enumerate(self.detector_groups):
            current_position += len(group['detectors'])
            if group_idx < len(self.detector_groups) - 1:
                separators.append(current_position - 0.5)

        # Draw separator lines if in heatmap mode
        if plot_mode == "heatmap":
            for sep_pos in separators:
                ax.axvline(x=sep_pos, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)

        ax.set_title("Overlay of Muon Tracks", fontsize=16, pad=15)
        ax.set_xlabel("Detector", fontsize=14)
        ax.set_ylabel("Element", fontsize=14)

        ax.set_xlim(-0.5, len(tick_positions) - 0.5)
        ax.set_ylim(0, 202)  

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90)

        ax.set_yticks(np.arange(0, 202, 20))

        plt.tight_layout()

        plt.savefig("Dimuon_Sim.jpeg", dpi=600, bbox_inches='tight')
        plt.show()
    
    
if __name__ == "__main__":
    # Configuration
    root_file = "Devin/Mock_Dimuon/Data_Files/Dimuon_target_100K.root"
    max_events = 1  # Number of events to process
    plot_mode = "hits"  # Options: "hits" or "heatmap"
    
    dp = DataProcessing(root_file)

    # Get total number of events in the file
    num_events = dp.get_num_events()

    ### Let's skip this for now

    # # Create an array of ideal events
    # ideal_events = np.zeros(num_events)
    # for event in range(num_events):
    #     if dp.find_ideal_events(event):
    #         ideal_events[event] = event

    # # Filter out zeros (non-ideal events)
    # ideal_events = ideal_events[ideal_events != 0]
    
    ideal_events = [event for event in range(dp.get_num_events()) if dp.find_ideal_events(event)]

    selected_events = ideal_events[:max_events]
    print(f"There are {len(selected_events)} selected ideal events.")

    if len(selected_events) == 0:
        print("No ideal events found. Exiting.")
        exit()

    start_time = time()
    


    (
        truth_elementID_mup,  
        truth_elementID_mum,  
        truth_values_drift_mup,  
        truth_values_drift_mum,  
        hit_matrix_mup,  
        hit_matrix_mum   
    ) = dp.make_hit_matrix(selected_events)

    dp.visualize_tracks(hit_matrix_mup, hit_matrix_mum, plot_mode)

    print(f"Execution time: {time() - start_time:.2f} seconds")
