import os
import numpy as np
import cupy as cp
import tqdm
import uproot
import json
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

class DetectorPlot:
    def __init__(self):
        self.file_paths = []
        self.current_file_index = 0
        self.event_number = 0
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
    
    def read_event(self, file_path, event_number):
        """Read a specific event from the given ROOT file."""
        with uproot.open(file_path + ":save") as file:
            detectorid = cp.array(file["fAllHits.detectorID"].array(library="np")[event_number])
            elementid = cp.array(file["fAllHits.elementID"].array(library="np")[event_number])
        return detectorid, elementid
    
    def process_files(self, run_directory, save_directory, max_files=None):
        """Process all events across the first `max_files` ROOT files in the given directory."""
        self.aggregated_detectorid = cp.array([], dtype=int)
        self.aggregated_elementid = cp.array([], dtype=int)

        # Gather all ROOT files in the directory
        for root, _, files in os.walk(run_directory):
            for file in files:
                if file.endswith(".root"):
                    self.file_paths.append(os.path.join(root, file))

        if not self.file_paths:
            print("[DEBUG] No ROOT files found in the specified directory.")
            return

        # Limit to the first `max_files` files if specified
        if max_files is not None:
            self.file_paths = self.file_paths[:max_files]

        # Calculate total events for the progress bar
        total_events = 0
        for file_path in self.file_paths:
            with uproot.open(file_path + ":save") as file:
                total_events += len(file["fAllHits.detectorID"].array(library="np"))

        # Process events with a progress bar
        with tqdm.tqdm(total=total_events, desc="Processing Events", unit="event") as pbar:
            for file_path in self.file_paths:
                with uproot.open(file_path + ":save") as file:
                    num_events = len(file["fAllHits.detectorID"].array(library="np"))
                    for event_number in range(num_events):
                        detectorid, elementid = self.read_event(file_path, event_number)
                        self.aggregated_detectorid = cp.concatenate((self.aggregated_detectorid, detectorid))
                        self.aggregated_elementid = cp.concatenate((self.aggregated_elementid, elementid))
                        pbar.update(1)

        # Debug message if no hits are found
        if len(self.aggregated_detectorid) == 0 or len(self.aggregated_elementid) == 0:
            print("[DEBUG] No hits found across all events in the ROOT files.")
            print("[DEBUG] Check if the ROOT files contain valid detector and element data.")
        else:
            print(f"[DEBUG] Aggregated {len(self.aggregated_detectorid)} detector hits and {len(self.aggregated_elementid)} element hits.")

        self.create_plots(save_directory)

    def accumulate_hits(self, detectorid, elementid):
        """
        Accumulate hits based on detectorid and elementid.
        Returns a dictionary of (detectorID, elementID) pairs and their corresponding hit counts.
        """
        occupancy = {}
        for i in range(len(detectorid)):
            det_id = detectorid[i]
            elem_id = elementid[i]
            if (det_id, elem_id) not in occupancy:
                occupancy[(det_id, elem_id)] = 0
            occupancy[(det_id, elem_id)] += 1
        return occupancy
    
    def create_plots(self, save_directory):
        """Create multiple versions of plots with detector occupancy."""
        if len(self.aggregated_detectorid) == 0 or len(self.aggregated_elementid) == 0:
            print("[DEBUG] No hits found in the aggregated data. Skipping plot generation.")
            return

        # Make sure output directories exist
        os.makedirs(save_directory, exist_ok=True)
        output_data_dir = os.path.join(save_directory, "output_data")
        os.makedirs(output_data_dir, exist_ok=True)

        # Transfer aggregated data back to CPU for matplotlib
        detectorid = np.array(self.aggregated_detectorid.get())
        elementid = np.array(self.aggregated_elementid.get())

        # Accumulate hits
        occupancy = self.accumulate_hits(detectorid, elementid)

        # Find max hits for normalization
        max_hits = max(occupancy.values()) if occupancy else 1
        norm = Normalize(vmin=0, vmax=max_hits)

        # Calculate total layout information
        y_max = max([det['elements'] for group in self.detector_groups for det in group['detectors']])
        
        # Prepare data for all plots
        x_labels = []
        x_ticks = []
        x_offset = 0
        all_box_data = []  # Will store all box data for the plots
        separators = []    # Will store positions for separator lines
        
        # First pass: collect all data
        print("Collecting data for plots...")
        for group_idx, group in enumerate(self.detector_groups):
            group_data = []  # Data for this group's JSON file
            
            for det_idx, detector in enumerate(group['detectors']):
                x_pos = x_offset + det_idx
                
                for elem_id in range(detector['elements']):
                    hit_count = occupancy.get((detector['id'], elem_id), 0)
                    
                    if hit_count > 0:
                        # Store box data: x position, y position, hit count
                        all_box_data.append((x_pos, elem_id, hit_count))
                    
                    # Add to group data for JSON export
                    group_data.append({
                        "detector": detector['name'], 
                        "element": elem_id, 
                        "hits": hit_count
                    })
                
                x_labels.append(detector['name'])
                x_ticks.append(x_pos)
            
            # Save individual group data
            group_save_path = os.path.join(output_data_dir, f"{group['label'].replace(' ', '_')}_data.json")
            with open(group_save_path, "w") as group_file:
                json.dump(group_data, group_file, indent=4)
                
            # Add a separator position after this group
            if group_idx < len(self.detector_groups) - 1:
                separators.append(x_offset + len(group['detectors']) - 0.5)
            
            # Update x offset for the next group
            x_offset += len(group['detectors'])
        
        # Create and save individual group plots
        self._create_group_plots(all_box_data, x_labels, x_ticks, y_max, norm, max_hits, separators, output_data_dir)
        
        # Create and save overall plots (with and without separators)
        self._create_overall_plots(all_box_data, x_labels, x_ticks, y_max, norm, max_hits, separators, save_directory)
        
        print(f"[INFO] All plots saved in directory: {save_directory}")

    def _create_group_plots(self, all_box_data, x_labels, x_ticks, y_max, norm, max_hits, separators, output_dir):
        """Create individual plots for each detector group."""
        print("Creating group plots...")
        
        # Keep track of which detectors belong to which group
        group_start_idx = 0
        
        for group_idx, group in enumerate(self.detector_groups):
            # Create figure for this group
            group_fig, group_ax = plt.subplots(figsize=(12, 8))
            
            # Get range of x positions for this group
            group_end_idx = group_start_idx + len(group['detectors'])
            group_x_ticks = x_ticks[group_start_idx:group_end_idx]
            group_x_labels = x_labels[group_start_idx:group_end_idx]
            
            # Filter box data for this group
            group_box_data = [(x, y, count) for x, y, count in all_box_data 
                             if group_start_idx <= x < group_end_idx]
            
            # Plot boxes for this group
            self._plot_boxes(group_ax, group_box_data, norm)
            
            # Customize the group plot
            group_ax.set_title(f"{group['label']} Detector Occupancy", fontsize=14, pad=10)
            group_ax.set_xlabel("Detector", fontsize=12)
            group_ax.set_ylabel("Element", fontsize=12)
            group_ax.set_xticks([x - group_start_idx for x in group_x_ticks])  # Shift x ticks to start from 0
            group_ax.set_xticklabels(group_x_labels, rotation=90)
            group_ax.set_ylim(0, y_max)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
            sm.set_array([])
            cbar = group_fig.colorbar(sm, ax=group_ax, pad=0.02)
            cbar.set_label("Hit Count", rotation=90, labelpad=15, fontsize=12)
            
            plt.tight_layout()
            
            # Save group plot
            group_plot_path = os.path.join(output_dir, f"{group['label'].replace(' ', '_')}_plot.png")
            group_fig.savefig(group_plot_path, dpi=150, bbox_inches='tight')
            plt.close(group_fig)
            print(f"[INFO] Saved group plot to {group_plot_path}")
            
            # Update for next group
            group_start_idx = group_end_idx

    def _create_overall_plots(self, all_box_data, x_labels, x_ticks, y_max, norm, max_hits, separators, save_dir):
        """Create overall plots with and without separators."""
        print("Creating overall plots...")
        
        # Version 1: With separators between detector groups
        fig_sep, ax_sep = plt.subplots(figsize=(16, 10))
        self._plot_boxes(ax_sep, all_box_data, norm)
        
        # Add separator lines
        for sep_pos in separators:
            ax_sep.axvline(x=sep_pos, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)
        
        self._customize_overall_plot(fig_sep, ax_sep, "Overall Detector Occupancy (With Group Separators)", 
                                   x_labels, x_ticks, y_max, norm)
        overall_plot_path_sep = os.path.join(save_dir, "overall_occupancy_with_separators.png")
        fig_sep.savefig(overall_plot_path_sep, dpi=150, bbox_inches='tight')
        plt.close(fig_sep)
        print(f"[INFO] Saved overall plot with separators to {overall_plot_path_sep}")
        
        # Version 2: Without separators
        fig_no_sep, ax_no_sep = plt.subplots(figsize=(16, 10))
        self._plot_boxes(ax_no_sep, all_box_data, norm)
        self._customize_overall_plot(fig_no_sep, ax_no_sep, "Overall Detector Occupancy (Without Separators)", 
                                   x_labels, x_ticks, y_max, norm)
        overall_plot_path_no_sep = os.path.join(save_dir, "overall_occupancy_no_separators.png")
        fig_no_sep.savefig(overall_plot_path_no_sep, dpi=150, bbox_inches='tight')
        plt.close(fig_no_sep)
        print(f"[INFO] Saved overall plot without separators to {overall_plot_path_no_sep}")

    def _plot_boxes(self, ax, box_data, norm):
        """Plot boxes representing hits on the given axis."""
        for x, y, count in box_data:
            # Calculate color based on hit count
            color_intensity = norm(count)
            color = plt.cm.Reds(color_intensity)
            
            # Draw rectangle with no edge
            rect = Rectangle((x - 0.5, y - 0.5), 1, 1, 
                           facecolor=color, 
                           edgecolor='none',
                           alpha=0.9)
            ax.add_patch(rect)

    def _customize_overall_plot(self, fig, ax, title, x_labels, x_ticks, y_max, norm):
        """Apply customizations to an overall plot."""
        ax.set_title(title, fontsize=16, pad=15)
        ax.set_xlabel("Detector", fontsize=14)
        ax.set_ylabel("Element", fontsize=14)
        ax.set_xlim(-0.5, len(x_ticks) - 0.5)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_ylim(0, y_max)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Hit Count", rotation=90, labelpad=15, fontsize=14)
        
        plt.tight_layout()

if __name__ == "__main__":
    run_directory = r"/home/devin/Documents/Big_Data/run_005994"
    save_directory = r"/home/devin/Documents/Big_Data/QTracker_Station_Occupancies"
    max_files = 1  # For Debugging
    detector_plot = DetectorPlot()
    detector_plot.process_files(run_directory, save_directory, max_files=max_files)