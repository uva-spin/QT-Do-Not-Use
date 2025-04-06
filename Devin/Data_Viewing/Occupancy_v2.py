"""
This code is used to plot the occupancy of the detectors in a given run.
It reads in a ROOT file, and plots the occupancy of the detectors in a given run.

The occupancy is plotted as a heatmap, with the opacity of the color representing the number of hits in that detector element.

This is useful for understanding the noise in the data. This overlaps all of that data into a single plot.
"""

import os
import numpy as np
import cupy as cp
import tqdm
import uproot
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as patches

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
        
        # Create output directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        os.makedirs(os.path.join(save_directory, "output_data"), exist_ok=True)

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

        self.create_plot(save_directory)

    def accumulate_hits(self, detectorid, elementid):
        """
        Accumulate hits based on detectorid and elementid.
        Returns a dictionary of (detectorID, elementID) pairs and their corresponding hit counts.
        """
        occupancy = {}
        for i in range(len(detectorid)):
            det_id = detectorid[i]
            elem_id = elementid[i]
            key = (det_id, elem_id)
            occupancy[key] = occupancy.get(key, 0) + 1
        return occupancy
    
    def prepare_heatmap_data(self, occupancy):
        """
        Prepare data for heatmap visualization.
        Returns a dictionary with detector IDs as keys and element occupancy arrays as values.
        """
        detector_occupancy = {}
        
        # Initialize arrays for each detector
        for group in self.detector_groups:
            for detector in group['detectors']:
                detector_id = detector['id']
                elements = detector['elements']
                detector_occupancy[detector_id] = np.zeros(elements)
        
        # Fill in the occupancy data
        for (det_id, elem_id), count in occupancy.items():
            if det_id in detector_occupancy and 0 <= elem_id < len(detector_occupancy[det_id]):
                detector_occupancy[det_id][elem_id] = count
                
        return detector_occupancy
    
    def normalize_detector_data(self, detector_occupancy):
        """
        Normalize hit counts for each detector.
        Returns a dictionary with normalized values (0-1 range).
        """
        normalized_occupancy = {}
        
        for det_id, hits in detector_occupancy.items():
            max_hits = np.max(hits)
            if max_hits > 0:  # Avoid division by zero
                normalized_occupancy[det_id] = hits / max_hits
            else:
                normalized_occupancy[det_id] = hits  # All zeros
                
        return normalized_occupancy
    
    def create_combined_heatmap(self, detector_occupancy, save_directory):
        """
        Create a combined heatmap with dimensions 61×201.
        Maps all detectors to a standardized grid.
        """
        # Create a 61×201 matrix filled with NaN (to distinguish from 0 hits)
        combined_heatmap = np.full((61, 201), np.nan)
        
        # Create a mapping matrix to track detector IDs
        detector_map = np.full((61, 201), -1, dtype=int)
        
        # Create labels matrix
        label_map = np.full((61, 201), "", dtype=object)
        
        # Current row in the combined heatmap
        current_row = 0
        
        # Process each detector group
        for group in self.detector_groups:
            group_height = 0
            
            # Calculate the height needed for this group
            for detector in group['detectors']:
                group_height = max(group_height, len(detector_occupancy.get(detector['id'], [])))
            
            # Skip if no data or would exceed matrix dimensions
            if group_height == 0 or current_row + group_height > 61:
                continue
                
            # Process each detector in the group
            for det_idx, detector in enumerate(group['detectors']):
                det_id = detector['id']
                name = detector['name']
                
                # Get detector data
                det_data = detector_occupancy.get(det_id, np.array([]))
                if len(det_data) == 0:
                    continue
                    
                # Calculate column position
                col_start = det_idx * (201 // len(group['detectors']))
                col_end = min(col_start + len(det_data), 201)
                data_len = col_end - col_start
                
                # Map detector data to the combined heatmap
                data_to_place = det_data[:data_len]
                combined_heatmap[current_row:current_row+1, col_start:col_end] = data_to_place
                
                # Mark detector regions
                detector_map[current_row:current_row+1, col_start:col_end] = det_id
                
                # Set label
                label_map[current_row, col_start] = name
            
            # Move to next row
            current_row += 1
        
        # Create the visualization
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Create mask for NaN values
        mask = np.isnan(combined_heatmap)
        
        # Plot the data with a logarithmic scale
        norm_data = np.ma.array(combined_heatmap, mask=mask)
        cmap = plt.cm.viridis
        cmap.set_bad('white', 1.0)
        
        im = ax.imshow(norm_data, cmap=cmap, aspect='auto', 
                       norm=LogNorm(vmin=0.01, vmax=np.nanmax(combined_heatmap) or 1))
        
        # Add detector labels where there's data
        for i in range(61):
            for j in range(201):
                if label_map[i, j]:
                    ax.text(j, i, label_map[i, j], fontsize=8, 
                            ha='left', va='center', color='black')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Normalized Hit Count")
        
        # Set title and labels
        ax.set_title("Combined Detector Occupancy (61×201)")
        ax.set_xlabel("Element ID (Standardized)")
        ax.set_ylabel("Detector Group (Row)")
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save the combined heatmap
        plt.tight_layout()
        combined_plot_path = os.path.join(save_directory, "combined_heatmap_61x201.png")
        fig.savefig(combined_plot_path, dpi=200)
        print(f"[INFO] Saved combined heatmap to {combined_plot_path}")
        plt.close(fig)
    
    def create_plot(self, save_directory):
        """Create normalized heatmap visualizations with a combined 61×201 matrix."""
        if len(self.aggregated_detectorid) == 0 or len(self.aggregated_elementid) == 0:
            print("[DEBUG] No hits found in the aggregated data. Skipping plot generation.")
            return

        # Transfer aggregated data back to CPU for processing
        detectorid = np.array(self.aggregated_detectorid.get())
        elementid = np.array(self.aggregated_elementid.get())

        # Accumulate hits
        occupancy = self.accumulate_hits(detectorid, elementid)
        
        # Prepare data for heatmap visualization
        detector_occupancy = self.prepare_heatmap_data(occupancy)
        
        # Create the 61×201 combined heatmap
        self.create_combined_heatmap(detector_occupancy, save_directory)
        
        # Normalize detector data for individual plots
        normalized_occupancy = self.normalize_detector_data(detector_occupancy)
        
        # Find global maximum for reference
        max_hits = max(occupancy.values()) if occupancy else 1
        
        # Create individual group plots with normalized data
        for group_idx, group in enumerate(tqdm.tqdm(self.detector_groups, desc="Creating normalized group plots")):
            # Skip groups with no data
            if all(np.sum(detector_occupancy.get(detector['id'], np.array([]))) == 0 for detector in group['detectors']):
                continue
                
            # Create figure for this group
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Layout parameters
            x_offset = 0
            detector_width = 1.0
            group_data = []
            x_labels = []
            x_positions = []
            
            # Plot each detector in the group
            for detector_idx, detector in enumerate(group['detectors']):
                detector_id = detector['id']
                name = detector['name']
                elements = detector['elements']
                
                # Get normalized occupancy data for this detector
                norm_data = normalized_occupancy.get(detector_id, np.zeros(elements))
                
                # Create image for this detector with normalized values
                im = ax.imshow(
                    norm_data.reshape(-1, 1),
                    extent=[x_offset, x_offset + detector_width, 0, elements],
                    aspect='auto',
                    origin='lower',
                    cmap='viridis',
                    vmin=0,
                    vmax=1  # Normalized range is 0-1
                )
                
                # Add detector information to labels
                x_labels.append(name)
                x_positions.append(x_offset + detector_width/2)
                
                # Collect data for JSON (store both raw and normalized)
                raw_data = detector_occupancy.get(detector_id, np.zeros(elements))
                for elem_id in range(elements):
                    if raw_data[elem_id] > 0:
                        group_data.append({
                            "detector": name,
                            "element": elem_id,
                            "hits": int(raw_data[elem_id]),
                            "normalized": float(norm_data[elem_id])
                        })
                
                # Add borders between detectors
                ax.add_patch(
                    patches.Rectangle(
                        (x_offset, 0),
                        detector_width,
                        elements,
                        fill=False,
                        edgecolor='black',
                        linewidth=0.5
                    )
                )
                
                # Update x_offset for next detector
                x_offset += detector_width + 0.2  # Add small gap between detectors
            
            # Add labels and title
            ax.set_title(f"{group['label']} - Normalized Occupancy")
            ax.set_xlabel("Detector")
            ax.set_ylabel("Element ID")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45, ha="right")
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Normalized Hit Count (0-1)")
            
            # Save group plot
            plt.tight_layout()
            group_plot_path = os.path.join(save_directory, "output_data", f"{group['label'].replace(' ', '_')}_normalized.png")
            fig.savefig(group_plot_path, dpi=150)
            plt.close(fig)
            
            # Save group data as JSON
            group_data_path = os.path.join(save_directory, "output_data", f"{group['label'].replace(' ', '_')}_normalized_data.json")
            with open(group_data_path, "w") as f:
                json.dump(group_data, f, indent=2)
        
        # Create overall visualization with clear group separation
        self.create_overall_visualization(normalized_occupancy, detector_occupancy, save_directory)
    
    def create_overall_visualization(self, normalized_occupancy, raw_occupancy, save_directory):
        """Create an overall visualization with normalized values."""
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Layout parameters
        x_offset = 0
        detector_width = 0.8
        group_gap = 2.0  # Gap between groups
        
        x_labels = []
        x_positions = []
        group_boundaries = []
        current_group = ""
        
        # Create rectangular heatmaps for each detector
        for group_idx, group in enumerate(self.detector_groups):
            # Skip empty groups for clarity
            if all(np.sum(raw_occupancy.get(detector['id'], np.array([]))) == 0 for detector in group['detectors']):
                continue
                
            # Mark group boundaries
            if group['label'] != current_group:
                group_boundaries.append(x_offset)
                current_group = group['label']
            
            # Process each detector in the group
            for detector_idx, detector in enumerate(group['detectors']):
                detector_id = detector['id']
                name = detector['name']
                elements = detector['elements']
                
                # Get normalized occupancy data
                norm_data = normalized_occupancy.get(detector_id, np.zeros(elements))
                
                # Create heatmap for this detector with normalized values
                im = ax.imshow(
                    norm_data.reshape(-1, 1),
                    extent=[x_offset, x_offset + detector_width, 0, elements],
                    aspect='auto',
                    origin='lower',
                    cmap='viridis',
                    vmin=0,
                    vmax=1  # Normalized range
                )
                
                # Store position for labels
                x_labels.append(name)
                x_positions.append(x_offset + detector_width/2)
                
                # Add border
                ax.add_patch(
                    patches.Rectangle(
                        (x_offset, 0),
                        detector_width,
                        elements,
                        fill=False,
                        edgecolor='black',
                        linewidth=0.5
                    )
                )
                
                # Move to next position
                x_offset += detector_width + 0.2
            
            # Add gap after group
            x_offset += group_gap
        
        # Add last group boundary
        group_boundaries.append(x_offset)
        
        # Add group labels at the top
        current_label = ""
        for i, group in enumerate(self.detector_groups):
            if group['label'] != current_label and i < len(group_boundaries) - 1:
                midpoint = (group_boundaries[i] + group_boundaries[i+1]) / 2
                ax.text(midpoint, ax.get_ylim()[1] * 1.05, group['label'], 
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
                current_label = group['label']
        
        # Add labels and title
        ax.set_title("Overall Normalized Detector Occupancy")
        ax.set_xlabel("Detector")
        ax.set_ylabel("Element ID")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=90, fontsize=8)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Normalized Hit Count (0-1)")
        
        # Add vertical lines to separate groups
        for boundary in group_boundaries[1:-1]:  # Skip first and last
            ax.axvline(x=boundary - group_gap/2, color='gray', linestyle='--', alpha=0.7)
        
        # Save overall plot
        plt.tight_layout()
        overall_plot_path = os.path.join(save_directory, "overall_normalized_occupancy.png")
        fig.savefig(overall_plot_path, dpi=200)
        print(f"[INFO] Saved overall normalized plot to {overall_plot_path}")
        plt.close(fig)

if __name__ == "__main__":
    run_directory = r"Devin\Data_Viewing"
    save_directory = r"Devin\Data_Viewing\QTracker_Station_Occupancies"
    os.makedirs(save_directory, exist_ok=True)
    max_files = 1  # For Debugging
    detector_plot = DetectorPlot()
    detector_plot.process_files(run_directory, save_directory, max_files=max_files)
    print(f"Plots saved at: {save_directory}")