import os
import numpy as np
import cupy as cp
import tqdm
import uproot
import json
import matplotlib.pyplot as plt

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
    
    def process_files(self, run_directory,save_directory,max_files=None):
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

        self.create_plot(save_directory)

    def accumulate_hits(self,detectorid, elementid):
        """
        Accumulate hits based on detectorid and elementid using Numba for speed.
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
    
    def create_plot(self,save_directory):
        """Create a final aggregated plot with occupancy indicated by opacity, save group data, and generate individual plots."""
        if len(self.aggregated_detectorid) == 0 or len(self.aggregated_elementid) == 0:
            print("[DEBUG] No hits found in the aggregated data. Skipping plot generation.")
            return

        # Transfer aggregated data back to CPU for matplotlib
        detectorid = np.array(self.aggregated_detectorid.get())  # Convert to numpy arrays for numba compatibility
        elementid = np.array(self.aggregated_elementid.get())

        # Accumulate hits using Numba
        occupancy = self.accumulate_hits(detectorid, elementid)

        # Normalize occupancy values for opacity (0 to 1)
        max_hits = max(occupancy.values())
        normalized_opacity = {key: value / max_hits for key, value in occupancy.items()}

        fig, ax = plt.subplots(figsize=(12, 8))
        y_max = max([det['elements'] for group in self.detector_groups for det in group['detectors']])
        x_labels = []
        x_ticks = []
        x_offset = 0
        print("Plotting")

        # Wrapping the outer loop with tqdm for progress bar
        for group in tqdm.tqdm(self.detector_groups, desc="Processing detector groups", unit="group"):
            x_positions = range(x_offset, x_offset + len(group['detectors']))
            group_data = []  # Collect data to save for this group

            # Create a plot for the group
            group_fig, group_ax = plt.subplots(figsize=(12, 8))

            for idx, x in enumerate(x_positions):
                detector = group['detectors'][idx]
                y_positions = range(detector['elements'])

                for y in y_positions:
                    scaled_y = y * y_max / detector['elements']
                    height = y_max / detector['elements']

                    hit_opacity = normalized_opacity.get((detector['id'], y + 1), 0)  # Default opacity is 0
                    facecolor = (1, 0, 0, hit_opacity)  # RGBA: Yellow with variable alpha based on opacity

                    # Add the rectangle to the overall plot
                    ax.add_patch(plt.Rectangle((x, scaled_y), 1, height, edgecolor='black', facecolor=facecolor))

                    # Add the rectangle to the group-specific plot
                    group_ax.add_patch(plt.Rectangle((idx, scaled_y), 1, height, edgecolor='black', facecolor=facecolor))

                    # Collect data for this detector and element
                    group_data.append({
                        'detector_id': detector['id'],
                        'detector_name': detector['name'],
                        'element_id': y + 1,
                        'opacity': hit_opacity
                    })

                x_labels.append(detector['name'])
                x_ticks.append(x + 0.5)

            ax.text(x_offset + len(group['detectors']) / 2, y_max + 15, group['label'], ha='center', va='center')
            x_offset += len(group['detectors']) + 2

            # Save the group data to a file (e.g., JSON or CSV)
            group_filename = f"{group['label']}_data.json"  # Change the extension for different formats
            group_file_path = os.path.join(save_directory,"output_data", group['label'], group_filename)

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(group_file_path), exist_ok=True)

            # Save the data in JSON format
            with open(group_file_path, 'w') as json_file:
                json.dump(group_data, json_file, indent=4)

            # Finalize and save the group-specific plot
            group_ax.set_xlim(0, len(group['detectors']))
            group_ax.set_ylim(0, y_max + 20)
            group_ax.set_xticks(range(len(group['detectors'])))
            group_ax.set_xticklabels([det['name'] for det in group['detectors']], rotation=90)
            group_ax.set_ylabel('Element ID')
            plt.tight_layout()
            group_plot_path = os.path.join(save_directory,"output_data", group['label'], f"{group['label']}_plot.png")
            group_fig.savefig(group_plot_path)
            plt.close(group_fig)

        # Finalize and save the overall plot
        ax.set_xlim(0, x_offset)
        ax.set_ylim(0, y_max + 20)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_ylabel('Element ID')
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory,"final_detector_plot_gpu_opacity.png"))
        plt.close(fig)

if __name__ == "__main__":
    run_directory = r"/home/devin/Documents/Big_Data/run_005994"
    save_directory = r"/home/devin/Documents/Big_Data/QTracker_Station_Occupancies"
    max_files = 1  # For Debugging
    detector_plot = DetectorPlot()
    detector_plot.process_files(run_directory,save_directory, max_files=max_files)
    print("Plot saved as 'final_detector_plot.png' at: {save_directory}")
