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
    
    def create_plot(self, save_directory):
      """Create a final aggregated plot with occupancy indicated by opacity, save group data, and generate individual plots."""
      if len(self.aggregated_detectorid) == 0 or len(self.aggregated_elementid) == 0:
          print("[DEBUG] No hits found in the aggregated data. Skipping plot generation.")
          return

      # Transfer aggregated data back to CPU for matplotlib
      detectorid = np.array(self.aggregated_detectorid.get())  # Convert to numpy arrays for numba compatibility
      elementid = np.array(self.aggregated_elementid.get())

      # Accumulate hits using Numba
      occupancy = self.accumulate_hits(detectorid, elementid)

      # # Save hit information as JSON
      # output_data = []
      # for (det_id, elem_id), hit_count in occupancy.items():
      #     output_data.append({
      #         "detector_id": int(det_id),
      #         "element_id": int(elem_id),
      #         "hit_count": int(hit_count)
      #     })
      # json_file_path = os.path.join(save_directory, "hits_data.json")
      # with open(json_file_path, "w") as json_file:
      #     json.dump(output_data, json_file, indent=4)
      # print(f"[INFO] Saved hit information to {json_file_path}")

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
          group_fig, group_ax = plt.subplots(figsize=(12, 6))
          for x, detector in zip(x_positions, group['detectors']):
              detector_hits = []
              for elem_id in range(detector['elements']):
                  hit_count = occupancy.get((detector['id'], elem_id), 0)
                  opacity = normalized_opacity.get((detector['id'], elem_id), 0)
                  group_data.append({"detector": detector['name'], "element": elem_id, "hits": hit_count})
                  if hit_count > 0:
                      ax.scatter(x, elem_id, color="red", alpha=opacity)
                      detector_hits.append({"element": elem_id, "hits": hit_count})
              x_labels.append(detector['name'])
              x_ticks.append(x)

          # Save individual group data
          group_save_path = os.path.join(save_directory, "output_data", f"{group['label']}_data.json")
          with open(group_save_path, "w") as group_file:
              json.dump(group_data, group_file, indent=4)

          # Customize the group plot
          group_ax.set_title(group['label'])
          group_ax.set_xlabel("Detector")
          group_ax.set_ylabel("Element")
          group_ax.set_xticks(x_ticks)
          group_ax.set_xticklabels(x_labels, rotation=90)
          group_ax.set_ylim(0, y_max)
          # Add hotbar for hit count
          sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=max_hits))
          sm.set_array([])
          cbar = fig.colorbar(sm, ax=group_ax, orientation='vertical', pad=0.02)
          cbar.set_label("Hit Count", rotation=90)
          plt.tight_layout()
          group_plot_path = os.path.join(group_save_path, f"{group['label']}_plot.png")
          group_fig.savefig(group_plot_path)
          plt.close(group_fig)

          # Update x offset for the next group
          x_offset += len(group['detectors'])

      # Customize the overall plot
      ax.set_title("Detector Occupancy")
      ax.set_xlabel("Detector")
      ax.set_ylabel("Element")
      ax.set_xticks(x_ticks)
      ax.set_xticklabels(x_labels, rotation=90)
      ax.set_ylim(0, y_max)
      sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=max_hits))
      sm.set_array([])
      cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
      cbar.set_label("Hit Count", rotation=90)
      plt.tight_layout()

      # Save the overall plot
      overall_plot_path = os.path.join(save_directory, "overall_occupancy_plot.png")
      fig.savefig(overall_plot_path)
      plt.close(fig)
      print(f"[INFO] Saved overall plot to {overall_plot_path}")

if __name__ == "__main__":
    run_directory = r"/home/devin/Documents/Big_Data/run_005994"
    save_directory = r"/home/devin/Documents/Big_Data/QTracker_Station_Occupancies"
    max_files = 1  # For Debugging
    detector_plot = DetectorPlot()
    detector_plot.process_files(run_directory,save_directory, max_files=max_files)
    print("Plot saved as 'final_detector_plot.png' at: {save_directory}")
