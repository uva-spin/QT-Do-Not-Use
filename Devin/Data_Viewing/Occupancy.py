import os
import numpy as np
import tqdm
import uproot
import json
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import argparse
import pandas as pd

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
            detectorid = np.array(file["fAllHits.detectorID"].array(library="np")[event_number])
            
            elementid = np.array(file["fAllHits.elementID"].array(library="np")[event_number])
        return detectorid, elementid
    
    def process_files(self, run_directory, save_directory, max_files=None):
        """Process all events across the first `max_files` ROOT files in the given directory."""
        self.aggregated_detectorid = np.array([], dtype=int)
        self.aggregated_elementid = np.array([], dtype=int)
        
        # Dictionary to store accumulated hits per file
        self.file_hits_data = {}
        
        # List to store hits per event
        self.hits_per_event = []

        # Gather all ROOT files in the directory
        for root_path, _, files in os.walk(run_directory):
            for file in files:
                if file.endswith(".root"):
                    self.file_paths.append(os.path.join(root_path, file))

        if not self.file_paths:
            print("[DEBUG] No ROOT files found in the specified directory.")
            return

        if max_files is not None:
            self.file_paths = self.file_paths[:max_files]

        total_events = 0
        for file_path in self.file_paths:
            with uproot.open(file_path + ":save") as file:
                total_events += len(file["fAllHits.detectorID"].array(library="np"))

        with tqdm.tqdm(total=total_events, desc="Processing Events", unit="event") as pbar:
            for file_path in self.file_paths:
                file_detectorid = np.array([], dtype=int)
                file_elementid = np.array([], dtype=int)
                
                with uproot.open(file_path + ":save") as file:
                    num_events = len(file["fAllHits.detectorID"].array(library="np"))
                    for event_number in range(num_events):
                        detectorid, elementid = self.read_event(file_path, event_number)
                        if len(detectorid) > 0 and len(elementid) > 0:
                            file_detectorid = np.concatenate((file_detectorid, detectorid))
                            file_elementid = np.concatenate((file_elementid, elementid))
                            self.aggregated_detectorid = np.concatenate((self.aggregated_detectorid, detectorid))
                            self.aggregated_elementid = np.concatenate((self.aggregated_elementid, elementid))
                            
                            # Only count hits for events that have actual hits
                            hits_dict = self.accumulate_hits(detectorid, elementid)
                            total_hits = sum(hits_dict.values())
                            self.hits_per_event.append(total_hits)
                        
                        pbar.update(1)
                
                # Accumulate hits for this file
                file_hits = self.accumulate_hits(file_detectorid, file_elementid)
                self.file_hits_data[os.path.basename(file_path)] = file_hits

        # Debug message if no hits are found
        if len(self.aggregated_detectorid) == 0 or len(self.aggregated_elementid) == 0:
            print("[DEBUG] No hits found across all events in the ROOT files.")
            print("[DEBUG] Check if the ROOT files contain valid detector and element data.")
        else:
            print(f"[DEBUG] Aggregated {len(self.aggregated_detectorid)} detector hits and {len(self.aggregated_elementid)} element hits.")

        # Save accumulated hits data
        self.save_accumulated_hits(save_directory)
        
        # Create plots
        self.create_plots(save_directory)
        
        # Create hits per event histogram
        self.create_hits_histogram(save_directory)

    def save_accumulated_hits(self, save_directory):
        """Save accumulated hits data for all files and total hits in a readable format."""
        os.makedirs(save_directory, exist_ok=True)
        output_data_dir = os.path.join(save_directory, "output_data")
        os.makedirs(output_data_dir, exist_ok=True)
        
        # Save per-file hits data
        for filename, hits in self.file_hits_data.items():
            # Create DataFrame for this file
            data_rows = []
            for (det_id, elem_id), count in hits.items():
                # Find detector name and group
                detector_info = None
                group_name = None
                for group in self.detector_groups:
                    for det in group['detectors']:
                        if det['id'] == det_id:
                            detector_info = det
                            group_name = group['label']
                            break
                    if detector_info:
                        break
                
                if detector_info:
                    data_rows.append({
                        'Group': group_name,
                        'Detector_Name': detector_info['name'],
                        'Detector_ID': det_id,
                        'Element_ID': elem_id,
                        'Hits': count
                    })
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(data_rows)
            csv_filename = os.path.splitext(filename)[0] + '_hits.csv'
            csv_path = os.path.join(output_data_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            print(f"[INFO] Saved file hits data to {csv_path}")
        
        # Save total accumulated hits
        total_hits = self.accumulate_hits(self.aggregated_detectorid, self.aggregated_elementid)
        
        # Create DataFrame for total hits
        total_data_rows = []
        for (det_id, elem_id), count in total_hits.items():
            # Find detector name and group
            detector_info = None
            group_name = None
            for group in self.detector_groups:
                for det in group['detectors']:
                    if det['id'] == det_id:
                        detector_info = det
                        group_name = group['label']
                        break
                if detector_info:
                    break
            
            if detector_info:
                total_data_rows.append({
                    'Group': group_name,
                    'Detector_Name': detector_info['name'],
                    'Detector_ID': det_id,
                    'Element_ID': elem_id,
                    'Hits': count
                })
        
        # Create DataFrame and save to CSV
        total_df = pd.DataFrame(total_data_rows)
        
        # Sort the data to ensure consistent orientation
        total_df = total_df.sort_values(['Detector_ID', 'Element_ID'])
        
        # Save the data
        total_csv_path = os.path.join(save_directory, "total_hits_data.csv")
        total_df.to_csv(total_csv_path, index=False)
        print(f"[INFO] Saved total hits data to {total_csv_path}")
        
        # Also save a summary by detector
        summary_rows = []
        for group in self.detector_groups:
            for det in group['detectors']:
                det_hits = sum(count for (d_id, _), count in total_hits.items() if d_id == det['id'])
                summary_rows.append({
                    'Group': group['label'],
                    'Detector_Name': det['name'],
                    'Detector_ID': det['id'],
                    'Total_Hits': det_hits,
                    'Elements': det['elements']
                })
        
        # Save summary to CSV
        summary_df = pd.DataFrame(summary_rows)
        # Sort summary by detector ID for consistency
        summary_df = summary_df.sort_values('Detector_ID')
        summary_csv_path = os.path.join(save_directory, "detector_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"[INFO] Saved detector summary to {summary_csv_path}")

        # Save the data in JSON format for process_hits_data.py
        # Create a 2D array representation of the hits
        max_detector = max(det_id for det_id, _ in total_hits.keys())
        max_element = max(elem_id for _, elem_id in total_hits.keys())
        
        # Initialize the array with zeros
        hits_array = np.zeros((max_detector + 1, max_element + 1), dtype=int)
        
        # Fill the array with hit counts
        for (det_id, elem_id), count in total_hits.items():
            hits_array[det_id, elem_id] = count
        
        # Convert to list and ensure all values are Python native types
        hits_list = [[int(x) for x in row] for row in hits_array.tolist()]
        
        # Save to JSON
        json_path = os.path.join(save_directory, "total_hits_data.json")
        with open(json_path, 'w') as f:
            json.dump({
                'hits_array': hits_list,
                'max_detector': int(max_detector),
                'max_element': int(max_element)
            }, f)
        print(f"[INFO] Saved total hits data to {json_path}")

    def load_accumulated_hits(self, file_path):
        """Load accumulated hits data from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert string keys back to tuples
        hits = {}
        for key, count in data.items():
            det_id, elem_id = map(int, key.split('_'))
            hits[(det_id, elem_id)] = count
        
        return hits

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

        os.makedirs(save_directory, exist_ok=True)
        output_data_dir = os.path.join(save_directory, "output_data")
        os.makedirs(output_data_dir, exist_ok=True)

        detectorid = np.array(self.aggregated_detectorid)
        elementid = np.array(self.aggregated_elementid)

        # Accumulate hits
        occupancy = self.accumulate_hits(detectorid, elementid)

        max_hits = max(occupancy.values()) if occupancy else 1
        norm = Normalize(vmin=0, vmax=max_hits)

        # Calculate total layout information
        y_max = max([det['elements'] for group in self.detector_groups for det in group['detectors']])
        
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
            group_fig, group_ax = plt.subplots(figsize=(12, 8))
            
            # Get range of x positions for this group
            group_end_idx = group_start_idx + len(group['detectors'])  # start index + number of detectors in group
            group_x_ticks = x_ticks[group_start_idx:group_end_idx]  # get the x ticks for this group
            group_x_labels = x_labels[group_start_idx:group_end_idx]  # get the x labels for this group
            
            group_box_data = [(x - group_start_idx, y, count) for x, y, count in all_box_data 
                            if group_start_idx <= x < group_end_idx]  # filter box data for this group
            
            self._plot_boxes(group_ax, group_box_data, norm)
            
            
            group_ax.set_title(f"{group['label']} Detector Occupancy", fontsize=14, pad=10)
            group_ax.set_xlabel("Detector", fontsize=12)
            group_ax.set_ylabel("Element", fontsize=12)
            group_ax.set_xticks([x - group_start_idx for x in group_x_ticks])  # Shift x ticks to start from 0
            group_ax.set_xticklabels(group_x_labels, rotation=90)
            group_ax.set_ylim(0, y_max)
            
            
            # Colorbar
            sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
            sm.set_array([])
            cbar = group_fig.colorbar(sm, ax=group_ax, pad=0.02)
            cbar.set_label("Hit Count", rotation=90, labelpad=15, fontsize=12)
            
            plt.tight_layout()

            os.makedirs(os.path.join(output_dir, group['label']), exist_ok=True)
            group_plot_path = os.path.join(output_dir, group['label'], f"{group['label'].replace(' ', '_')}_plot.png")
            group_fig.savefig(group_plot_path, dpi=150, bbox_inches='tight')
            plt.close(group_fig)
            print(f"[INFO] Saved group plot to {group_plot_path}")
            
            # Update for next group
            group_start_idx = group_end_idx

    def _create_overall_plots(self, all_box_data, x_labels, x_ticks, y_max, norm, max_hits, separators, save_dir):
        """Create overall plots with and without separators."""
        print("Creating overall plots...")
        
        # With separators between detector groups
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
        
        # Without separators
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
        
        sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Hit Count", rotation=90, labelpad=15, fontsize=14)
        
        plt.tight_layout()

    def create_hits_histogram(self, save_directory):
        """Create and save a histogram of hits per event."""
        if not self.hits_per_event:
            print("[DEBUG] No hits per event data available for histogram.")
            return

        plt.figure(figsize=(12, 6))


        print("Hits per event: ", self.hits_per_event)
        print("self.hits_per_event type: ", type(self.hits_per_event))
        print("self.hits_per_event shape: ", np.array(self.hits_per_event).shape)

        # Create histogram
        plt.hist(self.hits_per_event, bins=50, alpha=0.75, color='blue', edgecolor='black')
        
        # Add labels and title
        plt.xlabel('Number of Hits per Event', fontsize=12)
        plt.ylabel('Number of Events', fontsize=12)
        plt.title('Distribution of Hits per Event', fontsize=14, pad=15)
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        
        # Calculate and display statistics
        mean_hits = np.mean(self.hits_per_event)
        median_hits = np.median(self.hits_per_event)
        std_hits = np.std(self.hits_per_event)
        
        stats_text = f'Mean: {mean_hits:.1f}\nMedian: {median_hits:.1f}\nStd Dev: {std_hits:.1f}'
        plt.text(0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save the plot
        histogram_path = os.path.join(save_directory, "hits_per_event_histogram.png")
        plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved hits per event histogram to {histogram_path}")
        
        # Save the raw data
        data_path = os.path.join(save_directory, "hits_per_event_data.csv")
        np.savetxt(data_path, self.hits_per_event, delimiter=',', header='hits_per_event', comments='')
        print(f"[INFO] Saved hits per event data to {data_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detector Hit Display')
    parser.add_argument('--directory', type=str, default=".", help='Initial directory to monitor')
    args = parser.parse_args()
    run_directory = args.directory
    save_directory = r"QTracker_Station_Occupancies"
    max_files = 1  # For Debugging
    detector_plot = DetectorPlot()
    detector_plot.process_files(run_directory, save_directory, max_files=max_files)